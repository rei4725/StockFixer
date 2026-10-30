"""
Unit Test: load_features が Backtester に生の High・Low を渡すこと (#1424)

指値の約定の判定 (#1418) は、値幅と寄付の指値の届きを High・Low で数える。
週の最適化の道（load_features → WalkForwardValidator）の表に High・Low が無いと、
値幅だけで寄付の指値になる銘柄が翌バーの始値で約定したものとして数えられる。
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from src.backtest.data_port import set_backtest_data_port
from src.backtest.task import ReturnRegressionTask
from src.backtest.walk_forward import WalkForwardValidator
from src.market_data.technical import create_basic_lag_features

_N = 60


def _wide_range_ohlcv() -> pd.DataFrame:
    """出来高は多いが値幅が広く、毎日上に窓を開けて寄付の指値に届かない仮の銘柄。

    終値は毎日 20 円上がり、安値は終値の 5 円下。前の日の終値 + 余裕の幅の指値に、
    翌バーの安値（前の日の終値 + 15 円）が届かない。値幅 (High - Low) / Close は約 3.5%。
    """
    dates = pd.date_range("2024-01-01", periods=_N, freq="B", name="Date")
    close = 1000.0 + np.arange(_N, dtype=float) * 20.0
    return pd.DataFrame(
        {
            "Open": close - 2.0,
            "High": close + 30.0,
            "Low": close - 5.0,
            "Close": close,
            "Volume": np.full(_N, 2_000_000.0),
        },
        index=dates,
    )


class _FakeDataPort:
    """生の OHLCV を返し、指標は atr だけ足す偽のデータポート。"""

    def __init__(self, raw: Optional[pd.DataFrame]) -> None:
        self.raw = raw

    def get_stock_data(self, market: str, ticker: str, start: Any, end: Any) -> Any:
        return None if self.raw is None else self.raw.copy()

    def get_raw_ohlcv_from_db(
        self, market: str, symbol: str, start: Any = None, end: Any = None
    ) -> Any:
        return None if self.raw is None else self.raw.copy()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = df["High"] - df["Low"]
        return df

    def create_basic_lag_features(self, df: pd.DataFrame, n_lags: int = 10) -> Any:
        return create_basic_lag_features(df, n_lags=n_lags)


def _stock_features_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """stock_features の形（生の価格は無く、ラグと date と y だけ）を作る。"""
    X, y = create_basic_lag_features(raw, n_lags=10)
    out = X.copy()
    out["y"] = y
    out.insert(0, "date", X.index.strftime("%Y-%m-%d"))
    return out.reset_index(drop=True)


@pytest.fixture
def raw() -> pd.DataFrame:
    return _wide_range_ohlcv()


def _load(source: str, raw: pd.DataFrame, monkeypatch) -> pd.DataFrame:
    set_backtest_data_port(_FakeDataPort(raw))
    monkeypatch.setattr("src.utils.db.load_stock_features", lambda m, s: _stock_features_frame(raw))

    from src.backtest.pipeline import load_features

    return load_features("jp", "7203", source)


class TestLoadFeaturesHasRawHighLow:
    """各経路の出力に生の High・Low が載り、特徴量に入らないこと"""

    @pytest.mark.parametrize("source", ["file", "api", "raw"])
    def test_has_raw_high_low(self, source, raw, monkeypatch):
        df = _load(source, raw, monkeypatch)

        assert isinstance(df.index, pd.DatetimeIndex)
        expected = raw.reindex(df.index)
        for col in ("High", "Low"):
            assert col in df.columns
            np.testing.assert_allclose(df[col].to_numpy(), expected[col].to_numpy())

    @pytest.mark.parametrize("source", ["file", "api", "raw"])
    def test_high_low_not_features(self, source, raw, monkeypatch):
        from src.backtest.pipeline.features import NON_FEATURE_COLUMNS

        df = _load(source, raw, monkeypatch)

        features = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        assert "High" not in features
        assert "Low" not in features
        # 前の日までの高値・安値のラグは今までどおり特徴量に残る
        assert "High_lag1" in features
        assert "Low_lag1" in features

    def test_file_missing_day_does_not_look_ahead(self, raw, monkeypatch):
        """market_data_raw に無い日は、その日の Close で埋める（先の日の値を使わない）"""
        missing = raw.index[30]
        set_backtest_data_port(_FakeDataPort(raw.drop(index=missing)))
        monkeypatch.setattr(
            "src.utils.db.load_stock_features", lambda m, s: _stock_features_frame(raw)
        )

        from src.backtest.pipeline import load_features

        df = load_features("jp", "7203", "file")

        assert df.loc[missing, "High"] == pytest.approx(df.loc[missing, "Close"])
        assert df.loc[missing, "Low"] == pytest.approx(df.loc[missing, "Close"])
        assert not df[["High", "Low"]].isna().any().any()


class _AlwaysUpModel:
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.01)


class _FakeModelManager:
    def get_model(self, name: str) -> _AlwaysUpModel:
        return _AlwaysUpModel()


class TestWalkForwardAppliesOpenLimit:
    """週の最適化の道で、値幅だけが広い銘柄の買いは寄付の指値に届かない日に約定しない"""

    def test_fold_does_not_fill_buy_when_next_bar_misses_limit(self, raw, monkeypatch):
        df = _load("file", raw, monkeypatch)
        task = ReturnRegressionTask()
        df[task.label_col] = task.make_labels(df)
        df = df.dropna(subset=[task.label_col])
        half = len(df) // 2

        wfv = WalkForwardValidator(
            market="jp",
            symbol="7203",
            model_manager=_FakeModelManager(),
            signal_generator=None,
            initial_cash=10_000_000,
        )
        metrics = wfv._run_fold(
            train_df=df.iloc[:half],
            val_df=df.iloc[half:],
            model_name="fake",
            model_type=None,
            task=task,
        )

        assert metrics["num_trades"] == 0
