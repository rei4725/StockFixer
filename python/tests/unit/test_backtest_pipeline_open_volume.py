"""
Unit Test: load_features が Backtester に生の Open・Volume・Close を渡すこと (#803)

翌バー始値の約定 (#493) と動的スリッページ (#494) は、Backtester に渡す表に
Open・Volume が無いと Close 約定・フラットのみに落ちる。api・raw・file の
各経路の出力に生の値が載り、Backtester がそれで約定することを確かめる。
"""

from typing import Any, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.data_port import set_backtest_data_port
from src.market_data.technical import create_basic_lag_features

_N = 40


def _raw_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=_N, freq="B", name="Date")
    close = 100.0 + np.arange(_N, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 10_000.0 + np.arange(_N, dtype=float) * 100.0,
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
    return _raw_ohlcv()


class TestLoadFeaturesHasRawOpenVolume:
    """各経路の出力に生の Open・Volume・Close が載ること"""

    @pytest.mark.parametrize("source", ["api", "raw"])
    def test_api_and_raw_have_open_volume(self, source, raw):
        set_backtest_data_port(_FakeDataPort(raw))

        from src.backtest.pipeline import load_features

        df = load_features("jp", "7203", source)

        assert "Open" in df.columns
        assert "Volume" in df.columns
        pd.testing.assert_series_equal(df["Open"], raw["Open"].reindex(df.index), check_names=False)
        pd.testing.assert_series_equal(
            df["Volume"], raw["Volume"].reindex(df.index), check_names=False
        )
        pd.testing.assert_series_equal(
            df["Close"], raw["Close"].reindex(df.index), check_names=False
        )

    def test_file_has_raw_open_volume_close(self, raw, monkeypatch):
        set_backtest_data_port(_FakeDataPort(raw))
        monkeypatch.setattr(
            "src.utils.db.load_stock_features", lambda m, s: _stock_features_frame(raw)
        )

        from src.backtest.pipeline import load_features

        df = load_features("jp", "7203", "file")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert "Open" in df.columns
        assert "Volume" in df.columns
        expected = raw.reindex(df.index)
        np.testing.assert_allclose(df["Open"].to_numpy(), expected["Open"].to_numpy())
        np.testing.assert_allclose(df["Volume"].to_numpy(), expected["Volume"].to_numpy())
        # Close は Close_lag1（前の日の終値）ではなく、その日の生の終値
        np.testing.assert_allclose(df["Close"].to_numpy(), expected["Close"].to_numpy())

    def test_file_without_raw_keeps_lag1_close(self, raw, monkeypatch):
        """market_data_raw が無ければ今の動き（Close_lag1 で代替）のまま"""
        set_backtest_data_port(_FakeDataPort(None))
        monkeypatch.setattr(
            "src.utils.db.load_stock_features", lambda m, s: _stock_features_frame(raw)
        )

        from src.backtest.pipeline import load_features

        df = load_features("jp", "7203", "file")

        assert "Open" not in df.columns
        assert "Volume" not in df.columns
        pd.testing.assert_series_equal(df["Close"], df["Close_lag1"], check_names=False)


class TestOpenVolumeNotUsedAsFeatures:
    """Open・Volume は約定とスリッページ用で、モデルの特徴量に入らないこと"""

    @patch("src.backtest.pipeline.runner.load_features")
    @patch("src.backtest.pipeline.runner.get_model_manager")
    @patch("src.backtest.backtester.Backtester")
    def test_run_backtest_single_excludes_open_volume(self, mock_bt, mock_get_mm, mock_lf):
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        mock_lf.return_value = pd.DataFrame(
            {
                "feat1": np.linspace(0.0, 1.0, n),
                "y": np.linspace(-0.01, 0.01, n),
                "Close": 100.0 + np.arange(n, dtype=float),
                "Open": 99.5 + np.arange(n, dtype=float),
                "Volume": np.full(n, 10_000.0),
            },
            index=dates,
        )
        mock_mm = mock_get_mm.return_value
        mock_mm.predict_with_model.return_value = [0.01] * (n - int(n * 0.8))
        mock_bt.return_value.simulate_trading.return_value = (pd.DataFrame(), {})

        from src.backtest.pipeline import run_backtest_single

        run_backtest_single("jp", "7203")

        X_train = mock_mm.train_model.call_args[0][1]
        assert list(X_train.columns) == ["feat1"]
        sim_df = mock_bt.return_value.simulate_trading.call_args[0][0]
        assert {"Open", "Volume"} <= set(sim_df.columns)


class TestBacktesterUsesOpenAndVolume:
    """file 経路の出力で Backtester が翌バー始値で約定し、平均出来高が0でないこと"""

    def test_simulate_trading_fills_at_open_with_volume(self, raw, monkeypatch):
        set_backtest_data_port(_FakeDataPort(raw))
        monkeypatch.setattr(
            "src.utils.db.load_stock_features", lambda m, s: _stock_features_frame(raw)
        )

        from src.backtest.backtester import Backtester
        from src.backtest.pipeline import load_features

        df = load_features("jp", "7203", "file")

        seen_volumes: list[float] = []

        def _slippage_fn(qty: int, price: float, avg_daily_volume: float) -> float:
            seen_volumes.append(avg_daily_volume)
            return 0.0

        signal = pd.Series(0, index=df.index)
        signal.iloc[2] = 1
        bt = Backtester(
            model_manager=None,
            signal_generator=None,
            data_loader=None,
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            slippage_fn=_slippage_fn,
        )
        result_df, _ = bt.simulate_trading(df, signal)

        buys = result_df[result_df["action"] == "buy"]
        assert len(buys) == 1
        fill_date = df.index[3]  # 信号の出たバーの次のバー
        assert buys["date"].iloc[0] == fill_date
        assert buys["price"].iloc[0] == pytest.approx(raw.loc[fill_date, "Open"])
        assert buys["price"].iloc[0] != pytest.approx(df.loc[fill_date, "Close"])
        assert seen_volumes
        assert all(v > 0 for v in seen_volumes)
