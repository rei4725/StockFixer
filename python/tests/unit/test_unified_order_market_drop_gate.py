"""発注する市場 jp の行が NaN で線を超えて落ちた表では、学習も保存もしないことのテスト（#1686）。"""

import logging
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction import unified_model_pipeline as ump
from src.prediction.manager import ModelManager

_N_ROWS = 100
_OLD_BYTES = b"production model"
_DATES = pd.bdate_range("2026-01-01", periods=_N_ROWS)


def _market_frame(market: str, nan_ratio: float, seed: int) -> pd.DataFrame:
    """1市場1銘柄の表。先頭の nan_ratio の割合の行の f2 を NaN にする。"""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "market": market,
            "symbol": f"{market}1",
            "date": _DATES,
            "f1": rng.normal(size=_N_ROWS),
            "f2": rng.normal(size=_N_ROWS),
        }
    )
    frame["y"] = frame["f1"] * 0.01
    frame.loc[frame.index[: int(_N_ROWS * nan_ratio)], "f2"] = np.nan
    return frame


def _table(jp_nan: float, us_nan: float = 0.0) -> pd.DataFrame:
    return pd.concat(
        [_market_frame("us", us_nan, 0), _market_frame("jp", jp_nan, 1)], ignore_index=True
    )


@pytest.fixture
def small_training(monkeypatch: pytest.MonkeyPatch) -> Callable[[pd.DataFrame], None]:
    """読み込みの結果を差し替え、木の少ないモデルで学習するようにする。"""
    monkeypatch.setattr(ump, "load_excluded_features", lambda *a, **k: set())
    monkeypatch.setattr(ump, "any_sentiment_source_configured", lambda markets: True)
    orig_create = ModelManager.create_model

    def small_create(self, model_type, model_name, **kwargs):
        return orig_create(self, model_type, model_name, n_estimators=5, **kwargs)

    monkeypatch.setattr(ModelManager, "create_model", small_create)

    def use_table(df: pd.DataFrame) -> None:
        monkeypatch.setattr(ump, "load_all_stock_data", lambda data_dir=None: df.copy())

    return use_table


def _saved_model(save_dir: Path) -> Path:
    """今の本番のファイルに見立てて、中身の分かるファイルを置く。"""
    path = save_dir / "T.joblib"
    path.write_bytes(_OLD_BYTES)
    return path


@pytest.mark.parametrize("jp_nan", [0.51, 1.0])
def test_raises_and_keeps_model_file_when_jp_rows_drop_over_line(
    small_training, tmp_path: Path, caplog: pytest.LogCaptureFixture, jp_nan: float
) -> None:
    """jp が線を超えて落ちたら投げ、置いてあるモデルのファイルを上書きしない。"""
    small_training(_table(jp_nan=jp_nan))
    saved = _saved_model(tmp_path)

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="jp"):
        ump.train_unified_model(model_name="T", save_dir=str(tmp_path), save_selection=False)

    assert saved.read_bytes() == _OLD_BYTES
    assert [p.name for p in tmp_path.iterdir()] == ["T.joblib"]
    # 落ちた行の WARNING は投げる前に今までどおり出る
    assert any("市場ごとの落ちた行" in r.getMessage() for r in caplog.records)


def test_does_not_create_a_model_file_when_jp_rows_drop_over_line(
    small_training, tmp_path: Path
) -> None:
    """モデルのファイルが無いときも、線を超えた表では何も保存しない。"""
    small_training(_table(jp_nan=1.0))

    with pytest.raises(RuntimeError, match="jp"):
        ump.train_unified_model(model_name="T", save_dir=str(tmp_path), save_selection=False)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("df", "label"),
    [
        (_table(jp_nan=0.2), "jp が線の下"),
        (_table(jp_nan=0.5), "jp がちょうど線の上"),
        (_market_frame("us", 0.0, 0), "jp の行が0件"),
        (_market_frame("us", 0.9, 0), "jp の行が0件で us は線を超えて落ちた"),
        (_table(jp_nan=0.0, us_nan=0.9), "us だけが線を超えて落ちた"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_trains_and_saves_when_jp_rows_are_not_over_line(
    small_training, tmp_path: Path, df: pd.DataFrame, label: str
) -> None:
    """線を超えていなければ、今までどおり学習して保存する。"""
    small_training(df)
    saved = _saved_model(tmp_path)

    ump.train_unified_model(model_name="T", save_dir=str(tmp_path), save_selection=False)

    assert saved.read_bytes() != _OLD_BYTES


def test_line_is_read_from_settings(
    small_training, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """線は DAILY_FETCH_MAX_UNAVAILABLE_RATIO の値に従う。"""
    monkeypatch.setattr(ump, "DAILY_FETCH_MAX_UNAVAILABLE_RATIO", 0.1)
    small_training(_table(jp_nan=0.2))

    with pytest.raises(RuntimeError, match="線 10%"):
        ump.train_unified_model(model_name="T", save_dir=str(tmp_path), save_selection=False)


def _train_multi_horizon(df: pd.DataFrame, save_dir: Path) -> None:
    """多ホライズンの道で学習する。OHLCV は市場ごとの表を、そのまま特徴量にして返す。"""

    def lag_features(raw: pd.DataFrame, target_horizon: int) -> tuple[pd.DataFrame, pd.Series]:
        X_sym = raw[["f1", "f2"]].set_index(raw["date"])
        return X_sym, pd.Series(raw["y"].to_numpy(), index=X_sym.index)

    mds = MagicMock()
    mds.add_technical_indicators.side_effect = lambda raw: raw
    mds.create_basic_lag_features.side_effect = lag_features
    symbols = [(m, f"{m}1") for m in df["market"].unique()]
    with (
        patch.object(ump, "get_market_data_port", return_value=mds),
        patch("src.utils.db.load_all_raw_ohlcv_symbols", return_value=symbols),
        patch("src.utils.db.load_raw_ohlcv", lambda m, s: df[df["market"] == m]),
    ):
        ump.train_unified_model(
            model_name="T", save_dir=str(save_dir), horizon=3, save_selection=False
        )


def test_multi_horizon_raises_when_jp_rows_drop_over_line(small_training, tmp_path: Path) -> None:
    """多ホライズンの道でも、jp が線を超えて落ちたら投げ、ファイルを上書きしない。"""
    saved = _saved_model(tmp_path)

    with pytest.raises(RuntimeError, match="jp"):
        _train_multi_horizon(_table(jp_nan=0.6), tmp_path)

    assert saved.read_bytes() == _OLD_BYTES


def test_multi_horizon_does_not_stop_when_only_us_drops(small_training, tmp_path: Path) -> None:
    """多ホライズンの道でも、us だけが線を超えて落ちた表では止めない。"""
    saved = _saved_model(tmp_path)

    _train_multi_horizon(_table(jp_nan=0.0, us_nan=0.9), tmp_path)

    assert saved.read_bytes() != _OLD_BYTES
