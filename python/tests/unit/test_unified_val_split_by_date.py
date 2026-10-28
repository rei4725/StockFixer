"""統合モデルの学習で、検証を直近の日付の行にし、どの銘柄も学習に入ることのテスト（#1363）。"""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

_N_SYMBOLS = 4
_N_DAYS = 400


def _stock_features_like() -> pd.DataFrame:
    """stock_features と同じく market・symbol 順に並んだ、4銘柄 × 400日の表。"""
    dates = pd.date_range("2024-01-01", periods=_N_DAYS, freq="B")
    rng = np.random.default_rng(0)
    frames = []
    for code in range(_N_SYMBOLS):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "market": "us",
                    "symbol": f"S{code}",
                    "sym_code": float(code),
                    "day": np.arange(_N_DAYS, dtype=float),
                    "close_lag1": rng.uniform(99, 101, _N_DAYS),
                    "y": rng.uniform(-0.05, 0.05, _N_DAYS),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _assert_split_by_date(mock_model: MagicMock) -> None:
    args, kwargs = mock_model.train.call_args
    X_train = args[0]
    eval_set = kwargs["eval_set"]
    assert eval_set is not None
    X_val = eval_set[0][0]
    assert len(X_train) + len(X_val) == _N_SYMBOLS * _N_DAYS
    assert X_val["day"].min() > X_train["day"].max()
    assert set(X_train["sym_code"]) == set(float(c) for c in range(_N_SYMBOLS))


@patch("src.prediction.unified_model_pipeline._compute_and_save_unified_feature_selection")
@patch("src.prediction.unified_model_pipeline.load_excluded_features", return_value=[])
@patch("src.prediction.manager.ModelManager")
@patch("src.prediction.unified_model_pipeline.load_all_stock_data")
def test_validation_rows_are_latest_dates_and_every_symbol_is_trained(
    mock_load, mock_mm_cls, _mock_excluded, _mock_compute
):
    """検証の行の日付はどれも学習の行の日付より後で、どの銘柄も学習に入る。"""
    from src.prediction.unified_model_pipeline import train_unified_model

    mock_load.return_value = _stock_features_like()
    mock_model = mock_mm_cls.return_value.create_model.return_value

    with tempfile.TemporaryDirectory() as tmp:
        train_unified_model(model_type="XGBoostModel", model_name="T", save_dir=tmp)

    _assert_split_by_date(mock_model)


def test_same_date_rows_do_not_straddle_train_and_validation():
    """切れ目が日付の途中に来ても、その日付の行はすべて検証に入る。"""
    from src.prediction.unified_model_pipeline import _sort_by_date, _split_train_val_by_date

    n_days = 600
    dates = pd.Series(np.tile(pd.date_range("2024-01-01", periods=n_days, freq="B"), 3))
    X = pd.DataFrame({"day": np.tile(np.arange(n_days, dtype=float), 3)})
    y = pd.Series(np.zeros(len(X)))

    X, y, day = _sort_by_date(X, y, dates)
    X_train, _, eval_set = _split_train_val_by_date(X, y, day)

    X_val = eval_set[0][0]
    assert len(X_train) == 433 * 3
    assert X_val["day"].min() == 433.0
    assert not set(X_train["day"]) & set(X_val["day"])


def test_split_without_dates_keeps_row_order():
    """日付が無いときは、今までどおり行の並びの末尾を検証にする。"""
    from src.prediction.unified_model_pipeline import _sort_by_date, _split_train_val_by_date

    X = pd.DataFrame({"v": np.arange(2000, dtype=float)})
    y = pd.Series(np.zeros(2000))

    X, y, day = _sort_by_date(X, y, None)
    X_train, _, eval_set = _split_train_val_by_date(X, y, day)

    assert len(X_train) == 1500
    assert eval_set[0][0]["v"].min() == 1500.0


@patch("src.prediction.unified_model_pipeline._compute_and_save_unified_feature_selection")
@patch("src.prediction.unified_model_pipeline.load_excluded_features", return_value=[])
@patch("src.prediction.manager.ModelManager")
@patch("src.prediction.unified_model_pipeline._load_sentiment_lookup", return_value={})
@patch("src.prediction.unified_model_pipeline.get_market_data_port")
@patch("src.utils.db.load_raw_ohlcv")
@patch("src.utils.db.load_all_raw_ohlcv_symbols")
def test_multi_horizon_validation_rows_are_latest_dates(
    mock_symbols, mock_raw, mock_port, _mock_lookup, mock_mm_cls, _mock_excluded, _mock_compute
):
    """多ホライズンの道でも、検証は直近の日付の行で、どの銘柄も学習に入る。"""
    from src.prediction.unified_model_pipeline import train_unified_model

    table = _stock_features_like()
    mock_symbols.return_value = [("us", f"S{c}") for c in range(_N_SYMBOLS)]
    mock_raw.side_effect = lambda market, symbol: table[table["symbol"] == symbol]

    def _lag_features(df, target_horizon):
        X_sym = df[["sym_code", "day", "close_lag1"]].set_index(df["date"])
        return X_sym, pd.Series(df["y"].to_numpy(), index=X_sym.index)

    mock_mds = MagicMock()
    mock_mds.add_technical_indicators.side_effect = lambda raw: raw
    mock_mds.create_basic_lag_features.side_effect = _lag_features
    mock_port.return_value = mock_mds
    mock_model = mock_mm_cls.return_value.create_model.return_value

    with tempfile.TemporaryDirectory() as tmp:
        train_unified_model(model_type="XGBoostModel", model_name="T_3d", save_dir=tmp, horizon=3)

    _assert_split_by_date(mock_model)
