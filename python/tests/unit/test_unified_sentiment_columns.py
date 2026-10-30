"""統合モデルの学習と予想で、センチメントの列の扱いがそろうことのテスト（#1416）。

stock_features にあるセンチメントの列は ``sentiment_score_lag1`` などのラグの列だけで、
学習の道（market_data_raw → add_technical_indicators → create_basic_lag_features）は
センチメントの列を作らない。予想の道はモデルの持つ列にそろえるので、学習の X の列が
そのまま予想に渡る列になる。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.market_data import technical
from src.prediction.predict_unified import resolve_expected_features

_SENTIMENT_PREFIXES = ("sentiment_", "news_count")


def _raw_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1_000, 2_000, n).astype(float),
        },
        index=pd.date_range("2025-01-02", periods=n, freq="B"),
    )


def _stock_features_with_sentiment_lags(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"market": "jp", "symbol": "7203", "date": index}, index=index)
    for col in ("sentiment_score", "sentiment_ma5", "sentiment_momentum", "news_count"):
        for lag in range(1, 11):
            df[f"{col}_lag{lag}"] = 0.5
    return df


def _train_X() -> tuple[pd.DataFrame, MagicMock]:
    from src.prediction.unified_model_pipeline import _load_all_ohlcv_for_unified

    raw = _raw_ohlcv()
    port = MagicMock()
    port.add_technical_indicators.side_effect = technical.add_technical_indicators
    port.create_basic_lag_features.side_effect = (
        lambda df, target_horizon: technical.create_basic_lag_features(
            df, target_horizon=target_horizon
        )
    )
    with (
        patch(
            "src.prediction.unified_model_pipeline.load_all_stock_features",
            return_value=_stock_features_with_sentiment_lags(raw.index),
        ) as mock_sf,
        patch("src.prediction.unified_model_pipeline.get_market_data_port", return_value=port),
        patch("src.utils.db.load_raw_ohlcv", return_value=raw),
        patch("src.utils.db.load_all_raw_ohlcv_symbols", return_value=[("jp", "7203")]),
    ):
        X, _ = _load_all_ohlcv_for_unified(horizon=1)
    return X, mock_sf


def test_training_path_does_not_read_all_stock_features():
    """学習の道は stock_features の全行を読まない（センチメントを結合しない）。"""
    X, mock_sf = _train_X()

    assert not X.empty
    mock_sf.assert_not_called()


def test_training_X_has_no_sentiment_columns():
    """add_technical_indicators と create_basic_lag_features はセンチメントの列を作らない。"""
    X, _ = _train_X()

    assert [c for c in X.columns if c.startswith(_SENTIMENT_PREFIXES)] == []


def test_prediction_alignment_matches_training_columns():
    """予想の側は、センチメントのラグの列を持つ行からでも、学習の列の名前と順にそろう。"""
    X, _ = _train_X()
    model = MagicMock()
    model.model.feature_names_in_ = np.array(list(X.columns))

    expected = resolve_expected_features(model)
    pred_row = pd.concat([X.iloc[[-1]], _stock_features_with_sentiment_lags(X.index[-1:])], axis=1)
    aligned = pred_row.reindex(columns=expected, fill_value=0)

    assert expected == list(X.columns)
    assert list(aligned.columns) == list(X.columns)
