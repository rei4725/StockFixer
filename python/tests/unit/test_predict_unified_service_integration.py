"""predict_with_unified_model の推論サービス連携テスト。

サービスが使える場合はその結果を返し、使えない場合は従来のインプロセス推論に
フォールバックすることを検証する。
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prediction.predict_unified import predict_with_unified_model
from src.prediction.types import PredictionResult


@pytest.fixture
def _feature_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-28", "2026-07-29"]),
            "Close_lag1": [1000.0, 1010.0],
            "rsi_lag1": [50.0, 55.0],
            # 最新行(末尾)がNaN — pd.notna によるフィルタが数値限定と両立することを検証する
            "volume_ratio": [1.2, float("nan")],
            "y": [0.01, 0.02],
        }
    )


def _patch_common(feature_df):
    """特徴量読み込みと現在価格取得を固定するパッチ群を返す。"""
    return [
        patch("src.prediction.predict_unified.load_feature_data", return_value=feature_df),
        patch("src.prediction.predict_unified.load_model_weights", return_value=[0.5, 0.5]),
    ]


def _mock_model():
    model = MagicMock()
    inner = MagicMock()
    inner.feature_names_in_ = ["Close_lag1", "rsi_lag1"]
    model.model = inner
    model.predict.return_value = pd.Series([0.02])
    return model


def test_uses_service_result_when_available(_feature_df, monkeypatch):
    """サービスが結果を返した場合、インプロセス推論を行わずそれを返すこと。"""
    monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
    service_result = PredictionResult(
        market="jp",
        symbol="7203",
        current_price=1000.0,
        avg_pred_price=1050.0,
        diff_ratio=0.05,
        model_count=2,
    )

    patches = _patch_common(_feature_df)
    with patches[0], patches[1], patch(
        "src.prediction.predict_unified.predict_via_service", return_value=service_result
    ), patch("src.prediction.predict_unified.get_cached_model") as mock_get_model, patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    assert result is service_result
    mock_get_model.assert_not_called()


def test_falls_back_to_inprocess_when_service_returns_none(_feature_df, monkeypatch):
    """サービスが None を返した場合、従来のインプロセス推論が動くこと。"""
    monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
    model = _mock_model()

    patches = _patch_common(_feature_df)
    with patches[0], patches[1], patch(
        "src.prediction.predict_unified.predict_via_service", return_value=None
    ), patch("src.prediction.predict_unified.get_cached_model", return_value=model), patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    assert result is not None
    assert result.model_count > 0
    assert model.predict.called


def test_service_not_called_when_url_unset(_feature_df, monkeypatch):
    """URL 未設定時はサービス呼び出しも重み計算(DBクエリ)も行わないこと。

    既定パスで銘柄ごとに余計な DB クエリが増えることを防ぐための回帰テスト。
    """
    monkeypatch.delenv("PREDICTION_SERVICE_URL", raising=False)
    model = _mock_model()

    with patch("src.prediction.predict_unified.load_feature_data", return_value=_feature_df), patch(
        "src.prediction.predict_unified.load_model_weights", return_value=[1.0]
    ) as mock_weights, patch(
        "src.prediction.predict_unified.predict_via_service"
    ) as mock_service, patch(
        "src.prediction.predict_unified.get_cached_model", return_value=model
    ), patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    mock_service.assert_not_called()
    # 既存経路のループ後の1回だけ（サービス用の事前計算が走っていないこと）
    assert mock_weights.call_count == 1
    assert result is not None


def test_date_column_excluded_from_service_features(_feature_df, monkeypatch):
    """date列(Timestamp)が含まれていても例外にならず、数値列のみが送信されること。"""
    monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
    service_result = PredictionResult(
        market="jp",
        symbol="7203",
        current_price=1000.0,
        avg_pred_price=1050.0,
        diff_ratio=0.05,
        model_count=2,
    )

    patches = _patch_common(_feature_df)
    with patches[0], patches[1], patch(
        "src.prediction.predict_unified.predict_via_service", return_value=service_result
    ) as mock_predict_via_service, patch(
        "src.prediction.predict_unified.get_cached_model"
    ) as mock_get_model, patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    assert result is service_result
    mock_get_model.assert_not_called()
    sent_features = mock_predict_via_service.call_args.kwargs["features"]
    assert "date" not in sent_features
    assert all(isinstance(v, float) for v in sent_features.values())


def test_service_exception_falls_back_to_inprocess(_feature_df, monkeypatch):
    """predict_via_service がペイロード構築等で例外を投げても、インプロセス推論に
    フォールバックすること（クラッシュせず全銘柄予測ゼロにもならない）。
    """
    monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
    model = _mock_model()

    patches = _patch_common(_feature_df)
    with patches[0], patches[1], patch(
        "src.prediction.predict_unified.predict_via_service",
        side_effect=RuntimeError("boom"),
    ), patch("src.prediction.predict_unified.get_cached_model", return_value=model), patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    assert result is not None
    assert result.model_count > 0
    assert model.predict.called
