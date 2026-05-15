"""src/api/external_v1.py の単体テスト"""
import os
from unittest.mock import patch

import pytest

from src.prediction.types import PredictionResult
from src.reporting.types import MarketPredictionSnapshot


def _make_prediction(symbol: str = "AAPL", diff_ratio: float = 0.02) -> PredictionResult:
    return PredictionResult(
        market="us",
        symbol=symbol,
        current_price=100.0,
        avg_pred_price=102.0,
        diff_ratio=diff_ratio,
        model_count=3,
        confidence_ratio=0.9,
        model_version="production",
    )


@pytest.fixture()
def client():
    from src.api.health import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestExternalV1Auth:
    def test_missing_api_key_returns_401(self, client):
        resp = client.get("/api/v1/predictions/top")
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "Unauthorized"

    def test_invalid_api_key_returns_401(self, client):
        with patch("src.api.external_v1._load_api_keys", return_value=frozenset(["valid-key"])):
            resp = client.get("/api/v1/predictions/top", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401

    def test_empty_api_key_env_returns_401(self, client):
        with patch("src.api.external_v1._load_api_keys", return_value=frozenset()):
            resp = client.get("/api/v1/predictions/top", headers={"X-API-Key": "any-key"})
        assert resp.status_code == 401

    def test_valid_api_key_passes_auth(self, client):
        snapshot = MarketPredictionSnapshot(
            market="us",
            top_results=[_make_prediction("AAPL")],
            worst_results=[_make_prediction("TSLA", -0.02)],
        )
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                return_value=("2026-05-11T14:00:00", [snapshot]),
            ),
        ):
            resp = client.get("/api/v1/predictions/top", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200


class TestExternalV1RateLimit:
    def test_rate_limited_returns_429(self, client):
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=True),
        ):
            resp = client.get("/api/v1/predictions/top", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 429
        assert resp.get_json()["error"] == "Too Many Requests"

    def test_not_rate_limited_under_threshold(self):
        from src.api.external_v1 import (
            _MAX_REQUESTS_PER_MINUTE,
            _is_rate_limited,
            _rate_lock,
            _request_log,
        )

        key = "__test_not_limited__"
        with _rate_lock:
            _request_log.pop(key, None)

        for _ in range(_MAX_REQUESTS_PER_MINUTE - 1):
            assert not _is_rate_limited(key)

    def test_rate_limited_at_threshold(self):
        import time

        from src.api.external_v1 import (
            _MAX_REQUESTS_PER_MINUTE,
            _is_rate_limited,
            _rate_lock,
            _request_log,
        )

        key = "__test_at_limit__"
        now = time.monotonic()
        with _rate_lock:
            _request_log[key] = [now] * _MAX_REQUESTS_PER_MINUTE

        assert _is_rate_limited(key)


class TestExternalV1PredictionsTop:
    def _get(self, client, market: str | None = None):
        url = "/api/v1/predictions/top"
        if market:
            url += f"?market={market}"
        return client.get(url, headers={"X-API-Key": "test-key"})

    def test_no_predictions_returns_empty(self, client):
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                return_value=(None, []),
            ),
        ):
            resp = self._get(client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["markets"] == []
        assert data["predicted_at"] is None

    def test_returns_top_and_worst_per_market(self, client):
        snapshot = MarketPredictionSnapshot(
            market="us",
            top_results=[_make_prediction("AAPL", 0.03)],
            worst_results=[_make_prediction("TSLA", -0.03)],
        )
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                return_value=("2026-05-11T14:00:00", [snapshot]),
            ),
        ):
            resp = self._get(client)
        data = resp.get_json()
        assert data["predicted_at"] == "2026-05-11T14:00:00"
        assert len(data["markets"]) == 1
        market_data = data["markets"][0]
        assert market_data["market"] == "us"
        assert len(market_data["top"]) == 1
        assert len(market_data["worst"]) == 1

    def test_public_fields_only_no_internal_fields(self, client):
        """内部フィールド（model_count, confidence_ratio, model_version）が除外される"""
        pred = _make_prediction("AAPL")
        snapshot = MarketPredictionSnapshot(market="us", top_results=[pred], worst_results=[])
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                return_value=("2026-05-11T14:00:00", [snapshot]),
            ),
        ):
            resp = self._get(client)
        top_item = resp.get_json()["markets"][0]["top"][0]
        assert "model_count" not in top_item
        assert "confidence_ratio" not in top_item
        assert "model_version" not in top_item
        assert "symbol" in top_item
        assert "diff_ratio" in top_item
        assert "current_price" in top_item

    def test_market_filter_excludes_other_markets(self, client):
        snapshots = [
            MarketPredictionSnapshot(
                market="jp",
                top_results=[_make_prediction("7203.T")],
                worst_results=[],
            ),
            MarketPredictionSnapshot(
                market="us",
                top_results=[_make_prediction("AAPL")],
                worst_results=[],
            ),
        ]
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                return_value=("2026-05-11T14:00:00", snapshots),
            ),
        ):
            resp = self._get(client, market="us")
        data = resp.get_json()
        assert len(data["markets"]) == 1
        assert data["markets"][0]["market"] == "us"

    def test_no_market_filter_returns_all_markets(self, client):
        snapshots = [
            MarketPredictionSnapshot(
                market="jp", top_results=[_make_prediction("7203.T")], worst_results=[]
            ),
            MarketPredictionSnapshot(
                market="us", top_results=[_make_prediction("AAPL")], worst_results=[]
            ),
        ]
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                return_value=("2026-05-11T14:00:00", snapshots),
            ),
        ):
            resp = self._get(client)
        data = resp.get_json()
        assert len(data["markets"]) == 2

    def test_internal_error_returns_500(self, client):
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_latest_market_prediction_snapshots",
                side_effect=RuntimeError("DB障害"),
            ),
        ):
            resp = self._get(client)
        assert resp.status_code == 500
        assert resp.get_json()["error"] == "Internal Server Error"


class TestToPublicDict:
    def test_extracts_public_fields_only(self):
        from src.api.external_v1 import _to_public_dict

        pred = _make_prediction("AAPL")
        result = _to_public_dict(pred)
        assert result["symbol"] == "AAPL"
        assert result["market"] == "us"
        assert result["diff_ratio"] == pytest.approx(0.02)
        assert "model_count" not in result
        assert "model_version" not in result
        assert "confidence_ratio" not in result

    def test_none_for_missing_optional_fields(self):
        from src.api.external_v1 import _to_public_dict

        pred = _make_prediction("AAPL")
        result = _to_public_dict(pred)
        # pred_lower_10, pred_upper_90, confluence_score は未設定 → None
        assert result["pred_lower_10"] is None
        assert result["pred_upper_90"] is None
        assert result["confluence_score"] is None


# ---------------------------------------------------------------------------
# R-304: /external/v1/* エンドポイントのテスト
# ---------------------------------------------------------------------------

_EXTERNAL_KEY = "ext-test-key"
_EXTERNAL_KEY_ENV = {"EXTERNAL_API_KEY": _EXTERNAL_KEY}


class TestExternalV1Auth304:
    """EXTERNAL_API_KEY 未設定時に 503 が返ることを確認する。"""

    def test_predictions_latest_no_key_configured_returns_503(self, client):
        with patch.dict(os.environ, {"EXTERNAL_API_KEY": ""}, clear=False):
            resp = client.get("/external/v1/predictions/latest", headers={"X-API-Key": "any"})
        assert resp.status_code == 503

    def test_monthly_report_no_key_configured_returns_503(self, client):
        with patch.dict(os.environ, {"EXTERNAL_API_KEY": ""}, clear=False):
            resp = client.get("/external/v1/monthly-report", headers={"X-API-Key": "any"})
        assert resp.status_code == 503

    def test_symbols_no_key_configured_returns_503(self, client):
        with patch.dict(os.environ, {"EXTERNAL_API_KEY": ""}, clear=False):
            resp = client.get("/external/v1/symbols", headers={"X-API-Key": "any"})
        assert resp.status_code == 503

    def test_predictions_latest_wrong_key_returns_401(self, client):
        with patch.dict(os.environ, _EXTERNAL_KEY_ENV, clear=False):
            resp = client.get("/external/v1/predictions/latest", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_symbols_missing_header_returns_401(self, client):
        with patch.dict(os.environ, _EXTERNAL_KEY_ENV, clear=False):
            resp = client.get("/external/v1/symbols")
        assert resp.status_code == 401


class TestExternalV1PredictionsLatest:
    def _get(self, client, extra_env=None):
        env = {**_EXTERNAL_KEY_ENV, **(extra_env or {})}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
        ):
            return client.get(
                "/external/v1/predictions/latest", headers={"X-API-Key": _EXTERNAL_KEY}
            )

    def test_returns_predictions_list(self, client):
        mock_preds = [
            {"market": "us", "symbol": "AAPL", "diff_ratio": 0.02, "prediction_date": "2026-05-15"}
        ]
        with patch("src.api.external_data_service.get_public_predictions", return_value=mock_preds):
            resp = self._get(client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predictions" in data
        assert data["count"] == 1
        assert data["predictions"][0]["symbol"] == "AAPL"

    def test_empty_predictions(self, client):
        with patch("src.api.external_data_service.get_public_predictions", return_value=[]):
            resp = self._get(client)
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 0

    def test_no_internal_fields_in_response(self, client):
        mock_preds = [
            {"market": "us", "symbol": "AAPL", "diff_ratio": 0.02, "prediction_date": "2026-05-15"}
        ]
        with patch("src.api.external_data_service.get_public_predictions", return_value=mock_preds):
            resp = self._get(client)
        item = resp.get_json()["predictions"][0]
        assert "current_price" not in item
        assert "avg_pred_price" not in item
        assert "model_count" not in item


class TestExternalV1MonthlyReport:
    def _get(self, client):
        with (
            patch.dict(os.environ, _EXTERNAL_KEY_ENV, clear=False),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
        ):
            return client.get("/external/v1/monthly-report", headers={"X-API-Key": _EXTERNAL_KEY})

    def test_returns_kpi_subset(self, client):
        mock_report = {
            "target_month": "2026-05",
            "net_return": 0.05,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.2,
            "hit_rate": 0.6,
        }
        with patch(
            "src.api.external_data_service.get_public_monthly_report", return_value=mock_report
        ):
            resp = self._get(client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["target_month"] == "2026-05"
        assert data["sharpe_ratio"] == pytest.approx(1.2)

    def test_no_internal_fields_in_response(self, client):
        mock_report = {
            "target_month": "2026-05",
            "net_return": 0.05,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.2,
            "hit_rate": 0.6,
        }
        with patch(
            "src.api.external_data_service.get_public_monthly_report", return_value=mock_report
        ):
            resp = self._get(client)
        data = resp.get_json()
        assert "avg_slippage" not in data
        assert "wf_snapshot_file" not in data
        assert "symbol_count" not in data

    def test_report_unavailable_returns_503(self, client):
        with patch("src.api.external_data_service.get_public_monthly_report", return_value=None):
            resp = self._get(client)
        assert resp.status_code == 503


class TestExternalV1Symbols:
    def _get(self, client):
        with (
            patch.dict(os.environ, _EXTERNAL_KEY_ENV, clear=False),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
        ):
            return client.get("/external/v1/symbols", headers={"X-API-Key": _EXTERNAL_KEY})

    def test_returns_symbol_list(self, client):
        mock_symbols = [
            {"market": "us", "symbol": "AAPL"},
            {"market": "jp", "symbol": "7203.T"},
        ]
        with patch("src.api.external_data_service.get_public_symbols", return_value=mock_symbols):
            resp = self._get(client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert data["symbols"][0]["symbol"] == "AAPL"

    def test_empty_symbols(self, client):
        with patch("src.api.external_data_service.get_public_symbols", return_value=[]):
            resp = self._get(client)
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 0

    def test_rate_limited_returns_429(self, client):
        with (
            patch.dict(os.environ, _EXTERNAL_KEY_ENV, clear=False),
            patch("src.api.external_v1._is_rate_limited", return_value=True),
        ):
            resp = client.get("/external/v1/symbols", headers={"X-API-Key": _EXTERNAL_KEY})
        assert resp.status_code == 429


class TestExternalDataService:
    """external_data_service のユニットテスト（サービス層）。"""

    def test_get_public_predictions_fields(self):
        import pandas as pd

        from src.api.external_data_service import get_public_predictions

        mock_df = pd.DataFrame(
            [
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "current_price": 150.0,
                    "avg_pred_price": 153.0,
                    "diff_ratio": 0.02,
                    "model_count": 3,
                }
            ]
        )
        with (
            patch(
                "src.utils.db.load_latest_prediction_timestamp",
                return_value="2026-05-15T10:00:00",
            ),
            patch("src.utils.db.load_prediction_results", return_value=mock_df),
        ):
            results = get_public_predictions()

        assert len(results) == 1
        item = results[0]
        assert item["market"] == "us"
        assert item["symbol"] == "AAPL"
        assert "diff_ratio" in item
        assert "prediction_date" in item
        # 内部フィールドが含まれないこと
        assert "current_price" not in item
        assert "avg_pred_price" not in item
        assert "model_count" not in item

    def test_get_public_predictions_no_timestamp(self):
        from src.api.external_data_service import get_public_predictions

        with patch("src.utils.db.load_latest_prediction_timestamp", return_value=None):
            results = get_public_predictions()

        assert results == []

    def test_get_public_symbols_returns_market_symbol(self):
        from src.api.external_data_service import get_public_symbols

        with patch("src.utils.db.get_all_symbols", return_value=[("us", "AAPL"), ("jp", "7203.T")]):
            results = get_public_symbols()

        assert len(results) == 2
        assert results[0] == {"market": "us", "symbol": "AAPL"}
        assert results[1] == {"market": "jp", "symbol": "7203.T"}
