"""src/api/external_v1.py の単体テスト"""
from unittest.mock import patch

import pytest

from src.prediction.types import PredictionResult
from src.reporting.types import MarketPredictionSnapshot, MonthlyReportSummary


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


def _make_monthly_summary(month: str = "2026-04") -> MonthlyReportSummary:
    return MonthlyReportSummary(
        generated_at="2026-05-01T00:00:00",
        target_month=month,
        net_return=0.05,
        max_drawdown=-0.10,
        sharpe_ratio=1.2,
        hit_rate=0.58,
        avg_slippage=0.003,
        wf_snapshot_file="wf_summary_2026-04.csv",
        symbol_count=30,
    )


class TestExternalV1ReportsMonthly:
    def _get(self, client, month: str | None = None):
        url = "/api/v1/reports/monthly"
        if month:
            url += f"?month={month}"
        return client.get(url, headers={"X-API-Key": "test-key"})

    def test_missing_api_key_returns_401(self, client):
        resp = client.get("/api/v1/reports/monthly")
        assert resp.status_code == 401

    def test_rate_limited_returns_429(self, client):
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=True),
        ):
            resp = self._get(client)
        assert resp.status_code == 429

    def test_returns_public_fields_only(self, client):
        summary = _make_monthly_summary()
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_monthly_report_summary",
                return_value=summary,
            ),
        ):
            resp = self._get(client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["target_month"] == "2026-04"
        assert data["net_return"] == pytest.approx(0.05)
        assert data["sharpe_ratio"] == pytest.approx(1.2)
        assert data["hit_rate"] == pytest.approx(0.58)
        assert data["symbol_count"] == 30
        # 内部フィールドが除外されている
        assert "avg_slippage" not in data
        assert "wf_snapshot_file" not in data

    def test_month_param_is_forwarded(self, client):
        summary = _make_monthly_summary("2026-03")
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_monthly_report_summary",
                return_value=summary,
            ) as mock_get,
        ):
            resp = self._get(client, month="2026-03")
        assert resp.status_code == 200
        mock_get.assert_called_once_with(target_month="2026-03")

    def test_internal_error_returns_500(self, client):
        with (
            patch("src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])),
            patch("src.api.external_v1._is_rate_limited", return_value=False),
            patch(
                "src.reporting.query_service.get_monthly_report_summary",
                side_effect=RuntimeError("DB障害"),
            ),
        ):
            resp = self._get(client)
        assert resp.status_code == 500
        assert resp.get_json()["error"] == "Internal Server Error"


class TestToPublicMonthlyDict:
    def test_extracts_public_fields_only(self):
        from src.api.external_v1 import _to_public_monthly_dict

        summary = _make_monthly_summary()
        result = _to_public_monthly_dict(summary)
        assert result["target_month"] == "2026-04"
        assert result["net_return"] == pytest.approx(0.05)
        assert result["symbol_count"] == 30
        assert "avg_slippage" not in result
        assert "wf_snapshot_file" not in result

    def test_none_for_missing_optional_fields(self):
        from src.api.external_v1 import _to_public_monthly_dict

        summary = MonthlyReportSummary(
            generated_at="2026-05-01T00:00:00",
            target_month="2026-04",
            net_return=None,
            max_drawdown=None,
            sharpe_ratio=None,
            hit_rate=None,
            avg_slippage=None,
            symbol_count=None,
        )
        result = _to_public_monthly_dict(summary)
        assert result["net_return"] is None
        assert result["sharpe_ratio"] is None
        assert result["hit_rate"] is None
