"""ユニットテスト: 引け前ポジション再評価アラートサービス（R-409）"""

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.in_memory import InMemoryPredictionRepository, InMemoryTradeDiffSink
from src.trading.pre_close_alert_service import (
    STOP_LOSS_PRED_THRESHOLD,
    TAKE_PROFIT_HOLD_THRESHOLD,
    AlertType,
    PositionAlert,
    evaluate_positions,
    get_pre_close_alerts,
)

# 注入する予測リポジトリ。テストは latest_timestamp / results_at を patch.object で差し替える
_PREDICTIONS = InMemoryPredictionRepository()


@pytest.fixture(autouse=True)
def _fixed_today():
    """既存のテストの予想の回（2026-01-01）が古いと見なされないよう、今日を固定する。"""
    with patch(
        "src.trading.pre_close_alert_service._today_jst",
        return_value=date(2026, 1, 1),
    ):
        yield


def _make_position(symbol: str, qty: int, avg_price: float, current_price: float) -> dict:
    unrealized_pnl = (current_price - avg_price) * qty
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_price": avg_price,
        "current_price": current_price,
        "unrealized_pnl": unrealized_pnl,
    }


def _make_pred_df(symbol: str, avg_pred_price: float, diff_ratio: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": symbol, "avg_pred_price": avg_pred_price, "diff_ratio": diff_ratio}]
    )


class TestAlertTypeClassification(unittest.TestCase):
    """evaluate_positions の分類ロジックをユニットテスト"""

    def _run_with_one_position(self, position: dict, pred_df: pd.DataFrame) -> PositionAlert:
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
            patch.object(
                _PREDICTIONS,
                "latest_timestamp",
                return_value="20260101_150000",
            ),
            patch.object(
                _PREDICTIONS,
                "results_at",
                return_value=pred_df,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = [position]
            mock_broker_cls.return_value = broker

            alerts = evaluate_positions(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )

        self.assertEqual(len(alerts), 1)
        return alerts[0]

    def test_stop_loss_when_diff_ratio_below_threshold(self):
        pos = _make_position("7203", 100, 1000.0, 980.0)
        pred = _make_pred_df(
            "7203", avg_pred_price=970.0, diff_ratio=STOP_LOSS_PRED_THRESHOLD - 0.001
        )
        alert = self._run_with_one_position(pos, pred)
        self.assertEqual(alert.alert_type, AlertType.STOP_LOSS)

    def test_take_profit_when_profitable_and_pred_declining(self):
        avg_price = 1000.0
        current_price = avg_price * (1 + TAKE_PROFIT_HOLD_THRESHOLD + 0.005)
        pos = _make_position("7203", 100, avg_price, current_price)
        pred = _make_pred_df("7203", avg_pred_price=current_price * 0.99, diff_ratio=-0.005)
        alert = self._run_with_one_position(pos, pred)
        self.assertEqual(alert.alert_type, AlertType.TAKE_PROFIT)

    def test_hold_when_pred_rising(self):
        pos = _make_position("7203", 100, 1000.0, 1010.0)
        pred = _make_pred_df("7203", avg_pred_price=1030.0, diff_ratio=0.02)
        alert = self._run_with_one_position(pos, pred)
        self.assertEqual(alert.alert_type, AlertType.HOLD)

    def test_hold_when_profitable_but_pred_also_rising(self):
        avg_price = 1000.0
        current_price = avg_price * (1 + TAKE_PROFIT_HOLD_THRESHOLD + 0.01)
        pos = _make_position("7203", 100, avg_price, current_price)
        pred = _make_pred_df("7203", avg_pred_price=current_price * 1.01, diff_ratio=0.01)
        alert = self._run_with_one_position(pos, pred)
        self.assertEqual(alert.alert_type, AlertType.HOLD)

    def test_position_missing_from_run_is_no_prediction_not_hold(self):
        pos = _make_position("9999", 50, 500.0, 510.0)
        pred = pd.DataFrame()  # no prediction data
        alert = self._run_with_one_position(pos, pred)
        self.assertEqual(alert.alert_type, AlertType.NO_PREDICTION)
        self.assertNotEqual(alert.alert_type, AlertType.HOLD)
        self.assertEqual(alert.diff_ratio, 0.0)
        self.assertEqual(alert.avg_pred_price, 510.0)

    def test_position_fields_are_set_correctly(self):
        pos = _make_position("7203", 100, 1000.0, 980.0)
        pred = _make_pred_df("7203", avg_pred_price=970.0, diff_ratio=-0.02)
        alert = self._run_with_one_position(pos, pred)

        self.assertEqual(alert.symbol, "7203")
        self.assertEqual(alert.qty, 100)
        self.assertAlmostEqual(alert.avg_price, 1000.0)
        self.assertAlmostEqual(alert.current_price, 980.0)
        self.assertAlmostEqual(alert.avg_pred_price, 970.0)
        self.assertAlmostEqual(alert.diff_ratio, -0.02)
        self.assertAlmostEqual(alert.unrealized_pnl, -2000.0)
        self.assertAlmostEqual(alert.hold_pnl_ratio, -0.02)


class TestEvaluatePositionsEarlyExit(unittest.TestCase):
    """ポジションなし・live モード・予測なし時の早期リターンをテスト"""

    def test_returns_empty_list_in_live_mode(self):
        with patch.dict("os.environ", {"AUTO_TRADE_MODE": "live"}, clear=False):
            result = evaluate_positions(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )
        self.assertEqual(result, [])

    def test_returns_empty_list_when_no_positions(self):
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
        ):
            broker = MagicMock()
            broker.get_positions.return_value = []
            mock_broker_cls.return_value = broker
            result = evaluate_positions(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )
        self.assertEqual(result, [])

    def test_returns_empty_list_when_no_prediction_timestamp(self):
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
            patch.object(
                _PREDICTIONS,
                "latest_timestamp",
                return_value=None,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = [_make_position("7203", 100, 1000.0, 1010.0)]
            mock_broker_cls.return_value = broker
            result = evaluate_positions(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )
        self.assertEqual(result, [])

    def test_multiple_positions_classified_independently(self):
        positions = [
            _make_position("7203", 100, 1000.0, 980.0),
            _make_position("6758", 50, 500.0, 510.0),
            _make_position("9984", 200, 2000.0, 2010.0),
        ]
        pred_df = pd.DataFrame(
            [
                {"symbol": "7203", "avg_pred_price": 960.0, "diff_ratio": -0.02},
                {"symbol": "6758", "avg_pred_price": 505.0, "diff_ratio": -0.005},
                {"symbol": "9984", "avg_pred_price": 2050.0, "diff_ratio": 0.02},
            ]
        )
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
            patch.object(
                _PREDICTIONS,
                "latest_timestamp",
                return_value="20260101_150000",
            ),
            patch.object(
                _PREDICTIONS,
                "results_at",
                return_value=pred_df,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = positions
            mock_broker_cls.return_value = broker
            alerts = evaluate_positions(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )

        self.assertEqual(len(alerts), 3)
        alert_map = {a.symbol: a.alert_type for a in alerts}
        self.assertEqual(alert_map["7203"], AlertType.STOP_LOSS)
        self.assertEqual(alert_map["9984"], AlertType.HOLD)


class TestGetPreCloseAlerts(unittest.TestCase):
    """get_pre_close_alerts のユニットテスト"""

    def test_returns_no_position_lines_when_empty(self):
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
        ):
            broker = MagicMock()
            broker.get_positions.return_value = []
            mock_broker_cls.return_value = broker
            lines = get_pre_close_alerts(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )

        self.assertIsInstance(lines, list)
        self.assertTrue(any("保有ポジションなし" in line for line in lines))

    def test_returns_formatted_lines_with_positions(self):
        pos = _make_position("7203", 100, 1000.0, 980.0)
        pred_df = _make_pred_df("7203", avg_pred_price=960.0, diff_ratio=-0.02)
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
            patch.object(
                _PREDICTIONS,
                "latest_timestamp",
                return_value="20260101_150000",
            ),
            patch.object(
                _PREDICTIONS,
                "results_at",
                return_value=pred_df,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = [pos]
            mock_broker_cls.return_value = broker
            lines = get_pre_close_alerts(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=_PREDICTIONS
            )

        self.assertIsInstance(lines, list)
        self.assertTrue(len(lines) > 0)
        full_text = "\n".join(lines)
        self.assertIn("7203", full_text)
        self.assertIn("損切り推奨", full_text)


if __name__ == "__main__":
    unittest.main()


class TestPredictionRepositoryWiring(unittest.TestCase):
    """注入されたリポジトリから「最新時点の予測」を読むこと。"""

    def test_reads_results_at_latest_timestamp(self):
        repo = InMemoryPredictionRepository()
        repo.add("20260101_090000", [{"market": "jp", "symbol": "7203", "diff_ratio": 0.05}])
        repo.add(
            "20260101_150000",
            [{"market": "jp", "symbol": "7203", "avg_pred_price": 960.0, "diff_ratio": -0.02}],
        )
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
        ):
            broker = MagicMock()
            broker.get_positions.return_value = [_make_position("7203", 100, 1000.0, 980.0)]
            mock_broker_cls.return_value = broker
            alerts = evaluate_positions(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=repo
            )

        self.assertEqual(len(alerts), 1)
        # 古い時点（+5%）ではなく最新時点（-2%）の予測で判定される
        self.assertAlmostEqual(alerts[0].diff_ratio, -0.02)
        self.assertEqual(alerts[0].alert_type, AlertType.STOP_LOSS)

    def _evaluate_one_position(self, repo: InMemoryPredictionRepository) -> list[PositionAlert]:
        with patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls:
            broker = MagicMock()
            broker.get_positions.return_value = [_make_position("7203", 100, 1000.0, 980.0)]
            mock_broker_cls.return_value = broker
            return evaluate_positions(trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=repo)

    def test_uses_production_run_not_newer_challenger(self):
        repo = InMemoryPredictionRepository()
        repo.add(
            "20260101_073000",
            [{"market": "jp", "symbol": "7203", "avg_pred_price": 960.0, "diff_ratio": -0.02}],
        )
        # production の後に保存される challenger の回は時刻が新しいが、アラートには使わない
        repo.add(
            "20260101_073500",
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "avg_pred_price": 1030.0,
                    "diff_ratio": 0.05,
                    "model_version": "challenger",
                }
            ],
        )
        alerts = self._evaluate_one_position(repo)

        self.assertEqual(len(alerts), 1)
        self.assertAlmostEqual(alerts[0].diff_ratio, -0.02)
        self.assertEqual(alerts[0].alert_type, AlertType.STOP_LOSS)

    def test_skips_when_production_run_older_than_previous_business_day(self):
        repo = InMemoryPredictionRepository()
        repo.add(
            "20260109_073000",
            [{"market": "jp", "symbol": "7203", "avg_pred_price": 960.0, "diff_ratio": -0.02}],
        )
        # 2026-01-14（水）の前の営業日は 01-13（火）。01-09（金）の回は古い
        with (
            patch(
                "src.trading.pre_close_alert_service._today_jst",
                return_value=date(2026, 1, 14),
            ),
            self.assertLogs("src.trading.pre_close_alert_service", level="WARNING") as logs,
        ):
            alerts = self._evaluate_one_position(repo)

        self.assertEqual(alerts, [])
        self.assertTrue(any("古い" in line for line in logs.output))

    def test_uses_run_of_previous_business_day(self):
        repo = InMemoryPredictionRepository()
        repo.add(
            "20260109_073000",
            [{"market": "jp", "symbol": "7203", "avg_pred_price": 960.0, "diff_ratio": -0.02}],
        )
        # 2026-01-13（火）の前の営業日は、成人の日（01-12）を飛ばして 01-09（金）
        with patch(
            "src.trading.pre_close_alert_service._today_jst",
            return_value=date(2026, 1, 13),
        ):
            alerts = self._evaluate_one_position(repo)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].alert_type, AlertType.STOP_LOSS)

    def _pre_close_lines(self, repo: InMemoryPredictionRepository, positions: list[dict]):
        with patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls:
            broker = MagicMock()
            broker.get_positions.return_value = positions
            mock_broker_cls.return_value = broker
            return get_pre_close_alerts(
                trade_diff_sink=InMemoryTradeDiffSink(), prediction_repo=repo
            )

    def test_lines_differ_from_no_position_when_no_production_run(self):
        repo = InMemoryPredictionRepository()
        lines = self._pre_close_lines(repo, [_make_position("7203", 100, 1000.0, 980.0)])

        full_text = "\n".join(lines)
        self.assertNotIn("保有ポジションなし", full_text)
        self.assertIn("予想の回が無い", full_text)

    def test_lines_differ_from_no_position_when_production_run_is_stale(self):
        repo = InMemoryPredictionRepository()
        repo.add(
            "20260109_073000",
            [{"market": "jp", "symbol": "7203", "avg_pred_price": 960.0, "diff_ratio": -0.02}],
        )
        with patch(
            "src.trading.pre_close_alert_service._today_jst",
            return_value=date(2026, 1, 14),
        ):
            lines = self._pre_close_lines(repo, [_make_position("7203", 100, 1000.0, 980.0)])

        full_text = "\n".join(lines)
        self.assertNotIn("保有ポジションなし", full_text)
        self.assertIn("予想の回が古い", full_text)
        self.assertIn("20260109_073000", full_text)

    def test_position_missing_from_run_is_listed_as_no_prediction(self):
        repo = InMemoryPredictionRepository()
        repo.add(
            "20260101_073000",
            [{"market": "jp", "symbol": "7203", "avg_pred_price": 1030.0, "diff_ratio": 0.02}],
        )
        positions = [
            _make_position("7203", 100, 1000.0, 1010.0),
            _make_position("6758", 50, 500.0, 510.0),
        ]
        with self.assertLogs("src.trading.pre_close_alert_service", level="WARNING") as logs:
            lines = self._pre_close_lines(repo, positions)

        full_text = "\n".join(lines)
        self.assertIn("保有継続: 1", full_text)
        self.assertIn("予想なし: 1", full_text)
        no_pred_index = next(i for i, line in enumerate(lines) if "予想なし（" in line)
        self.assertIn("6758", "\n".join(lines[no_pred_index:]))
        hold_index = next(i for i, line in enumerate(lines) if "**保有継続**" in line)
        self.assertNotIn("6758", "\n".join(lines[hold_index:no_pred_index]))
        self.assertTrue(any("予想なし" in line and "1件" in line for line in logs.output))

    def test_prediction_repo_is_required(self):
        with self.assertRaisesRegex(TypeError, "prediction_repo"):
            evaluate_positions(trade_diff_sink=InMemoryTradeDiffSink())  # type: ignore[call-arg]
