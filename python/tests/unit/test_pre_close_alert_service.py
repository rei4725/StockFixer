"""ユニットテスト: 引け前ポジション再評価アラートサービス（R-409）"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.pre_close_alert_service import (
    STOP_LOSS_PRED_THRESHOLD,
    TAKE_PROFIT_HOLD_THRESHOLD,
    AlertType,
    PositionAlert,
    evaluate_positions,
)


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
            patch(
                "src.trading.brokers.paper.paper_broker.PaperBroker"
            ) as mock_broker_cls,
            patch(
                "src.utils.db.prediction.load_latest_prediction_timestamp",
                return_value="20260101_150000",
            ),
            patch(
                "src.utils.db.prediction.load_prediction_results",
                return_value=pred_df,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = [position]
            mock_broker_cls.return_value = broker

            alerts = evaluate_positions()

        self.assertEqual(len(alerts), 1)
        return alerts[0]

    def test_stop_loss_when_diff_ratio_below_threshold(self):
        pos = _make_position("7203", 100, 1000.0, 980.0)
        pred = _make_pred_df("7203", avg_pred_price=970.0, diff_ratio=STOP_LOSS_PRED_THRESHOLD - 0.001)
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

    def test_no_prediction_defaults_to_hold(self):
        pos = _make_position("9999", 50, 500.0, 510.0)
        pred = pd.DataFrame()  # no prediction data
        alert = self._run_with_one_position(pos, pred)
        self.assertEqual(alert.alert_type, AlertType.HOLD)
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
            result = evaluate_positions()
        self.assertEqual(result, [])

    def test_returns_empty_list_when_no_positions(self):
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
        ):
            broker = MagicMock()
            broker.get_positions.return_value = []
            mock_broker_cls.return_value = broker
            result = evaluate_positions()
        self.assertEqual(result, [])

    def test_returns_empty_list_when_no_prediction_timestamp(self):
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
            patch(
                "src.utils.db.prediction.load_latest_prediction_timestamp",
                return_value=None,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = [_make_position("7203", 100, 1000.0, 1010.0)]
            mock_broker_cls.return_value = broker
            result = evaluate_positions()
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
            patch(
                "src.utils.db.prediction.load_latest_prediction_timestamp",
                return_value="20260101_150000",
            ),
            patch(
                "src.utils.db.prediction.load_prediction_results",
                return_value=pred_df,
            ),
        ):
            broker = MagicMock()
            broker.get_positions.return_value = positions
            mock_broker_cls.return_value = broker
            alerts = evaluate_positions()

        self.assertEqual(len(alerts), 3)
        alert_map = {a.symbol: a.alert_type for a in alerts}
        self.assertEqual(alert_map["7203"], AlertType.STOP_LOSS)
        self.assertEqual(alert_map["9984"], AlertType.HOLD)


if __name__ == "__main__":
    unittest.main()
