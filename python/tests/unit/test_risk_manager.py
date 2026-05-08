"""
ユニットテスト: RiskManager

外部依存（DuckDB・Broker）はすべて MagicMock で差し替える。
"""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.trading.brokers.base import BrokerBase, OrderSide
from src.trading.risk_manager import (
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS_RATE,
    MAX_POSITION_RATE,
    MAX_POSITIONS,
    RiskManager,
)


def _make_broker(balance: float = 1_000_000.0, positions: list | None = None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = positions or []
    return broker


class TestRiskManagerIsAllowed(unittest.TestCase):
    """is_trading_allowed() のゲートチェックテスト"""

    def _make_risk(self, broker, daily_loss=0.0, consecutive=0):
        risk = RiskManager(broker)
        risk._get_daily_realized_loss = MagicMock(return_value=daily_loss)
        risk._get_consecutive_losses = MagicMock(return_value=consecutive)
        return risk

    def test_all_clear_returns_true(self):
        broker = _make_broker()
        risk = self._make_risk(broker)
        self.assertTrue(risk.is_trading_allowed())

    def test_daily_loss_exceed_blocks(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        # 損失が残高の2%ちょうど → ブロック
        loss = balance * MAX_DAILY_LOSS_RATE
        risk = self._make_risk(broker, daily_loss=loss)
        self.assertFalse(risk.is_trading_allowed())

    def test_daily_loss_status_contains_reason(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        loss = balance * MAX_DAILY_LOSS_RATE
        risk = self._make_risk(broker, daily_loss=loss)

        status = risk.evaluate_trading_gate()

        self.assertFalse(status.is_allowed)
        self.assertTrue(status.stop_active)
        self.assertEqual(status.reason_code, "daily_loss_limit")
        self.assertIn("日次損失上限", status.reason)
        self.assertEqual(status.daily_loss, loss)

    def test_daily_loss_guard_can_be_disabled_by_env(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        loss = balance * MAX_DAILY_LOSS_RATE * 3
        risk = self._make_risk(broker, daily_loss=loss)

        with patch.dict("os.environ", {"DISABLE_DAILY_LOSS_GUARD": "1"}, clear=False):
            status = risk.evaluate_trading_gate()

        self.assertTrue(status.is_allowed)
        self.assertIsNone(status.daily_loss_limit)
        risk._get_daily_realized_loss.assert_not_called()

    def test_daily_loss_rate_can_be_overridden_by_env(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        risk = self._make_risk(broker, daily_loss=15_000.0)

        with patch.dict("os.environ", {"MAX_DAILY_LOSS_RATE": "0.01"}, clear=False):
            status = risk.evaluate_trading_gate()

        self.assertFalse(status.is_allowed)
        self.assertEqual(status.daily_loss_limit, 10_000.0)

    def test_consecutive_loss_blocks(self):
        broker = _make_broker()
        risk = self._make_risk(broker, consecutive=MAX_CONSECUTIVE_LOSSES)
        self.assertFalse(risk.is_trading_allowed())

    def test_max_positions_blocks_new_buy(self):
        positions = [{"symbol": f"100{i}", "qty": 100} for i in range(MAX_POSITIONS)]
        broker = _make_broker(positions=positions)
        risk = self._make_risk(broker)
        # 保有銘柄数が上限に達しているので False
        self.assertFalse(risk.is_trading_allowed())

    def test_under_limit_returns_true(self):
        positions = [{"symbol": "7203", "qty": 100}]
        broker = _make_broker(positions=positions)
        risk = self._make_risk(broker)
        self.assertTrue(risk.is_trading_allowed())


class TestRiskManagerCalcPositionSize(unittest.TestCase):
    """calc_position_size() のポジションサイジングテスト"""

    def _make_risk(self, balance=1_000_000.0):
        broker = _make_broker(balance=balance)
        risk = RiskManager(broker)
        return risk

    def test_returns_multiple_of_100(self):
        risk = self._make_risk()
        qty = risk.calc_position_size("7203", price=1000.0)
        self.assertEqual(qty % 100, 0)

    def test_zero_price_returns_zero(self):
        risk = self._make_risk()
        qty = risk.calc_position_size("7203", price=0.0)
        self.assertEqual(qty, 0)

    def test_within_position_cap(self):
        balance = 1_000_000.0
        price = 500.0
        risk = self._make_risk(balance=balance)
        qty = risk.calc_position_size("7203", price=price)
        # 上限: balance × 10% / price
        self.assertLessEqual(qty * price, balance * MAX_POSITION_RATE + price)

    def test_high_win_rate_yields_more_than_low(self):
        risk = self._make_risk()
        qty_high = risk.calc_position_size(
            "7203", 1000.0, win_rate=0.7, avg_win=0.02, avg_loss=0.01
        )
        qty_low = risk.calc_position_size("7203", 1000.0, win_rate=0.4, avg_win=0.02, avg_loss=0.01)
        self.assertGreaterEqual(qty_high, qty_low)

    def test_lower_confidence_ratio_reduces_position_size(self):
        risk = self._make_risk()
        qty_high_conf = risk.calc_position_size("7203", 1000.0, confidence_ratio=1.0)
        qty_low_conf = risk.calc_position_size("7203", 1000.0, confidence_ratio=0.4)

        self.assertGreaterEqual(qty_high_conf, qty_low_conf)
        self.assertGreater(qty_high_conf, 0)


class TestRiskManagerPersistence(unittest.TestCase):
    """DB参照を伴う内部ヘルパーのテスト"""

    def test_daily_realized_loss_uses_paper_orders(self):
        broker = _make_broker()
        risk = RiskManager(broker)

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = (-1234.0,)

        @contextmanager
        def mock_db():
            yield mock_con

        with patch("src.trading.risk_manager._db_connection", new=mock_db):
            daily_loss = risk._get_daily_realized_loss()

        self.assertEqual(daily_loss, 1234.0)
        query, params = mock_con.execute.call_args[0]
        self.assertIn("FROM paper_orders", query)
        self.assertEqual(params, [int(OrderSide.SELL), int(OrderSide.SHORT_COVER)])

    def test_consecutive_losses_uses_paper_orders(self):
        broker = _make_broker()
        risk = RiskManager(broker)

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [(-100.0,), (-50.0,), (20.0,)]

        @contextmanager
        def mock_db():
            yield mock_con

        with patch("src.trading.risk_manager._db_connection", new=mock_db):
            consecutive = risk._get_consecutive_losses()

        self.assertEqual(consecutive, 2)
        query, params = mock_con.execute.call_args[0]
        self.assertIn("FROM paper_orders", query)
        self.assertEqual(
            params, [int(OrderSide.SELL), int(OrderSide.SHORT_COVER), MAX_CONSECUTIVE_LOSSES]
        )

    def test_missing_trade_pnl_returns_zero_for_non_paper_broker(self):
        broker = _make_broker()
        broker.broker_name = "live"
        risk = RiskManager(broker)

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = (0,)

        @contextmanager
        def mock_db():
            yield mock_con

        with patch("src.trading.risk_manager._db_connection", new=mock_db):
            daily_loss = risk._get_daily_realized_loss()

        self.assertEqual(daily_loss, 0.0)
        query, params = mock_con.execute.call_args[0]
        self.assertIn("information_schema.tables", query)
        self.assertEqual(params, ["trade_pnl"])


# ---------------------------------------------------------------------------
# R-216: RiskManager optimal_params.json 自動ロード
# ---------------------------------------------------------------------------


class TestRiskManagerOptimalParamsAutoLoad(unittest.TestCase):
    """RiskManager.__init__ が optimal_params.json を自動ロードするテスト"""

    _MOCK_TARGET = "src.trading.risk_manager.get_optimal_params"

    def test_stop_loss_and_take_profit_loaded_from_json(self):
        """market/symbol を指定すると JSON の stop_loss_pct/take_profit_pct が設定される"""
        params = {"threshold": 0.01, "stop_loss_pct": 0.03, "take_profit_pct": 0.07}
        with patch(self._MOCK_TARGET, return_value=params) as mock_fn:
            risk = RiskManager(_make_broker(), market="jp", symbol="7203")
        mock_fn.assert_called_once_with("jp", "7203")
        self.assertAlmostEqual(risk.stop_loss_pct, 0.03)
        self.assertAlmostEqual(risk.take_profit_pct, 0.07)

    def test_stop_loss_take_profit_none_when_json_not_found(self):
        """JSON が存在しない（None が返る）場合は stop_loss_pct/take_profit_pct は None"""
        with patch(self._MOCK_TARGET, return_value=None):
            risk = RiskManager(_make_broker(), market="jp", symbol="9999")
        self.assertIsNone(risk.stop_loss_pct)
        self.assertIsNone(risk.take_profit_pct)

    def test_stop_loss_take_profit_null_in_json(self):
        """JSON にキーが存在しても値が null の場合は None になる"""
        params = {"threshold": 0.005, "stop_loss_pct": None, "take_profit_pct": None}
        with patch(self._MOCK_TARGET, return_value=params):
            risk = RiskManager(_make_broker(), market="us", symbol="AAPL")
        self.assertIsNone(risk.stop_loss_pct)
        self.assertIsNone(risk.take_profit_pct)

    def test_no_json_load_when_market_symbol_not_given(self):
        """market/symbol を指定しない場合は get_optimal_params が呼ばれない"""
        with patch(self._MOCK_TARGET) as mock_fn:
            risk = RiskManager(_make_broker())
        mock_fn.assert_not_called()
        self.assertIsNone(risk.stop_loss_pct)
        self.assertIsNone(risk.take_profit_pct)

    def test_no_json_load_when_only_symbol_given(self):
        """symbol のみ指定（market なし）の場合は get_optimal_params が呼ばれない"""
        with patch(self._MOCK_TARGET) as mock_fn:
            risk = RiskManager(_make_broker(), symbol="7203")
        mock_fn.assert_not_called()
        self.assertIsNone(risk.stop_loss_pct)
        self.assertIsNone(risk.take_profit_pct)

    def test_existing_gate_check_unaffected_by_market_symbol(self):
        """market/symbol を渡してもゲートチェックの動作が変わらない"""
        broker = _make_broker()
        params = {"stop_loss_pct": 0.03, "take_profit_pct": 0.07}
        with patch(self._MOCK_TARGET, return_value=params):
            risk = RiskManager(broker, market="jp", symbol="7203")
        risk._get_daily_realized_loss = MagicMock(return_value=0.0)
        risk._get_consecutive_losses = MagicMock(return_value=0)
        self.assertTrue(risk.is_trading_allowed())


# ---------------------------------------------------------------------------
# R-217: RiskManager BT実績Kelly値フィードバック
# ---------------------------------------------------------------------------


class TestRiskManagerKellyBtFeedback(unittest.TestCase):
    """RiskManager がBT実績Kelly値をロード・参照するテスト"""

    _MOCK_TARGET = "src.trading.risk_manager.get_optimal_params"

    def test_kelly_params_loaded_from_metrics(self):
        """metrics に win_rate/avg_win/avg_loss があればインスタンス属性にセットされる"""
        params = {
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "metrics": {
                "win_rate": 0.62,
                "avg_win": 0.021,
                "avg_loss": 0.011,
            },
        }
        with patch(self._MOCK_TARGET, return_value=params):
            risk = RiskManager(_make_broker(), market="jp", symbol="7203")

        self.assertAlmostEqual(risk.kelly_win_rate, 0.62)
        self.assertAlmostEqual(risk.kelly_avg_win, 0.021)
        self.assertAlmostEqual(risk.kelly_avg_loss, 0.011)

    def test_kelly_params_fallback_when_json_not_found(self):
        """JSON が存在しない（None）場合はデフォルト値が使われる"""
        from config.trading_policy import DEFAULT_AVG_LOSS, DEFAULT_AVG_WIN, DEFAULT_WIN_RATE

        with patch(self._MOCK_TARGET, return_value=None):
            risk = RiskManager(_make_broker(), market="jp", symbol="9999")

        self.assertAlmostEqual(risk.kelly_win_rate, DEFAULT_WIN_RATE)
        self.assertAlmostEqual(risk.kelly_avg_win, DEFAULT_AVG_WIN)
        self.assertAlmostEqual(risk.kelly_avg_loss, DEFAULT_AVG_LOSS)

    def test_kelly_params_fallback_when_no_market_symbol(self):
        """market/symbol 未指定の場合もデフォルト値が使われる"""
        from config.trading_policy import DEFAULT_AVG_LOSS, DEFAULT_AVG_WIN, DEFAULT_WIN_RATE

        with patch(self._MOCK_TARGET) as mock_fn:
            risk = RiskManager(_make_broker())

        mock_fn.assert_not_called()
        self.assertAlmostEqual(risk.kelly_win_rate, DEFAULT_WIN_RATE)
        self.assertAlmostEqual(risk.kelly_avg_win, DEFAULT_AVG_WIN)
        self.assertAlmostEqual(risk.kelly_avg_loss, DEFAULT_AVG_LOSS)

    def test_explicit_args_override_instance_attributes(self):
        """calc_position_size に明示的に引数を渡すとインスタンス属性より優先される"""
        params = {
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "metrics": {"win_rate": 0.62, "avg_win": 0.021, "avg_loss": 0.011},
        }
        with patch(self._MOCK_TARGET, return_value=params):
            risk = RiskManager(_make_broker(balance=1_000_000.0), market="jp", symbol="7203")

        # 明示的に低い勝率を指定 → インスタンス属性の 0.62 より少ない株数になるはず
        qty_explicit = risk.calc_position_size(
            "7203", 1000.0, win_rate=0.4, avg_win=0.01, avg_loss=0.01
        )
        qty_instance = risk.calc_position_size("7203", 1000.0)

        self.assertGreater(qty_instance, qty_explicit)

    def test_avg_win_fallback_when_key_missing_in_old_format(self):
        """旧形式（avg_win/avg_loss キー不在）でもデフォルト値にフォールバックする"""
        from config.trading_policy import DEFAULT_AVG_LOSS, DEFAULT_AVG_WIN

        params = {
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "metrics": {
                "win_rate": 0.58,
                # avg_win / avg_loss キーなし（旧形式）
            },
        }
        with patch(self._MOCK_TARGET, return_value=params):
            risk = RiskManager(_make_broker(), market="jp", symbol="7203")

        self.assertAlmostEqual(risk.kelly_win_rate, 0.58)
        self.assertAlmostEqual(risk.kelly_avg_win, DEFAULT_AVG_WIN)
        self.assertAlmostEqual(risk.kelly_avg_loss, DEFAULT_AVG_LOSS)

    def test_none_args_use_instance_kelly_attributes(self):
        """win_rate=None のとき self.kelly_win_rate が使われる（結果確認）"""
        params = {
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "metrics": {"win_rate": 0.70, "avg_win": 0.025, "avg_loss": 0.010},
        }
        with patch(self._MOCK_TARGET, return_value=params):
            risk = RiskManager(_make_broker(balance=1_000_000.0), market="jp", symbol="7203")

        # None を明示 → インスタンス属性 (0.70) を使用
        qty_none = risk.calc_position_size("7203", 1000.0, win_rate=None)
        # 直接値を渡す（同値）
        qty_direct = risk.calc_position_size(
            "7203", 1000.0, win_rate=0.70, avg_win=0.025, avg_loss=0.010
        )

        self.assertEqual(qty_none, qty_direct)


if __name__ == "__main__":
    unittest.main()
