"""
ユニットテスト: OrderExecutionPipeline

DuckDB・Broker・yfinance はすべて MagicMock で差し替える。
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import (
    BUY_THRESHOLD,
    MAX_ORDERS_PER_RUN,
    MIN_CHANGE_RATIO,
    _apply_buy_sector_limit,
    _apply_split_qty,
    _attach_dynamic_thresholds,
    _calc_split_ratio,
    _choose_order_params,
    _compute_market_threshold_scale,
    run_daily_orders,
)
from src.trading.types import TradingGateStatus


def _make_broker(balance=1_000_000.0, positions=None):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = positions or []
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


def _make_predictions(n_buy=3, n_sell=0, confidence_ratio=1.0):
    """テスト用予測 DataFrame を生成する"""
    rows = []
    for i in range(n_buy):
        rows.append(
            {
                "market": "jp",
                "symbol": f"720{i}",
                "current_price": 1000.0 + i * 10,
                "diff_ratio": BUY_THRESHOLD + 0.01 * (i + 1),
                "confidence_ratio": confidence_ratio,
            }
        )
    for j in range(n_sell):
        rows.append(
            {
                "market": "jp",
                "symbol": f"900{j}",
                "current_price": 800.0,
                "diff_ratio": -0.02,
                "confidence_ratio": confidence_ratio,
            }
        )
    return pd.DataFrame(rows)


class TestRunDailyOrders(unittest.TestCase):
    def _make_gate_status(
        self,
        is_allowed=True,
        stop_active=False,
        reason_code=None,
        reason=None,
        daily_loss=0.0,
        daily_loss_limit=None,
    ):
        return TradingGateStatus(
            is_allowed=is_allowed,
            stop_active=stop_active,
            reason_code=reason_code,
            reason=reason,
            daily_loss=daily_loss,
            daily_loss_limit=daily_loss_limit,
        )

    def _patch_pipeline(self, predictions, risk_allowed=True, calc_qty=100, gate_status=None):
        """共通パッチャー群をまとめて返す"""
        if gate_status is None:
            gate_status = self._make_gate_status(is_allowed=risk_allowed)
        patches = [
            patch(
                "src.trading.execution.runner._load_latest_predictions",
                return_value=predictions,
            ),
            patch(
                "src.trading.execution.runner._record_order",
            ),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate_status,
            ),
            patch(
                "src.trading.execution.RiskManager.calc_position_size",
                return_value=calc_qty,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
                return_value=0.0,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_consecutive_losses",
                return_value=0,
            ),
            patch(
                "src.trading.execution.runner.save_order_run_summary",
            ),
        ]
        return patches

    def _start_patches(self, patches):
        mocks = [p.start() for p in patches]
        return mocks, patches

    def _stop_patches(self, patches):
        for p in patches:
            p.stop()

    def test_buy_orders_placed(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=3)
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertGreater(stats["buy_orders"], 0)
            self.assertEqual(stats["errors"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_risk_blocked_no_orders(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=3)
        patches = self._patch_pipeline(predictions, risk_allowed=False)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    def test_risk_blocked_returns_stop_details(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=3)
        gate_status = self._make_gate_status(
            is_allowed=False,
            stop_active=True,
            reason_code="daily_loss_limit",
            reason="日次損失上限に到達: 損失=25000円 / 上限=20000円",
            daily_loss=25_000.0,
            daily_loss_limit=20_000.0,
        )
        patches = self._patch_pipeline(predictions, gate_status=gate_status)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertTrue(stats["trading_stopped"])
            self.assertEqual(stats["reason_code"], "daily_loss_limit")
            self.assertEqual(stats["daily_loss"], 25_000.0)
            self.assertEqual(stats["daily_loss_limit"], 20_000.0)
            self.assertIn("日次損失上限", stats["stop_reason"])
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    def test_empty_predictions_no_orders(self):
        broker = _make_broker()
        patches = self._patch_pipeline(pd.DataFrame())
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_max_orders_per_run_respected(self):
        broker = _make_broker()
        # MAX_ORDERS_PER_RUN より多い銘柄を提供
        predictions = _make_predictions(n_buy=MAX_ORDERS_PER_RUN + 3)
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertLessEqual(stats["buy_orders"], MAX_ORDERS_PER_RUN)
        finally:
            self._stop_patches(patch_list)

    def test_max_positions_cap_respected(self):
        # ゲートが許可でも、保有が MAX_POSITIONS に達していれば新規買いは出さない
        from config.settings import MAX_POSITIONS

        held = [{"symbol": f"S{i:03d}", "qty": 100} for i in range(MAX_POSITIONS)]
        broker = _make_broker(positions=held)
        predictions = _make_predictions(n_buy=3)  # 新規銘柄 7200-7202
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    def test_held_symbol_not_rebought(self):
        # 既に保有している銘柄には買い注文を出さない
        positions = [{"symbol": "7200", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}]
        broker = _make_broker(positions=positions)
        predictions = _make_predictions(n_buy=1)  # symbol="7200"
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_zero_qty_skipped(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=2)
        patches = self._patch_pipeline(predictions, calc_qty=0)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            self.assertGreater(stats["skipped"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_confidence_ratio_passed_to_position_size(self):
        broker = _make_broker()
        # R-308: 0.65 は半量発注レンジ（calc_position_size は呼ばれる）
        predictions = _make_predictions(n_buy=1, confidence_ratio=0.65)
        patches = self._patch_pipeline(predictions)
        mocks, patch_list = self._start_patches(patches)
        try:
            calc_size_mock = mocks[4]
            run_daily_orders(broker, market="jp", mode="paper")

            self.assertTrue(calc_size_mock.called)
            self.assertAlmostEqual(calc_size_mock.call_args.kwargs["confidence_ratio"], 0.65)
        finally:
            self._stop_patches(patch_list)

    @patch("src.trading.execution.params.get_optimal_params", return_value={})
    def test_dynamic_buy_threshold_blocks_borderline_signal(self, mock_params):
        broker = _make_broker()
        predictions = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7200",
                    "current_price": 1000.0,
                    "diff_ratio": 0.006,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7201",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7202",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7203",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7204",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
            ]
        )
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    @patch("src.trading.execution.params.get_optimal_params", return_value={"threshold": 0.01})
    def test_dynamic_sell_threshold_blocks_borderline_exit(self, mock_params):
        broker = _make_broker(positions=[{"symbol": "9000", "qty": 100, "avg_price": 1000.0}])
        predictions = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "9000",
                    "current_price": 1000.0,
                    "diff_ratio": -0.006,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "9001",
                    "current_price": 1000.0,
                    "diff_ratio": -0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "9002",
                    "current_price": 1000.0,
                    "diff_ratio": -0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "9003",
                    "current_price": 1000.0,
                    "diff_ratio": -0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "9004",
                    "current_price": 1000.0,
                    "diff_ratio": -0.001,
                    "confidence_ratio": 1.0,
                },
            ]
        )
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["sell_orders"], 0)
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    def test_skipped_min_change_below_threshold(self):
        """diff_ratio が MIN_CHANGE_RATIO 未満の買いシグナルは skipped_min_change にカウントされる"""
        small_dr = MIN_CHANGE_RATIO * 0.5  # 閾値の半分 → スキップ対象
        predictions_base = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7200",
                    "current_price": 1000.0,
                    "diff_ratio": small_dr,
                    "confidence_ratio": 1.0,
                }
            ]
        )

        def _mock_attach(df):
            result = df.copy()
            result["base_threshold"] = 0.001
            result["threshold_scale"] = 1.0
            result["effective_buy_threshold"] = 0.001
            result["effective_sell_threshold"] = -0.001
            return result

        def _mock_score(df):
            result = df.copy()
            result["multi_horizon_score"] = result["diff_ratio"]
            return result

        broker = _make_broker()
        patches = self._patch_pipeline(predictions_base)
        patches += [
            patch(
                "src.trading.execution.runner._attach_dynamic_thresholds",
                side_effect=_mock_attach,
            ),
            patch(
                "src.trading.execution.runner.apply_multi_horizon_score_column",
                side_effect=_mock_score,
            ),
        ]
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["skipped_min_change"], 1)
            self.assertGreater(stats["skipped"], 0)
            self.assertEqual(stats["buy_orders"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_skipped_min_change_above_threshold(self):
        """diff_ratio が MIN_CHANGE_RATIO 以上の買いシグナルはスキップされない"""
        large_dr = MIN_CHANGE_RATIO * 2.0  # 閾値の2倍 → スキップしない
        predictions_base = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7200",
                    "current_price": 1000.0,
                    "diff_ratio": large_dr,
                    "confidence_ratio": 1.0,
                }
            ]
        )

        def _mock_attach(df):
            result = df.copy()
            result["base_threshold"] = 0.001
            result["threshold_scale"] = 1.0
            result["effective_buy_threshold"] = 0.001
            result["effective_sell_threshold"] = -0.001
            return result

        def _mock_score(df):
            result = df.copy()
            result["multi_horizon_score"] = result["diff_ratio"]
            return result

        broker = _make_broker()
        patches = self._patch_pipeline(predictions_base)
        patches += [
            patch(
                "src.trading.execution.runner._attach_dynamic_thresholds",
                side_effect=_mock_attach,
            ),
            patch(
                "src.trading.execution.runner.apply_multi_horizon_score_column",
                side_effect=_mock_score,
            ),
        ]
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["skipped_min_change"], 0)
            self.assertEqual(stats["buy_orders"], 1)
        finally:
            self._stop_patches(patch_list)

    def test_save_order_run_summary_called(self):
        """run_daily_orders 実行後に save_order_run_summary が呼ばれる"""
        broker = _make_broker()
        predictions = _make_predictions(n_buy=1)
        patches = self._patch_pipeline(predictions)
        mocks, patch_list = self._start_patches(patches)
        try:
            run_daily_orders(broker, market="jp", mode="paper")
            mock_save = mocks[-1]  # save_order_run_summary は _patch_pipeline の最後
            mock_save.assert_called_once()
        finally:
            self._stop_patches(patch_list)


class TestDynamicThresholdHelpers(unittest.TestCase):
    def test_market_threshold_scale_expands_on_high_dispersion(self):
        predictions = pd.DataFrame({"diff_ratio": [0.006, 0.001, 0.001, 0.001, 0.001]})

        scale = _compute_market_threshold_scale(predictions)

        self.assertGreater(scale, 1.0)

    @patch("src.trading.execution.params.get_optimal_params")
    def test_attach_dynamic_thresholds_uses_optimal_threshold(self, mock_params):
        mock_params.side_effect = [{"threshold": 0.012}, {}]
        predictions = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7200",
                    "current_price": 1000.0,
                    "diff_ratio": 0.020,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7201",
                    "current_price": 1000.0,
                    "diff_ratio": 0.010,
                    "confidence_ratio": 1.0,
                },
            ]
        )

        enriched = _attach_dynamic_thresholds(predictions)

        self.assertAlmostEqual(enriched.loc[0, "base_threshold"], 0.012)
        self.assertAlmostEqual(enriched.loc[1, "base_threshold"], BUY_THRESHOLD)
        self.assertIn("effective_buy_threshold", enriched.columns)
        self.assertIn("effective_sell_threshold", enriched.columns)


class TestExecutionOrderTypeHelpers(unittest.TestCase):
    def _make_adapter(self, volume: int, high: float, low: float, close: float = 100.0):
        from src.infrastructure.in_memory import InMemoryMarketDataAdapter

        adapter = InMemoryMarketDataAdapter()
        adapter.set_ohlcv_data(
            "7203.T",
            pd.DataFrame(
                {
                    "High": [high] * 5,
                    "Low": [low] * 5,
                    "Close": [close] * 5,
                    "Volume": [volume] * 5,
                }
            ),
        )
        return adapter

    def test_choose_order_params_switches_to_limit_on_low_volume(self):
        adapter = self._make_adapter(volume=100_000, high=101.0, low=99.0)

        order_type, price, reason, session = _choose_order_params(
            "jp", "7203", OrderSide.BUY, 1000.0, market_data=adapter
        )

        self.assertEqual(order_type, OrderType.LIMIT)
        self.assertGreater(price, 1000.0)
        self.assertIn("low_volume", reason)
        self.assertEqual(session, "close")

    def test_choose_order_params_keeps_market_when_liquid(self):
        adapter = self._make_adapter(volume=2_000_000, high=100.3, low=99.7)

        order_type, price, reason, session = _choose_order_params(
            "jp", "7203", OrderSide.SELL, 1000.0, market_data=adapter
        )

        self.assertEqual(order_type, OrderType.MARKET)
        self.assertEqual(price, 0.0)
        self.assertEqual(reason, "market")
        self.assertEqual(session, "open")


class TestExecutionOrderTypeFlow(unittest.TestCase):
    @patch("src.trading.execution.runner._record_order")
    @patch(
        "src.trading.execution.runner._choose_order_params",
        return_value=(OrderType.LIMIT, 1001.0, "low_volume=100000", "close"),
    )
    @patch(
        "src.trading.execution.RiskManager.evaluate_trading_gate",
        return_value=TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        ),
    )
    @patch(
        "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
        return_value=0.0,
    )
    @patch(
        "src.trading.risk_manager.RiskManager._get_consecutive_losses",
        return_value=0,
    )
    @patch("src.trading.execution.RiskManager.calc_position_size", return_value=100)
    @patch(
        "src.trading.execution.runner._load_latest_predictions",
        return_value=pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "predicted_at": "20260405_090000",
                    "current_price": 1000.0,
                    "diff_ratio": 0.03,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7204",
                    "predicted_at": "20260405_090000",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7205",
                    "predicted_at": "20260405_090000",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7206",
                    "predicted_at": "20260405_090000",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
                {
                    "market": "jp",
                    "symbol": "7207",
                    "predicted_at": "20260405_090000",
                    "current_price": 1000.0,
                    "diff_ratio": 0.001,
                    "confidence_ratio": 1.0,
                },
            ]
        ),
    )
    def test_run_daily_orders_uses_limit_when_switch_rule_matches(
        self,
        _mock_predictions,
        _mock_qty,
        _mock_consecutive,
        _mock_loss,
        _mock_gate,
        mock_choose,
        mock_record,
    ):
        broker = _make_broker()

        stats = run_daily_orders(broker, market="jp", mode="paper")

        self.assertEqual(stats["buy_orders"], 1)
        broker.send_order.assert_called_once_with(
            "7203",
            OrderSide.BUY,
            100,
            price=1001.0,
            order_type=OrderType.LIMIT,
        )
        self.assertTrue(mock_choose.called)
        self.assertEqual(mock_record.call_args.kwargs["order_type"], OrderType.LIMIT)


class TestSectorLimitHelpers(unittest.TestCase):
    @patch("src.trading.execution.selection.get_symbol_sector")
    def test_apply_buy_sector_limit_reduces_same_sector_concentration(self, mock_get_sector):
        mock_get_sector.side_effect = ["Auto", "Auto", "Tech", "Bank"]
        predictions = pd.DataFrame(
            [
                {"market": "jp", "symbol": "7200", "diff_ratio": 0.03, "current_price": 1000.0},
                {"market": "jp", "symbol": "7201", "diff_ratio": 0.02, "current_price": 1000.0},
                {"market": "jp", "symbol": "6758", "diff_ratio": 0.018, "current_price": 1000.0},
                {"market": "jp", "symbol": "8306", "diff_ratio": 0.016, "current_price": 1000.0},
            ]
        )

        limited = _apply_buy_sector_limit(predictions, max_sector_positions=1)

        self.assertEqual(limited["symbol"].tolist(), ["7200", "6758", "8306"])
        self.assertEqual(limited["sector"].tolist(), ["Auto", "Tech", "Bank"])


class TestShortSide(unittest.TestCase):
    """R-215: ENABLE_SHORT_SIDE=True 時のショートサイド処理テスト"""

    def _make_gate_status(self, is_allowed=True):
        return TradingGateStatus(
            is_allowed=is_allowed,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )

    def _make_short_predictions(self):
        """売りシグナル（強い下落予測）の予測DataFrameを返す"""
        return pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "8001",
                    "predicted_at": "20260405_090000",
                    "current_price": 2000.0,
                    "diff_ratio": -0.03,
                    "confidence_ratio": 1.0,
                    "diff_ratio_3d": -0.02,
                    "diff_ratio_5d": -0.02,
                    "diff_ratio_10d": -0.01,
                    "confluence_score": 0.5,
                },
            ]
        )

    @patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True)
    def test_short_order_placed_when_enabled(self):
        """ENABLE_SHORT_SIDE=True かつ PaperBroker で売りシグナル → SHORT 発注"""
        broker = _make_broker()
        broker.broker_name = "paper"
        broker.get_short_positions = MagicMock(return_value=[])
        predictions = self._make_short_predictions()

        patches = [
            patch(
                "src.trading.execution.runner._load_latest_predictions",
                return_value=predictions,
            ),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=self._make_gate_status(),
            ),
            patch(
                "src.trading.execution.RiskManager.calc_position_size",
                return_value=50,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
                return_value=0.0,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_consecutive_losses",
                return_value=0,
            ),
            patch("src.trading.execution.runner.save_order_run_summary"),
        ]
        for p in patches:
            p.start()
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertGreater(stats["short_orders"], 0)
            broker.send_order.assert_called_once_with(
                "8001",
                OrderSide.SHORT,
                50,
                price=0.0,
                order_type=OrderType.MARKET,
            )
        finally:
            for p in patches:
                p.stop()

    @patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", False)
    def test_no_short_order_when_disabled(self):
        """ENABLE_SHORT_SIDE=False では売りシグナルがあっても SHORT 発注しない"""
        broker = _make_broker()
        broker.broker_name = "paper"
        predictions = self._make_short_predictions()

        patches = [
            patch(
                "src.trading.execution.runner._load_latest_predictions",
                return_value=predictions,
            ),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=self._make_gate_status(),
            ),
            patch(
                "src.trading.execution.RiskManager.calc_position_size",
                return_value=50,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
                return_value=0.0,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_consecutive_losses",
                return_value=0,
            ),
            patch("src.trading.execution.runner.save_order_run_summary"),
        ]
        for p in patches:
            p.start()
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["short_orders"], 0)
            broker.send_order.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    @patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True)
    def test_short_orders_counted_in_stats(self):
        """short_orders カウントが stats に正しく反映される"""
        broker = _make_broker()
        broker.broker_name = "paper"
        broker.get_short_positions = MagicMock(return_value=[])
        predictions = self._make_short_predictions()

        patches = [
            patch(
                "src.trading.execution.runner._load_latest_predictions",
                return_value=predictions,
            ),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=self._make_gate_status(),
            ),
            patch(
                "src.trading.execution.RiskManager.calc_position_size",
                return_value=30,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
                return_value=0.0,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_consecutive_losses",
                return_value=0,
            ),
            patch("src.trading.execution.runner.save_order_run_summary"),
        ]
        mocks = [p.start() for p in patches]
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["short_orders"], 1)
            # save_order_run_summary には short_orders=1 が渡される
            mock_save = mocks[-1]
            call_kwargs = mock_save.call_args.kwargs
            self.assertEqual(call_kwargs["short_orders"], 1)
        finally:
            for p in patches:
                p.stop()


class TestSplitRatio(unittest.TestCase):
    """R-308: 分割エントリー/エグジット（確信度連動）のテスト"""

    def test_high_confidence_full_order(self):
        self.assertEqual(_calc_split_ratio(0.80), 1.0)
        self.assertEqual(_calc_split_ratio(0.90), 1.0)
        self.assertEqual(_calc_split_ratio(1.00), 1.0)

    def test_mid_confidence_half_order(self):
        self.assertEqual(_calc_split_ratio(0.50), 0.5)
        self.assertEqual(_calc_split_ratio(0.65), 0.5)
        self.assertAlmostEqual(_calc_split_ratio(0.799), 0.5)

    def test_low_confidence_skip(self):
        self.assertEqual(_calc_split_ratio(0.49), 0.0)
        self.assertEqual(_calc_split_ratio(0.00), 0.0)

    def test_apply_split_qty_half(self):
        self.assertEqual(_apply_split_qty(200, 0.5), 100)
        self.assertEqual(_apply_split_qty(300, 0.5), 100)
        self.assertEqual(_apply_split_qty(400, 0.5), 200)

    def test_apply_split_qty_minimum_100(self):
        self.assertEqual(_apply_split_qty(100, 0.5), 100)
        self.assertEqual(_apply_split_qty(50, 0.5), 100)

    def test_low_confidence_skipped_in_run(self):
        """confidence_ratio < 0.50 の銘柄は発注されず skipped にカウントされる"""
        broker = _make_broker()
        predictions = _make_predictions(n_buy=2, confidence_ratio=0.30)
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            self.assertGreater(stats["skipped"], 0)
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    def test_mid_confidence_half_qty_ordered(self):
        """confidence_ratio が 0.50〜0.80 のとき qty が半分（100株単位）で発注される"""
        broker = _make_broker()
        predictions = _make_predictions(n_buy=1, confidence_ratio=0.65)
        patches = self._patch_pipeline(predictions, calc_qty=200)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 1)
            call_args = broker.send_order.call_args
            sent_qty = call_args.args[2] if call_args.args else call_args.kwargs.get("qty")
            self.assertEqual(sent_qty, 100)
        finally:
            self._stop_patches(patch_list)

    def test_high_confidence_full_qty_ordered(self):
        """confidence_ratio >= 0.80 のとき qty が変更されずに発注される"""
        broker = _make_broker()
        predictions = _make_predictions(n_buy=1, confidence_ratio=0.85)
        patches = self._patch_pipeline(predictions, calc_qty=200)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 1)
            call_args = broker.send_order.call_args
            sent_qty = call_args.args[2] if call_args.args else call_args.kwargs.get("qty")
            self.assertEqual(sent_qty, 200)
        finally:
            self._stop_patches(patch_list)

    def _make_gate_status(self, is_allowed=True):
        return TradingGateStatus(
            is_allowed=is_allowed,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )

    def _patch_pipeline(self, predictions, risk_allowed=True, calc_qty=100, gate_status=None):
        if gate_status is None:
            gate_status = self._make_gate_status(is_allowed=risk_allowed)
        return [
            patch(
                "src.trading.execution.runner._load_latest_predictions",
                return_value=predictions,
            ),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate_status,
            ),
            patch(
                "src.trading.execution.RiskManager.calc_position_size",
                return_value=calc_qty,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
                return_value=0.0,
            ),
            patch(
                "src.trading.risk_manager.RiskManager._get_consecutive_losses",
                return_value=0,
            ),
            patch("src.trading.execution.runner.save_order_run_summary"),
        ]

    def _start_patches(self, patches):
        mocks = [p.start() for p in patches]
        return mocks, patches

    def _stop_patches(self, patches):
        for p in patches:
            p.stop()


if __name__ == "__main__":
    unittest.main()
