"""ユニットテスト: ルールベース日次シグナルパイプライン（rule_engine BC）"""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.infrastructure.in_memory import InMemoryTradeDiffSink


class _MockMarketDataPort:
    """OHLCVWithIndicatorsPort の最小モック実装"""

    def __init__(self, df):
        self._df = df

    def get_ohlcv_with_indicators(self, ticker, lookback_days):
        return self._df


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1000, 1100, n)
    return pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
            "Volume": np.full(n, 500_000, dtype=float),
        },
        index=idx,
    )


class TestGetTodaySignal(unittest.TestCase):
    def test_unknown_rule_returns_none(self):
        """未知のルールは HOLD にせず、シグナルを出さない（#822）。"""
        from src.rule_engine.pipeline import _get_today_signal

        self.assertIsNone(_get_today_signal("jp", "7203", "unknown_rule", MagicMock()))

    def test_empty_df_returns_none(self):
        """データ不足は HOLD にせず、シグナルを出さない（#822）。"""
        from src.rule_engine.pipeline import _get_today_signal

        mock_port = _MockMarketDataPort(pd.DataFrame())
        self.assertIsNone(_get_today_signal("jp", "7203", "rsi_contrarian", mock_port))

    def test_valid_df_returns_signal_and_price(self):
        from src.rule_engine.pipeline import _get_today_signal

        df = _make_ohlcv()
        mock_port = _MockMarketDataPort(df)
        signal, price = _get_today_signal("jp", "7203", "rsi_contrarian", mock_port)
        self.assertIn(signal, [-1, 0, 1])
        self.assertIsInstance(price, float)

    def test_stale_last_bar_returns_none_with_warning(self):
        """最後の足が平日で N 日を超えて古ければシグナルを出さず WARNING を出す。"""
        from src.rule_engine.pipeline import _get_today_signal

        # 2026-09-30（水）で止まった足を、2026-10-20（火）の回で読む
        df = _make_ohlcv()
        df.index = pd.bdate_range(end="2026-09-30", periods=len(df))
        mock_port = _MockMarketDataPort(df)
        with self.assertLogs("src.rule_engine.pipeline", level="WARNING") as logs:
            result = _get_today_signal(
                "jp", "5202", "rsi_contrarian", mock_port, signal_date=date(2026, 10, 20)
            )
        self.assertIsNone(result)
        joined = "\n".join(logs.output)
        self.assertIn("jp/5202", joined)
        self.assertIn("2026-09-30", joined)

    def test_short_gap_after_holiday_still_returns_signal(self):
        """祝日明けの1〜2平日の空きなら今までどおりシグナルを出す。"""
        from src.rule_engine.pipeline import _get_today_signal

        # 2026-10-09（金）の足まで、10-12（月・祝）を挟んで 10-13（火）の回
        df = _make_ohlcv()
        df.index = pd.bdate_range(end="2026-10-09", periods=len(df))
        mock_port = _MockMarketDataPort(df)
        result = _get_today_signal(
            "jp", "7203", "rsi_contrarian", mock_port, signal_date=date(2026, 10, 13)
        )
        self.assertIsNotNone(result)
        assert result is not None
        signal, price = result
        self.assertIn(signal, [-1, 0, 1])
        self.assertIsInstance(price, float)


class TestRunRuleSignalPipeline(unittest.TestCase):
    def test_empty_effective_rules_returns_empty(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp", market_data_port=MagicMock())
        self.assertEqual(result, [])

    def test_returns_results_for_each_symbol(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        effective_df = pd.DataFrame(
            [
                {
                    "symbol": "7203",
                    "best_rule": "rsi_contrarian",
                    "win_rate": 0.6,
                    "net_profit": 1000,
                },
                {
                    "symbol": "6758",
                    "best_rule": "ema_momentum",
                    "win_rate": 0.55,
                    "net_profit": 500,
                },
            ]
        )
        df = _make_ohlcv()
        mock_port = _MockMarketDataPort(df)
        with (
            patch(
                "src.rule_engine.pipeline.load_effective_rules",
                return_value=effective_df,
            ),
            patch("src.rule_engine.pipeline.upsert_rule_signal"),
        ):
            # 足の最後（2024-03-22）の日の回とする（今日だと足が止まった扱いになる）
            result = run_rule_signal_pipeline(
                "jp", signal_date=date(2024, 3, 22), market_data_port=mock_port
            )
        self.assertEqual(len(result), 2)
        self.assertIn("symbol", result[0])
        self.assertIn("signal_label", result[0])

    def test_stale_symbol_is_not_saved(self):
        """足が止まった銘柄は BUY・SELL・HOLD のどれとしても保存しない。"""
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        effective_df = pd.DataFrame(
            [
                {
                    "symbol": "5202",
                    "best_rule": "rsi_contrarian",
                    "win_rate": 0.6,
                    "net_profit": 1000,
                },
            ]
        )
        df = _make_ohlcv()
        df.index = pd.bdate_range(end="2026-09-30", periods=len(df))
        mock_port = _MockMarketDataPort(df)
        with (
            patch(
                "src.rule_engine.pipeline.load_effective_rules",
                return_value=effective_df,
            ),
            patch("src.rule_engine.pipeline.upsert_rule_signal") as mock_upsert,
            self.assertLogs("src.rule_engine.pipeline", level="WARNING") as logs,
        ):
            result = run_rule_signal_pipeline(
                "jp", signal_date=date(2026, 10, 20), market_data_port=mock_port
            )
        self.assertEqual(result, [])
        mock_upsert.assert_not_called()
        joined = "\n".join(logs.output)
        self.assertIn("jp/5202", joined)
        self.assertIn("2026-09-30", joined)

    def test_signal_exception_is_caught(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        effective_df = pd.DataFrame(
            [{"symbol": "7203", "best_rule": "rsi_contrarian", "win_rate": 0.6, "net_profit": 100}]
        )
        with (
            patch(
                "src.rule_engine.pipeline.load_effective_rules",
                return_value=effective_df,
            ),
            patch(
                "src.rule_engine.pipeline._get_today_signal",
                side_effect=RuntimeError("fetch error"),
            ),
        ):
            result = run_rule_signal_pipeline("jp", market_data_port=MagicMock())
        self.assertEqual(result, [])

    def test_signal_date_defaults_to_today(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp", signal_date=None, market_data_port=MagicMock())
        self.assertEqual(result, [])

    def test_unavailable_symbols_are_counted_apart_from_hold_and_not_saved(self):
        """データ不足・未知のルール・例外の銘柄は HOLD と別に数え、保存しない（#822）。"""
        from src.rule_engine.pipeline import generate_rule_signals

        effective_df = pd.DataFrame(
            [
                {"symbol": "7203", "best_rule": "rsi_contrarian", "win_rate": 0.6, "net_profit": 1},
                {"symbol": "6758", "best_rule": "unknown_rule", "win_rate": 0.6, "net_profit": 1},
                {"symbol": "9984", "best_rule": "rsi_contrarian", "win_rate": 0.6, "net_profit": 1},
                {"symbol": "8306", "best_rule": "rsi_contrarian", "win_rate": 0.6, "net_profit": 1},
            ]
        )
        good = _make_ohlcv()
        short = _make_ohlcv(5)

        class _Port:
            def get_ohlcv_with_indicators(self, ticker, lookback_days):
                if ticker.startswith("9984"):
                    return short
                if ticker.startswith("8306"):
                    raise RuntimeError("fetch error")
                return good

        with (
            patch("src.rule_engine.pipeline.load_effective_rules", return_value=effective_df),
            patch("src.rule_engine.pipeline.upsert_rule_signal") as mock_upsert,
        ):
            batch = generate_rule_signals(
                "jp", signal_date=date(2024, 3, 22), market_data_port=_Port()
            )
        self.assertEqual(batch.total, 4)
        self.assertEqual([s["symbol"] for s in batch.signals], ["7203"])
        self.assertEqual(sorted(u.symbol for u in batch.unavailable), ["6758", "8306", "9984"])
        mock_upsert.assert_called_once()
        self.assertEqual(mock_upsert.call_args.args[2], "7203")

    def test_custom_signal_date(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline(
                "jp", signal_date=date(2024, 1, 15), market_data_port=MagicMock()
            )
        self.assertEqual(result, [])


class TestExecuteRulePaperTrades(unittest.TestCase):
    @staticmethod
    def _mock_risk(is_allowed=True, qty=100):
        """RiskManager のモックを生成する（ゲート判定と発注株数を制御）。"""
        risk = MagicMock()
        risk.update_peak_balance.return_value = None
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=is_allowed, reason="test")
        risk.calc_position_size.return_value = qty
        return risk

    def test_empty_signals_returns_zero_counts(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk()),
        ):
            result = execute_rule_paper_trades([], "jp", trade_diff_sink=InMemoryTradeDiffSink())
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(result["sell_orders"], 0)

    def test_buy_signal_places_order(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk(qty=100)),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["buy_orders"], 1)
        mock_broker.send_order.assert_called_once()

    def test_buy_skipped_when_gate_blocked(self):
        """リスクゲート不合格時は新規買いを発注しない。"""
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch(
                "src.trading.rule_execution.RiskManager",
                return_value=self._mock_risk(is_allowed=False),
            ),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(result["skipped"], 1)
        mock_broker.send_order.assert_not_called()

    def test_buy_skipped_when_max_positions_reached(self):
        """保有銘柄数が MAX_POSITIONS に達していれば新規買いをスキップする。"""
        from config.settings import MAX_POSITIONS
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        # MAX_POSITIONS 件をすでに保有している状態を作る
        mock_broker.get_positions.return_value = [
            {"symbol": f"S{i:04d}", "qty": 100} for i in range(MAX_POSITIONS)
        ]

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk()),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["buy_orders"], 0)
        mock_broker.send_order.assert_not_called()

    def test_buy_skipped_when_position_size_zero(self):
        """残高不足等で株数 0 のときは発注しない。"""
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk(qty=0)),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["buy_orders"], 0)
        mock_broker.send_order.assert_not_called()

    def test_sell_signal_places_order_when_holding(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [{"symbol": "7203", "qty": 100}]

        signals = [{"symbol": "7203", "signal": -1, "price": 1100.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk()),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["sell_orders"], 1)

    def test_sell_allowed_even_when_gate_blocked(self):
        """ゲート不合格でも決済（SELL）は許可する。"""
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [{"symbol": "7203", "qty": 100}]

        signals = [{"symbol": "7203", "signal": -1, "price": 1100.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch(
                "src.trading.rule_execution.RiskManager",
                return_value=self._mock_risk(is_allowed=False),
            ),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["sell_orders"], 1)

    def test_hold_signal_is_skipped(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        signals = [{"symbol": "7203", "signal": 0, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk()),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertEqual(result["skipped"], 1)

    def test_broker_exception_is_caught(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.side_effect = RuntimeError("broker error")

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.RiskManager", return_value=self._mock_risk()),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink()
            )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
