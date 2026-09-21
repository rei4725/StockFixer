"""simulate_trading の collect_equity オプションのテスト（ポートフォリオDD の入力）。

日次 mark-to-market equity は内部で構築済みだが外に出ていなかった。
既定では従来どおり返さず、明示的に要求されたときだけ metrics に載せる。
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.backtest.backtester import Backtester


def _make_backtester() -> Backtester:
    return Backtester(
        model_manager=None,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market="",
        symbol="",
        initial_cash=1_000_000,
        fee_rate=0.0,
        slippage=0.0,
        stop_loss_pct=None,
        take_profit_pct=None,
    )


class TestCollectEquity(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        self.df = pd.DataFrame(
            {
                "Close": [100, 105, 110, 108, 115, 120, 118, 125, 130, 128],
                "Open": [100, 105, 110, 108, 115, 120, 118, 125, 130, 128],
            },
            index=index,
        )
        # 買って持ち切り、最後に売る
        self.signal = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, -1], index=index)

    def test_default_does_not_include_equity_curve(self):
        _, metrics = _make_backtester().simulate_trading(self.df, self.signal)

        self.assertNotIn("equity_curve", metrics)

    def test_collect_equity_returns_daily_series(self):
        _, metrics = _make_backtester().simulate_trading(self.df, self.signal, collect_equity=True)

        curve = metrics["equity_curve"]
        self.assertIsInstance(curve, pd.Series)
        self.assertFalse(curve.empty)
        # 保有中の含み損益が反映された日次 equity であること
        self.assertEqual(len(curve), len(self.df))
        self.assertGreater(curve.iloc[-1], 0.0)

    def test_collect_equity_does_not_change_other_metrics(self):
        _, plain = _make_backtester().simulate_trading(self.df, self.signal)
        _, collected = _make_backtester().simulate_trading(
            self.df, self.signal, collect_equity=True
        )

        collected.pop("equity_curve")
        self.assertEqual(plain.keys(), collected.keys())
        self.assertAlmostEqual(plain["total_return"], collected["total_return"])
        self.assertAlmostEqual(plain["max_drawdown"], collected["max_drawdown"])


if __name__ == "__main__":
    unittest.main()
