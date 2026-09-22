"""OpenPosition / Portfolio の現金推移と保有計算。"""

import unittest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.longterm.portfolio import OpenPosition, Portfolio
from src.domain.types import PositionEvent


def _exec(fee=0.0, slip=0.0):
    return ExecutionModel(TradingCosts(fee_rate=fee, slippage_rate=slip))


def _pos(shares=100.0, price=10.0):
    return OpenPosition(
        symbol="A",
        entry_date="2024-01-02",
        entry_price=price,
        shares=shares,
        cost_basis=shares * price,
    )


def _ev(action, price, held_fraction, reason="x", date="2024-02-02"):
    return PositionEvent(
        date=date,
        action=action,
        price=price,
        held_fraction=held_fraction,
        reason=reason,
        multiple=price / 10.0,
    )


class TestOpenPosition(unittest.TestCase):
    def test_held_shares_reflects_scale_out(self):
        pos = _pos()
        pos.current_hf = 0.8
        self.assertAlmostEqual(pos.held_shares(), 80.0)

    def test_market_value(self):
        pos = _pos()
        pos.current_hf = 0.5
        self.assertAlmostEqual(pos.market_value(12.0), 600.0)


class TestPortfolio(unittest.TestCase):
    def test_enter_deducts_cost_basis(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        self.assertAlmostEqual(pf.cash, 9_000.0)
        self.assertIn("A", pf.positions)
        self.assertEqual(pf.ledger.to_frame()["action"].iloc[0], "buy")

    def test_scale_out_adds_proceeds_and_lowers_hf(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        pf.scale_out("A", _ev("scale_out", 20.0, 0.8), _exec())
        # 20% = 20株 を 20.0 で売却 → +400
        self.assertAlmostEqual(pf.cash, 9_400.0)
        self.assertAlmostEqual(pf.positions["A"].current_hf, 0.8)
        self.assertAlmostEqual(pf.positions["A"].realized, 400.0)

    def test_close_returns_trade_with_realized_pnl(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        trade = pf.close("A", _ev("exit", 15.0, 0.0), _exec(), max_multiple=1.5)
        self.assertNotIn("A", pf.positions)
        self.assertAlmostEqual(pf.cash, 10_500.0)
        # 取得原価 1000 に対し受取 1500
        self.assertAlmostEqual(trade.realized_pnl, 500.0)
        self.assertAlmostEqual(trade.return_rate, 0.5)
        self.assertAlmostEqual(trade.multiple, 1.5)
        self.assertEqual(trade.exit_reason, "x")

    def test_realized_pnl_counts_scale_out(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        pf.scale_out("A", _ev("scale_out", 20.0, 0.8), _exec())
        trade = pf.close("A", _ev("exit", 15.0, 0.0), _exec(), max_multiple=2.0)
        # 受取 = 400（部分利確） + 80株 * 15 = 1200 → 計 1600、原価 1000
        self.assertAlmostEqual(trade.realized_pnl, 600.0)

    def test_cash_never_negative_on_costs(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        pf.close("A", _ev("exit", 1.0, 0.0), _exec(fee=0.001, slip=0.0005), max_multiple=0.1)
        self.assertGreaterEqual(pf.cash, 0.0)

    def test_equity_sums_cash_and_holdings(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        lookups = {"A": {"2024-03-01": 12.0}}
        self.assertAlmostEqual(pf.equity(lookups, "2024-03-01"), 9_000.0 + 1_200.0)

    def test_equity_missing_price_treated_as_zero(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        self.assertAlmostEqual(pf.equity({"A": {}}, "2024-03-01"), 9_000.0)


if __name__ == "__main__":
    unittest.main()
