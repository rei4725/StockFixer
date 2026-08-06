"""有効銘柄数ゲートのテスト（#625）。"""

from __future__ import annotations

import unittest
import unittest.mock

from src.backtest import factory
from src.backtest.factory import apply_gate
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


def _make_eval(**kwargs) -> FactoryEvaluation:
    defaults = dict(
        hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
        sharpe_ratio=2.0,
        num_trades=50,
        max_drawdown=-0.10,
        dsr=0.97,
        pbo=0.30,
        n_effective_symbols=50,
    )
    defaults.update(kwargs)
    return FactoryEvaluation(**defaults)


class TestEffectiveSymbolsGate(unittest.TestCase):
    def test_fails_when_effective_symbols_below_minimum(self):
        ev = _make_eval(n_effective_symbols=5)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertEqual(ev.gate_reasons, ["effective_symbols 5 < 20"])

    def test_passes_when_effective_symbols_at_minimum(self):
        ev = _make_eval(n_effective_symbols=20)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed)

    def test_threshold_is_configurable(self):
        ev = _make_eval(n_effective_symbols=5)

        with unittest.mock.patch.object(factory, "FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 3):
            apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed)

    def test_artifact_hypothesis_fails_on_both_trades_and_symbols(self):
        # #598 相当: フィルタ後は取引数 0 / 有効銘柄 0
        ev = _make_eval(sharpe_ratio=0.0, num_trades=0, n_effective_symbols=0)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("num_trades" in r for r in ev.gate_reasons))
        self.assertTrue(any("effective_symbols" in r for r in ev.gate_reasons))


if __name__ == "__main__":
    unittest.main()
