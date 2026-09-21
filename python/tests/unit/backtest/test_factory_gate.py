"""有効銘柄数ゲートのテスト（#625）。"""

from __future__ import annotations

import unittest
import unittest.mock

from src.backtest import factory_gate
from src.backtest.factory_gate import apply_gate
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


def _make_eval(**kwargs) -> FactoryEvaluation:
    defaults = dict(
        hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
        sharpe_ratio=2.0,
        portfolio_sharpe_ratio=2.0,
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

        with unittest.mock.patch.object(factory_gate, "FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 3):
            apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed)

    def test_artifact_hypothesis_fails_on_both_trades_and_symbols(self):
        # #598 相当: フィルタ後は取引数 0 / 有効銘柄 0
        ev = _make_eval(sharpe_ratio=0.0, num_trades=0, n_effective_symbols=0)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("num_trades" in r for r in ev.gate_reasons))
        self.assertTrue(any("effective_symbols" in r for r in ev.gate_reasons))


class TestDrawdownGateUsesPortfolioDrawdown(unittest.TestCase):
    """DD ゲートは最悪銘柄DDではなくポートフォリオDDを見る。

    最悪銘柄DD（min over symbols）は有効銘柄数を増やすほど必ず悪化する最小値統計であり、
    等金額で20銘柄以上を保有する運用者が実際に経験する数字ではない。
    """

    def test_passes_when_portfolio_drawdown_is_within_threshold(self):
        # 最悪銘柄は -60% でも、ポートフォリオDDが -10% なら通す
        ev = _make_eval(max_drawdown=-0.60, portfolio_max_drawdown=-0.10)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed, ev.gate_reasons)

    def test_fails_when_portfolio_drawdown_breaches_threshold(self):
        # 最悪銘柄が浅くてもポートフォリオDDが閾値超過なら落とす
        ev = _make_eval(max_drawdown=-0.01, portfolio_max_drawdown=-0.40)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("portfolio_max_drawdown" in r for r in ev.gate_reasons))
        self.assertFalse(any(r.startswith("max_drawdown") for r in ev.gate_reasons))

    def test_falls_back_to_worst_symbol_drawdown_when_portfolio_is_nan(self):
        # 曲線が1本も取れなかった場合は安全側（従来指標）で判定する
        ev = _make_eval(max_drawdown=-0.60, portfolio_max_drawdown=float("nan"))

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("max_drawdown" in r for r in ev.gate_reasons))

    def test_threshold_is_unchanged(self):
        ev = _make_eval(portfolio_max_drawdown=-0.25)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed, ev.gate_reasons)


class TestChampionGateUsesPortfolioSharpe(unittest.TestCase):
    """champion 比較はプール済み per-trade ベースの Sharpe で行う。

    従来の sharpe_ratio（銘柄別 Sharpe の単純平均）は台帳の再現ペア130組で
    自己相関 0.446 しかなく、97% の候補を落とすゲートの判定基準としては
    ノイズが支配的だった。
    """

    def test_compares_portfolio_sharpe_not_symbol_average(self):
        # 銘柄別平均は champion を超えるが、プール済み Sharpe では超えない
        ev = _make_eval(sharpe_ratio=9.9, portfolio_sharpe_ratio=0.5)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("portfolio_sharpe" in r for r in ev.gate_reasons))

    def test_passes_when_portfolio_sharpe_beats_champion(self):
        ev = _make_eval(sharpe_ratio=-5.0, portfolio_sharpe_ratio=1.5)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed, ev.gate_reasons)

    def test_falls_back_to_symbol_average_when_portfolio_sharpe_is_nan(self):
        ev = _make_eval(sharpe_ratio=0.2, portfolio_sharpe_ratio=float("nan"))

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any(r.startswith("sharpe ") for r in ev.gate_reasons))


if __name__ == "__main__":
    unittest.main()
