"""サンプラーの予算浪費を止める変更のテスト。

台帳 893 件の再集計で判明した2つの無駄:

1. AND/OR は可換だが子の順序が違うとハッシュが変わるため重複排除が効かず、
   同一戦略が別日に二重評価された組が 189 組（全体の 21.2%）あった。
2. 原子の探索空間は 19 通り × 2 市場 = 38 件でとうに枯渇しているのに、
   サンプラーは 40% を原子に振り続け、すべて重複で弾かれていた。
"""

from __future__ import annotations

import unittest

from src.backtest.factory_sampling import (
    _PARAM_GRID,
    _RULE_CLASSES,
    canonical_rule_spec,
    enumerate_atomic_specs,
    sample_hypotheses,
)
from src.backtest.types import FactoryHypothesis


def _atomic(rule: str, **params) -> dict:
    return {"type": "atomic", "rule": rule, "params": params}


class TestCanonicalRuleSpec(unittest.TestCase):
    def test_atomic_spec_is_returned_unchanged(self):
        spec = _atomic("rsi_contrarian", oversold=30.0, overbought=70.0)

        self.assertEqual(canonical_rule_spec(spec), spec)

    def test_children_are_sorted_deterministically(self):
        a = _atomic("bollinger_band", sell_at_upper=False)
        b = _atomic("rsi_contrarian", oversold=25.0, overbought=75.0)

        one = canonical_rule_spec({"type": "and", "rules": [a, b]})
        other = canonical_rule_spec({"type": "and", "rules": [b, a]})

        self.assertEqual(one, other)

    def test_order_permutations_produce_the_same_hash(self):
        """#703 の台帳で 189 組を生んだ現象そのものを固定する。"""
        a = _atomic("bollinger_band", sell_at_upper=True)
        b = _atomic("rsi_contrarian", oversold=25.0, overbought=75.0)

        h1 = FactoryHypothesis(canonical_rule_spec({"type": "and", "rules": [a, b]}), "jp")
        h2 = FactoryHypothesis(canonical_rule_spec({"type": "and", "rules": [b, a]}), "jp")

        self.assertEqual(h1.hypothesis_hash, h2.hypothesis_hash)

    def test_and_and_or_remain_distinct(self):
        a = _atomic("bollinger_band", sell_at_upper=True)
        b = _atomic("rsi_contrarian", oversold=25.0, overbought=75.0)

        self.assertNotEqual(
            canonical_rule_spec({"type": "and", "rules": [a, b]}),
            canonical_rule_spec({"type": "or", "rules": [a, b]}),
        )

    def test_is_idempotent(self):
        a = _atomic("bollinger_band", sell_at_upper=False)
        b = _atomic("macd_rsi", rsi_filter=50.0)
        once = canonical_rule_spec({"type": "or", "rules": [b, a]})

        self.assertEqual(canonical_rule_spec(once), once)

    def test_generated_code_spec_is_left_alone(self):
        spec = {"type": "generated_code", "class_name": "X", "source_code": "class X: pass"}

        self.assertEqual(canonical_rule_spec(spec), spec)


class TestEnumerateAtomicSpecs(unittest.TestCase):
    def test_covers_the_whole_atomic_grid(self):
        specs = enumerate_atomic_specs()

        self.assertEqual(len(specs), sum(len(v) for v in _PARAM_GRID.values()))
        self.assertEqual({s["rule"] for s in specs}, set(_RULE_CLASSES))


class TestSamplerProducesCanonicalSpecs(unittest.TestCase):
    def test_every_sampled_composite_is_canonical(self):
        sampled = sample_hypotheses("jp", budget=40, existing_hashes=set(), seed=3)

        for h in sampled:
            self.assertEqual(h.rule_spec, canonical_rule_spec(h.rule_spec))

    def test_does_not_resample_an_order_permuted_duplicate(self):
        a = _atomic("bollinger_band", sell_at_upper=False)
        b = _atomic("rsi_contrarian", oversold=25.0, overbought=75.0)
        # 台帳には「非整列の順序」で保存されていた、という状況を作る
        already = FactoryHypothesis(canonical_rule_spec({"type": "and", "rules": [b, a]}), "jp")

        sampled = sample_hypotheses(
            "jp", budget=60, existing_hashes={already.hypothesis_hash}, seed=11
        )

        self.assertNotIn(already.hypothesis_hash, {h.hypothesis_hash for h in sampled})


class TestAtomicExhaustion(unittest.TestCase):
    def _exhausted_atomic_hashes(self, market: str) -> set[str]:
        return {
            FactoryHypothesis(spec, market).hypothesis_hash for spec in enumerate_atomic_specs()
        }

    def test_budget_is_filled_with_composites_when_atomic_space_is_exhausted(self):
        """原子が枯渇していても予算を使い切れること。

        従来は 40% の抽選が毎回重複で弾かれ、その分が暗黙に合成へ流れていた。
        """
        seen = self._exhausted_atomic_hashes("jp")

        sampled = sample_hypotheses("jp", budget=10, existing_hashes=seen, seed=5)

        self.assertEqual(len(sampled), 10)
        self.assertTrue(all(h.rule_spec["type"] in ("and", "or") for h in sampled))

    def test_atomic_is_still_sampled_while_the_space_remains(self):
        sampled = sample_hypotheses("jp", budget=60, existing_hashes=set(), seed=5)

        self.assertTrue(any(h.rule_spec["type"] == "atomic" for h in sampled))


if __name__ == "__main__":
    unittest.main()
