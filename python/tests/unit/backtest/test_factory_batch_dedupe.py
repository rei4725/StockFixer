"""run_factory_batch が過去の順序違い重複も重複排除に含めることのテスト。

台帳には AND/OR の子が非整列のまま保存された行がある（189 組）。保存ハッシュは
正準形と一致しないため、それだけを渡すとサンプラーが同じ戦略をもう一度引いてしまう。
保存ハッシュに加え、保存スペックを正規化して再計算したハッシュも渡す必要がある。
"""

from __future__ import annotations

import json
import unittest
import unittest.mock

from src.backtest import factory
from src.backtest.factory_sampling import canonical_rule_spec
from src.backtest.types import FactoryHypothesis

_A = {"type": "atomic", "rule": "bollinger_band", "params": {"sell_at_upper": False}}
_B = {"type": "atomic", "rule": "rsi_contrarian", "params": {"oversold": 25.0, "overbought": 75.0}}
# 台帳に「非整列の順序」で保存されている合成スペック。
# 正準順は json.dumps(sort_keys=True) 昇順であり、params の先頭キーが
# "overbought" < "sell_at_upper" のため _B が先に来る。よって [_A, _B] は非整列。
_STORED_SPEC = {"type": "and", "rules": [_A, _B]}


class TestBatchDedupeIncludesCanonicalHashes(unittest.TestCase):
    def test_canonical_hash_of_stored_spec_is_passed_to_sampler(self):
        stored_hash = FactoryHypothesis(_STORED_SPEC, "jp").hypothesis_hash
        canonical_hash = FactoryHypothesis(canonical_rule_spec(_STORED_SPEC), "jp").hypothesis_hash
        # 前提: 正規化すると別ハッシュになる（これが 189 組を生んだ現象）
        self.assertNotEqual(stored_hash, canonical_hash)

        captured: dict = {}

        def _fake_sample(market, budget, existing_hashes, seed=None, lookback_years=2):
            captured["hashes"] = set(existing_hashes)
            return []

        with unittest.mock.patch.object(
            factory, "load_factory_hashes", return_value={stored_hash}
        ), unittest.mock.patch.object(
            factory, "load_factory_specs", return_value=[("jp", json.dumps(_STORED_SPEC))]
        ), unittest.mock.patch.object(
            factory, "sample_hypotheses", side_effect=_fake_sample
        ), unittest.mock.patch.object(
            factory, "_load_symbol_data", return_value={}
        ):
            factory.run_factory_batch(market="jp", symbols=["X"], budget=1, n_windows=4)

        self.assertIn(stored_hash, captured["hashes"])
        self.assertIn(canonical_hash, captured["hashes"])

    def test_specs_from_other_markets_do_not_leak_in(self):
        captured: dict = {}

        def _fake_sample(market, budget, existing_hashes, seed=None, lookback_years=2):
            captured["hashes"] = set(existing_hashes)
            return []

        us_hash = FactoryHypothesis(canonical_rule_spec(_STORED_SPEC), "us").hypothesis_hash

        with unittest.mock.patch.object(
            factory, "load_factory_hashes", return_value=set()
        ), unittest.mock.patch.object(
            factory, "load_factory_specs", return_value=[("us", json.dumps(_STORED_SPEC))]
        ), unittest.mock.patch.object(
            factory, "sample_hypotheses", side_effect=_fake_sample
        ), unittest.mock.patch.object(
            factory, "_load_symbol_data", return_value={}
        ):
            factory.run_factory_batch(market="jp", symbols=["X"], budget=1, n_windows=4)

        # us 市場のスペックから導いたハッシュは jp のバッチには渡らない
        self.assertNotIn(us_hash, captured["hashes"])

    def test_malformed_spec_json_does_not_break_the_batch(self):
        captured: dict = {}

        def _fake_sample(market, budget, existing_hashes, seed=None, lookback_years=2):
            captured["hashes"] = set(existing_hashes)
            return []

        with unittest.mock.patch.object(
            factory, "load_factory_hashes", return_value={"deadbeef0001"}
        ), unittest.mock.patch.object(
            factory, "load_factory_specs", return_value=[("jp", "{not json")]
        ), unittest.mock.patch.object(
            factory, "sample_hypotheses", side_effect=_fake_sample
        ), unittest.mock.patch.object(
            factory, "_load_symbol_data", return_value={}
        ):
            factory.run_factory_batch(market="jp", symbols=["X"], budget=1, n_windows=4)

        self.assertEqual(captured["hashes"], {"deadbeef0001"})


if __name__ == "__main__":
    unittest.main()
