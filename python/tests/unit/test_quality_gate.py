import unittest
from unittest.mock import patch

from src.screening.quality_gate import apply_quality_gate
from src.screening.types import TrendCandidate


def _candidate(symbol: str, score: float = 0.8) -> TrendCandidate:
    """テスト用の TrendCandidate を最小フィールドで作る。"""
    return TrendCandidate(
        market="us",
        symbol=symbol,
        score=score,
        close=50.0,
        dist_from_52w_high=-0.05,
        above_200dma=True,
        sma200_rising=True,
        return_6m=0.3,
        return_12m=0.6,
        avg_volume=1_000_000.0,
    )


def _fund(**overrides) -> dict:
    """成長・質ゲートを満たす良好な財務 dict（override で悪化させる）。"""
    base = {
        "revenue_cagr_3y": 0.30,
        "roe": 0.25,
        "debt_to_equity": 0.5,
        "market_cap": 5e9,
        "op_margin": 0.20,
        "net_margin": 0.15,
        "op_margin_prev": 0.18,  # 上昇中（悪化していない）
    }
    base.update(overrides)
    return base


def _patch_loader(fund_map):
    """load_fundamentals を (market, symbol)->dict|None のマップでモックする。"""

    def _loader(market, symbol):
        return fund_map.get((market, symbol))

    return patch("src.screening.quality_gate.load_fundamentals", side_effect=_loader)


class TestApplyQualityGate(unittest.TestCase):
    def test_good_candidate_passes(self):
        """成長・質を満たす候補は残る。"""
        cands = [_candidate("GOOD")]
        fmap = {("us", "GOOD"): _fund()}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        self.assertEqual([c.symbol for c in result], ["GOOD"])
        self.assertFalse(result[0].fundamentals_missing)

    def test_low_growth_excluded(self):
        """売上CAGR が下限未満なら除外される。"""
        cands = [_candidate("GOOD"), _candidate("LOWGROWTH")]
        fmap = {
            ("us", "GOOD"): _fund(),
            ("us", "LOWGROWTH"): _fund(revenue_cagr_3y=0.05),
        }
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("LOWGROWTH", codes)

    def test_high_debt_excluded(self):
        """D/E が上限超なら除外される。"""
        cands = [_candidate("HIDEBT")]
        fmap = {("us", "HIDEBT"): _fund(debt_to_equity=3.0)}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        self.assertEqual(result, [])

    def test_large_cap_excluded(self):
        """時価総額が上限超なら除外される。"""
        cands = [_candidate("BIG")]
        fmap = {("us", "BIG"): _fund(market_cap=5e10)}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        self.assertEqual(result, [])

    def test_low_roe_excluded(self):
        """ROE が下限未満なら除外される。"""
        cands = [_candidate("LOWROE")]
        fmap = {("us", "LOWROE"): _fund(roe=0.05)}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        self.assertEqual(result, [])

    def test_declining_margin_excluded(self):
        """営業利益率が前期から悪化していれば除外される。"""
        cands = [_candidate("DECLINE")]
        fmap = {("us", "DECLINE"): _fund(op_margin=0.10, op_margin_prev=0.20)}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        self.assertEqual(result, [])

    def test_declining_margin_allowed_when_check_off(self):
        """require_margin_not_declining=False ならマージン悪化でも残る。"""
        cands = [_candidate("DECLINE")]
        fmap = {("us", "DECLINE"): _fund(op_margin=0.10, op_margin_prev=0.20)}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands, require_margin_not_declining=False)
        self.assertEqual([c.symbol for c in result], ["DECLINE"])

    def test_missing_excluded(self):
        """on_missing='exclude' で財務欠損銘柄は除外される。"""
        cands = [_candidate("GOOD"), _candidate("NOFUND")]
        fmap = {("us", "GOOD"): _fund()}  # NOFUND は欠損
        with _patch_loader(fmap):
            result = apply_quality_gate(cands, on_missing="exclude")
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("NOFUND", codes)

    def test_missing_penalized_but_kept(self):
        """on_missing='penalize' で欠損銘柄は減点されつつ残る。"""
        cands = [_candidate("GOOD", score=0.9), _candidate("NOFUND", score=0.5)]
        fmap = {("us", "GOOD"): _fund()}
        with _patch_loader(fmap):
            result = apply_quality_gate(cands, on_missing="penalize")
        codes = [c.symbol for c in result]
        self.assertIn("NOFUND", codes)
        nofund = next(c for c in result if c.symbol == "NOFUND")
        good = next(c for c in result if c.symbol == "GOOD")
        self.assertTrue(nofund.fundamentals_missing)
        self.assertIsNone(nofund.revenue_cagr_3y)
        # 減点され良好銘柄より下位
        self.assertLess(nofund.multibagger_score, good.multibagger_score)

    def test_sorted_descending(self):
        """合成スコア降順で並ぶ。"""
        cands = [
            _candidate("A", score=0.5),
            _candidate("B", score=0.9),
            _candidate("C", score=0.7),
        ]
        fmap = {
            ("us", "A"): _fund(revenue_cagr_3y=0.20, roe=0.12),
            ("us", "B"): _fund(revenue_cagr_3y=0.40, roe=0.30),
            ("us", "C"): _fund(revenue_cagr_3y=0.30, roe=0.20),
        }
        with _patch_loader(fmap):
            result = apply_quality_gate(cands)
        scores = [c.multibagger_score for c in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(result), 3)

    def test_empty_candidates(self):
        """空候補は安全に空リストを返す。"""
        with _patch_loader({}):
            result = apply_quality_gate([])
        self.assertEqual(result, [])

    def test_all_excluded_returns_empty(self):
        """全候補が財務欠損で除外されても空リストを返す。"""
        cands = [_candidate("NOFUND")]
        with _patch_loader({}):
            result = apply_quality_gate(cands, on_missing="exclude")
        self.assertEqual(result, [])

    def test_invalid_on_missing_raises(self):
        """不正な on_missing は ValueError。"""
        with self.assertRaises(ValueError):
            apply_quality_gate([_candidate("X")], on_missing="bogus")


if __name__ == "__main__":
    unittest.main()
