"""value_screener のユニットテスト。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.screening.value_screener import screen_value_candidates


def _row(
    symbol,
    market="jp",
    trailing_pe=8.0,
    payout_ratio=0.20,
    debt_to_equity=50.0,
    net_income=100.0,
    market_cap=1000.0,
):
    """screen_value_candidates が読む DataFrame の1行分を辞書で作る。"""
    return {
        "market": market,
        "symbol": symbol,
        "trailing_pe": trailing_pe,
        "payout_ratio": payout_ratio,
        "debt_to_equity": debt_to_equity,
        "net_income": net_income,
        "market_cap": market_cap,
    }


def _patch_loader(rows):
    """load_all_fundamentals を rows から作った DataFrame でモックする。"""
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return patch("src.screening.value_screener.load_all_fundamentals", return_value=df)


class TestScreenValueCandidates(unittest.TestCase):
    def test_good_candidate_passes(self):
        rows = [_row("GOOD")]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        self.assertEqual([c.symbol for c in result], ["GOOD"])
        self.assertEqual(result[0].trailing_pe, 8.0)
        self.assertEqual(result[0].payout_ratio, 0.20)
        self.assertEqual(result[0].market_cap, 1000.0)

    def test_high_per_excluded(self):
        rows = [_row("GOOD"), _row("HIPER", trailing_pe=50.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_per=10.0)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("HIPER", codes)

    def test_high_payout_ratio_excluded(self):
        rows = [_row("GOOD"), _row("HIPAYOUT", payout_ratio=0.80)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_payout_ratio=0.30)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("HIPAYOUT", codes)

    def test_high_debt_to_equity_excluded(self):
        rows = [_row("GOOD"), _row("HIDEBT", debt_to_equity=200.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_debt_to_equity=100.0)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("HIDEBT", codes)

    def test_unprofitable_excluded(self):
        rows = [_row("GOOD"), _row("LOSS", net_income=-10.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("LOSS", codes)

    def test_missing_field_excluded(self):
        rows = [_row("GOOD"), _row("MISSING", trailing_pe=None)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("MISSING", codes)

    def test_sorted_by_per_ascending(self):
        rows = [
            _row("HIGH", trailing_pe=9.0),
            _row("LOW", trailing_pe=3.0),
            _row("MID", trailing_pe=6.0),
        ]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        self.assertEqual([c.symbol for c in result], ["LOW", "MID", "HIGH"])

    def test_top_n_limits_results(self):
        rows = [_row(f"S{i}", trailing_pe=float(i)) for i in range(1, 6)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", top_n=2)
        self.assertEqual([c.symbol for c in result], ["S1", "S2"])

    def test_other_market_excluded(self):
        rows = [_row("JPGOOD", market="jp"), _row("USGOOD", market="us")]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        self.assertEqual([c.symbol for c in result], ["JPGOOD"])

    def test_empty_fundamentals_returns_empty(self):
        with _patch_loader([]):
            result = screen_value_candidates(market="jp")
        self.assertEqual(result, [])

    def test_no_candidates_pass_returns_empty(self):
        rows = [_row("HIPER", trailing_pe=999.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_per=10.0)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
