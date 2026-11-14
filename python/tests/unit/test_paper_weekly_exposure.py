"""紙の週の成績の投資比率 exposure と、投資比率で合わせた超過の騰落率のユニットテスト（#1792）"""

import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.trading import paper_equity
from src.trading.paper_equity import compute_weekly_results, write_paper_weekly_results

_INITIAL = 1_000_000.0
_COLUMNS = ["market", "symbol", "side", "qty", "fill_price", "filled_at"]


def _orders(rows):
    return pd.DataFrame(rows, columns=_COLUMNS)


def _price_loader(market, symbol):
    """前週金曜（2026-10-09）までは 100 円、今週（2026-10-12〜16）の金曜に 110 円。"""
    idx = pd.bdate_range("2026-09-28", "2026-10-16")
    prices = pd.Series(100.0, index=idx)
    prices[pd.Timestamp("2026-10-16")] = 110.0
    return prices


def _fetcher(ticker, start, end):
    returns = {"1306.T": 0.10, "^GSPC": -0.01}
    return {"ticker": ticker, "total_return": returns[ticker], "start": start, "end": end}


class TestExposureScaledExcess(unittest.TestCase):
    """投資比率で合わせて、ベンチマークと比べること"""

    def setUp(self):
        # 前週に元手の半分（5,000 株 × 100 円）を買い、残りは現金
        self.orders = _orders([("jp", "7203", 1, 5000, 100.0, pd.Timestamp("2026-10-07"))])
        self.result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), _fetcher
        )

    def test_no_holdings_market_has_zero_exposure_and_zero_excess(self):
        us = self.result["markets"]["us"]
        self.assertEqual(us["exposure"], 0.0)
        self.assertEqual(us["excess_return"], 0.0)
        # 全額との差は別の名前で残す
        self.assertAlmostEqual(us["excess_return_unscaled"], 0.01)
        self.assertEqual(us["weekly_return"], 0.0)
        self.assertEqual(us["benchmark_return"], -0.01)

    def test_half_cash_moving_with_benchmark_gives_zero_excess(self):
        jp = self.result["markets"]["jp"]
        # 1,000,000 → 1,000,000 + 5,000 株 × 10 円 = 1,050,000。保有は +10% でベンチマークと同じ
        self.assertAlmostEqual(jp["weekly_return"], 0.05)
        self.assertAlmostEqual(jp["benchmark_return"], 0.10)
        self.assertAlmostEqual(jp["exposure"], 0.5)
        self.assertAlmostEqual(jp["excess_return"], 0.0)
        self.assertAlmostEqual(jp["excess_return_unscaled"], -0.05)

    def test_basis_mark_is_written(self):
        for values in self.result["markets"].values():
            self.assertEqual(values["excess_basis"], paper_equity.EXCESS_BASIS)

    def test_null_benchmark_gives_null_excess(self):
        def failing(ticker, start, end):
            return {"ticker": ticker, "total_return": None, "start": start, "end": end}

        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), failing
        )
        jp = result["markets"]["jp"]
        self.assertIsNone(jp["excess_return"])
        self.assertIsNone(jp["excess_return_unscaled"])
        self.assertAlmostEqual(jp["exposure"], 0.5)


# 木曜の回。対象の週は W44、前の3週は W41〜W43
_AS_OF = date(2026, 11, 5)
_BACKFILL_ORDERS = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))])


def _rising_loader(market, symbol):
    idx = pd.bdate_range("2026-09-21", "2026-11-06")
    return pd.Series([100.0 + i for i in range(len(idx))], index=idx)


def _backfill_fetcher(ticker, start, end):
    returns = {"1306.T": 0.02, "^GSPC": -0.01}
    return {"ticker": ticker, "total_return": returns[ticker], "start": start, "end": end}


def _failing_fetcher(ticker, start, end):
    return {"ticker": ticker, "total_return": None, "start": start, "end": end}


def _week_result(as_of):
    return compute_weekly_results(
        _BACKFILL_ORDERS, _rising_loader, _INITIAL, as_of, _backfill_fetcher
    )


def _seed_old(directory, week, as_of):
    """印の無い、前の計算（全額のベンチマークとの差）の週のファイルを書く。"""
    result = _week_result(as_of)
    for values in result["markets"].values():
        for key in ("exposure", "excess_return_unscaled", "excess_basis"):
            values.pop(key, None)
        values["excess_return"] = round(values["weekly_return"] - values["benchmark_return"], 6)
    with open(os.path.join(directory, f"{week}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _read(directory, week):
    with open(os.path.join(directory, f"{week}.json"), encoding="utf-8") as f:
        return json.load(f)


class TestOldFilesAreRecomputed(unittest.TestCase):
    """印の無い古い週のファイルを、3週の範囲に限らず新しい計算で書き直すこと"""

    def setUp(self):
        for name, kwargs in {
            "_load_filled_orders": {"return_value": _BACKFILL_ORDERS},
            "_default_price_loader": {"side_effect": _rising_loader},
            "_default_split_loader": {"return_value": None},
        }.items():
            patcher = patch.object(paper_equity, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_all_old_weeks_are_rewritten(self):
        weeks = {"2026-W39": date(2026, 9, 26), "2026-W43": date(2026, 10, 24)}
        for week, as_of in weeks.items():
            _seed_old(self.tmp, week, as_of)

        write_paper_weekly_results(_AS_OF, _backfill_fetcher, self.tmp)

        for week, as_of in weeks.items():
            written = _read(self.tmp, week)
            self.assertEqual(written["markets"], _week_result(as_of)["markets"])
            for values in written["markets"].values():
                self.assertEqual(values["excess_basis"], paper_equity.EXCESS_BASIS)
        # W39 は3週の範囲の外。範囲の外のファイルの無い週は作らない
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "2026-W40.json")))
        # us は保有が無いので、ベンチマークの符号を変えた値でなく 0
        self.assertEqual(_read(self.tmp, "2026-W39")["markets"]["us"]["excess_return"], 0.0)

    def test_old_week_is_kept_when_rewrite_loses_values(self):
        old = _seed_old(self.tmp, "2026-W39", date(2026, 9, 26))

        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp)

        kept = _read(self.tmp, "2026-W39")
        self.assertEqual(kept["markets"], json.loads(json.dumps(old["markets"])))
        self.assertIsNotNone(kept["markets"]["jp"]["excess_return"])


if __name__ == "__main__":
    unittest.main()
