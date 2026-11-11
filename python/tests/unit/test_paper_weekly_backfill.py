"""紙の週の成績の、前の3週の見直しのユニットテスト（#1695）"""

import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading import paper_equity
from src.trading.paper_equity import compute_weekly_results, write_paper_weekly_results

_INITIAL = 1_000_000.0
_COLUMNS = ["market", "symbol", "side", "qty", "fill_price", "filled_at"]
# 木曜の回。対象の週は W44（10-26〜10-30）、その前の3週は W43・W42・W41
_AS_OF = date(2026, 11, 5)
_WEEK_AS_OF = {
    "2026-W39": date(2026, 9, 26),
    "2026-W40": date(2026, 10, 3),
    "2026-W41": date(2026, 10, 10),
    "2026-W42": date(2026, 10, 17),
    "2026-W43": date(2026, 10, 24),
}
_ORDERS = pd.DataFrame(
    [("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))], columns=_COLUMNS
)


def _price_loader(market, symbol):
    """営業日ごとに 1 円ずつ上がる終値。"""
    idx = pd.bdate_range("2026-09-21", "2026-11-06")
    return pd.Series([100.0 + i for i in range(len(idx))], index=idx)


def _fetcher(ticker, start, end):
    returns = {"1306.T": 0.02, "^GSPC": -0.01}
    return {"ticker": ticker, "total_return": returns[ticker], "start": start, "end": end}


def _failing_fetcher(ticker, start, end):
    return {"ticker": ticker, "total_return": None, "start": start, "end": end}


def _result(week, **jp_fields):
    """その週の as_of で計算した内容に、jp の値を上書きしたものを返す。"""
    result = compute_weekly_results(_ORDERS, _price_loader, _INITIAL, _WEEK_AS_OF[week], _fetcher)
    result["markets"]["jp"].update(jp_fields)
    return result


def _seed(directory, week, **jp_fields):
    with open(os.path.join(directory, f"{week}.json"), "w", encoding="utf-8") as f:
        json.dump(_result(week, **jp_fields), f, ensure_ascii=False, indent=2)


def _read(directory, week):
    with open(os.path.join(directory, f"{week}.json"), encoding="utf-8") as f:
        return json.load(f)


def _weeks(directory):
    return sorted(name[:-5] for name in os.listdir(directory) if name.endswith(".json"))


class _Base(unittest.TestCase):
    def setUp(self):
        for name, kwargs in {
            "_load_filled_orders": {"return_value": _ORDERS},
            "_default_price_loader": {"side_effect": _price_loader},
            "_default_split_loader": {"return_value": None},
        }.items():
            patcher = patch.object(paper_equity, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name


class TestRewrite(_Base):
    """ファイルの無い週・null の週・部分の週を計算し直して書くこと"""

    def test_missing_week_is_same_as_direct_write(self):
        _seed(self.tmp, "2026-W41")

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertEqual(_weeks(self.tmp), ["2026-W41", "2026-W42", "2026-W43", "2026-W44"])
        with tempfile.TemporaryDirectory() as direct:
            for week in ("2026-W42", "2026-W43"):
                write_paper_weekly_results(_WEEK_AS_OF[week], _fetcher, direct)
                self.assertEqual(_read(self.tmp, week), _read(direct, week))

    def test_null_excess_is_filled(self):
        _seed(self.tmp, "2026-W43", benchmark_return=None, excess_return=None)

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        expected = _result("2026-W43")["markets"]["jp"]
        self.assertEqual(_read(self.tmp, "2026-W43")["markets"]["jp"], expected)
        self.assertIsNotNone(expected["excess_return"])

    def test_partial_week_becomes_false(self):
        _seed(self.tmp, "2026-W43", partial_week=True, missing_week_end_close=["7203"])

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        jp = _read(self.tmp, "2026-W43")["markets"]["jp"]
        self.assertFalse(jp["partial_week"])
        self.assertEqual(jp["missing_week_end_close"], [])

    def test_broken_file_is_written_again(self):
        with open(os.path.join(self.tmp, "2026-W43.json"), "w", encoding="utf-8") as f:
            f.write('{"week": ')

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertEqual(_read(self.tmp, "2026-W43")["week"], "2026-W43")

    def test_rewrite_that_gets_worse_keeps_the_file(self):
        """us だけ null の週を、両方 null になる回の結果で上書きしないこと"""
        result = _result("2026-W43")
        result["markets"]["us"].update(benchmark_return=None, excess_return=None)
        with open(os.path.join(self.tmp, "2026-W43.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)

        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp)

        self.assertIsNotNone(_read(self.tmp, "2026-W43")["markets"]["jp"]["excess_return"])


class TestScope(_Base):
    """見る週の範囲"""

    def test_clean_file_is_not_rewritten(self):
        _seed(self.tmp, "2026-W43", weekly_return=0.123)
        path = os.path.join(self.tmp, "2026-W43.json")
        with open(path, "rb") as f:
            before = f.read()

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_no_file_writes_only_target_week(self):
        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertEqual(_weeks(self.tmp), ["2026-W44"])

    def test_week_before_oldest_file_is_not_created(self):
        _seed(self.tmp, "2026-W43")

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertEqual(_weeks(self.tmp), ["2026-W43", "2026-W44"])

    def test_only_three_previous_weeks_are_seen(self):
        _seed(self.tmp, "2026-W40", benchmark_return=None, excess_return=None)

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertEqual(
            _weeks(self.tmp),
            ["2026-W40", "2026-W41", "2026-W42", "2026-W43", "2026-W44"],
        )
        self.assertIsNone(_read(self.tmp, "2026-W40")["markets"]["jp"]["excess_return"])

    def test_week_older_than_three_is_not_created(self):
        _seed(self.tmp, "2026-W39")

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertNotIn("2026-W40", _weeks(self.tmp))


class TestNotice(_Base):
    """書き直しても残る週の知らせ"""

    def setUp(self):
        super().setUp()
        _seed(self.tmp, "2026-W43", benchmark_return=None, excess_return=None)

    def test_remaining_week_is_notified_with_name_and_reason(self):
        notify = MagicMock(return_value=True)

        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)

        notify.assert_called_once()
        text = notify.call_args[0][0]
        self.assertIn("2026-W43", text)
        self.assertIn("jp 超過の騰落率が null", text)
        # 対象の週は次の回に見直すので知らせない
        self.assertNotIn("2026-W44", text)

    def test_partial_week_reason_names_the_symbols(self):
        _seed(self.tmp, "2026-W43", partial_week=True, missing_week_end_close=["7203"])
        notify = MagicMock(return_value=True)

        def old_prices(market, symbol):
            return _price_loader(market, symbol)[:"2026-10-22"]

        with patch.object(paper_equity, "_default_price_loader", side_effect=old_prices):
            write_paper_weekly_results(_AS_OF, _fetcher, self.tmp, notify=notify)

        self.assertIn("2026-W43: jp 部分の週（7203）", notify.call_args[0][0])

    def test_filled_week_is_not_notified(self):
        notify = MagicMock(return_value=True)

        write_paper_weekly_results(_AS_OF, _fetcher, self.tmp, notify=notify)

        notify.assert_not_called()

    def test_second_run_adds_no_notice_and_same_result(self):
        """補完の回に同じ週を見ても、結果も知らせの数も増えないこと"""
        notify = MagicMock(return_value=True)

        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)
        first = _read(self.tmp, "2026-W43")
        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)

        notify.assert_called_once()
        self.assertEqual(_read(self.tmp, "2026-W43"), first)
        self.assertEqual(_weeks(self.tmp), ["2026-W43", "2026-W44"])

    def test_new_week_is_notified_once_more(self):
        notify = MagicMock(return_value=True)

        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)
        # 次の週の回: W44 が前の週に入り、W43 はもう知らせてある
        write_paper_weekly_results(date(2026, 11, 12), _failing_fetcher, self.tmp, notify=notify)

        self.assertEqual(notify.call_count, 2)
        second = notify.call_args[0][0]
        self.assertIn("2026-W44", second)
        self.assertNotIn("2026-W43", second)

    def test_unsent_notice_is_sent_on_the_next_run(self):
        notify = MagicMock(return_value=False)

        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)
        write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)

        self.assertEqual(notify.call_count, 2)

    def test_notice_failure_does_not_stop_writing(self):
        notify = MagicMock(side_effect=RuntimeError("discord"))

        path = write_paper_weekly_results(_AS_OF, _failing_fetcher, self.tmp, notify=notify)

        self.assertEqual(path, os.path.join(self.tmp, "2026-W44.json"))
        self.assertTrue(os.path.exists(path))


class TestFailure(_Base):
    """前の週の書き直しが落ちても対象の週のファイルが残ること"""

    def test_failed_rewrite_keeps_target_and_notifies(self):
        _seed(self.tmp, "2026-W43", benchmark_return=None, excess_return=None)
        real = paper_equity.compute_weekly_results
        calls = []

        def compute(*args, **kwargs):
            calls.append(args)
            if len(calls) > 1:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        notify = MagicMock(return_value=True)
        with patch.object(paper_equity, "compute_weekly_results", side_effect=compute):
            path = write_paper_weekly_results(_AS_OF, _fetcher, self.tmp, notify=notify)

        self.assertEqual(path, os.path.join(self.tmp, "2026-W44.json"))
        self.assertTrue(os.path.exists(path))
        text = notify.call_args[0][0]
        self.assertIn("2026-W43", text)
        self.assertIn("RuntimeError", text)

    def test_failed_listing_keeps_target(self):
        real = os.listdir

        def listdir(directory):
            if directory == self.tmp:
                raise OSError("disk")
            return real(directory)

        with patch("src.trading.paper_weekly_backfill.os.listdir", side_effect=listdir):
            path = write_paper_weekly_results(_AS_OF, _fetcher, self.tmp)

        self.assertTrue(os.path.exists(path))


class TestSplitAfterWeek(unittest.TestCase):
    """週より後の分割でも、前の週を計算し直した値が実際の値動きどおりになること"""

    def test_split_after_week_end_follows_actual_move(self):
        orders = pd.DataFrame(
            [("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))], columns=_COLUMNS
        )

        def loader(market, symbol):
            # 実際は前週金曜 100 円 → 週の金曜 110 円。後の分割で全期間が 2 で割られている
            idx = pd.bdate_range("2026-09-28", "2026-11-06")
            prices = pd.Series(100.0, index=idx)
            prices[pd.Timestamp("2026-10-16") :] = 110.0
            return prices / 2.0

        def splits(market, symbol):
            return pd.Series([2.0], index=[pd.Timestamp("2026-10-28")])

        result = compute_weekly_results(
            orders, loader, _INITIAL, date(2026, 10, 17), _fetcher, split_loader=splits
        )
        # 1,000 株 × 10 円 = +10,000
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], 0.01)


class TestJob(unittest.TestCase):
    """週の道が Discord の知らせを渡し、残る週があっても成功のままであること"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    @patch("src.trading.paper_equity.write_paper_weekly_results", return_value="2026-W44.json")
    def test_job_passes_webhook_as_notify(self, mock_write, mock_send):
        from src.orchestration.jobs.weekly import run_weekly_paper_results

        self.assertTrue(run_weekly_paper_results())
        self.assertIs(mock_write.call_args.kwargs["notify"], mock_send)
        mock_send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
