"""紙の売買の週の成績（results/paper/YYYY-Www.json）のユニットテスト（#812）"""

import json
import os
import tempfile
import unittest
from datetime import date, datetime
from unittest.mock import patch

import pandas as pd

from src.backtest.metrics import fetch_benchmark_returns
from src.trading.paper_equity import compute_weekly_results, write_paper_weekly_results

_INITIAL = 1_000_000.0
_COLUMNS = ["market", "symbol", "side", "qty", "fill_price", "filled_at"]


def _orders(rows):
    return pd.DataFrame(rows, columns=_COLUMNS)


def _price_loader(market, symbol):
    """前週金曜（2026-10-09）は 100 円、今週（2026-10-12〜16）は日ごとに上がり金曜 110 円。"""
    idx = pd.bdate_range("2026-09-28", "2026-10-16")
    prices = pd.Series(100.0, index=idx)
    prices[pd.Timestamp("2026-10-16")] = 110.0
    return prices


def _benchmark_fetcher(ticker, start, end):
    returns = {"1306.T": 0.02, "^GSPC": -0.01}
    return {"ticker": ticker, "total_return": returns[ticker], "start": start, "end": end}


class TestComputeWeeklyResults(unittest.TestCase):
    """決まった約定と終値から週の4つの値が出ること"""

    def setUp(self):
        # jp: 前週に 1,000 株を 100 円で買い、今週に 500 株を 100 円で買い増す（約定 1 件）
        self.orders = _orders(
            [
                ("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07")),
                ("jp", "7203", 1, 500, 100.0, pd.Timestamp("2026-10-13")),
            ]
        )

    def test_values_per_market(self):
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )

        self.assertEqual(result["week"], "2026-W42")
        jp = result["markets"]["jp"]
        # 前週末の評価額 1,000,000 → 今週末 1,000,000 + 1,500 株 × 10 円 = 1,015,000
        self.assertAlmostEqual(jp["weekly_return"], 0.015)
        self.assertAlmostEqual(jp["benchmark_return"], 0.02)
        self.assertAlmostEqual(jp["excess_return"], -0.005)
        self.assertEqual(jp["fills"], 1)
        self.assertEqual(jp["benchmark_ticker"], "1306.T")

        us = result["markets"]["us"]
        self.assertAlmostEqual(us["weekly_return"], 0.0)
        self.assertAlmostEqual(us["benchmark_return"], -0.01)
        self.assertAlmostEqual(us["excess_return"], 0.01)
        self.assertEqual(us["fills"], 0)
        self.assertEqual(us["benchmark_ticker"], "^GSPC")

    def test_benchmark_period_is_previous_week_end_to_this_week_end(self):
        calls = []

        def fetcher(ticker, start, end):
            calls.append((ticker, start, end))
            return {"ticker": ticker, "total_return": 0.0, "start": start, "end": end}

        compute_weekly_results(self.orders, _price_loader, _INITIAL, date(2026, 10, 17), fetcher)

        # yfinance の end は含まないため、金曜の翌日を渡す
        self.assertIn(("1306.T", "2026-10-09", "2026-10-17"), calls)

    def test_benchmark_unavailable_gives_null(self):
        def failing(ticker, start, end):
            return {"ticker": ticker, "total_return": None, "start": start, "end": end}

        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), failing
        )
        jp = result["markets"]["jp"]
        self.assertIsNone(jp["benchmark_return"])
        self.assertIsNone(jp["excess_return"])
        self.assertAlmostEqual(jp["weekly_return"], 0.015)

    def test_benchmark_exception_gives_null(self):
        def raising(ticker, start, end):
            raise RuntimeError("network")

        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), raising
        )
        self.assertIsNone(result["markets"]["jp"]["benchmark_return"])
        self.assertIsNone(result["markets"]["jp"]["excess_return"])

    def test_first_week_uses_initial_balance_as_base(self):
        """前週末より前に約定が無い週は、元手を基準にすること"""
        orders = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-13"))])
        result = compute_weekly_results(
            orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )
        # 1,000 株 × 10 円 = +10,000
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], 0.01)
        self.assertEqual(result["markets"]["jp"]["valued_at_fill_price"], [])

    def test_short_week_moves_by_short_pnl(self):
        """空売りを含む週は、週の騰落率が空売りの損益の分だけ動くこと（#1163）"""
        # 前週に 1,000 株を 100 円で空売りし、今週の金曜に終値が 110 円へ上がる
        orders = _orders([("jp", "7203", 3, 1000, 100.0, pd.Timestamp("2026-10-07"))])
        result = compute_weekly_results(
            orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )
        # 前週末 1,000,000 → 今週末 1,000,000 − 1,000 株 × 10 円 = 990,000
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], -0.01)

    def test_short_opened_in_week_does_not_add_proceeds(self):
        """週の中に空売りを建てても、売り代金が週の騰落率に入らないこと"""
        orders = _orders([("jp", "7203", 3, 1000, 100.0, pd.Timestamp("2026-10-13"))])
        result = compute_weekly_results(
            orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], -0.01)

    def test_missing_price_is_not_counted_as_zero(self):
        """値段の取れない保有は0円でなく約定の値段で評価し、その銘柄を結果に残すこと（#850）"""
        orders = _orders(
            [
                ("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-13")),
                ("jp", "9999", 1, 1000, 100.0, pd.Timestamp("2026-10-13")),
            ]
        )

        def loader(market, symbol):
            return _price_loader(market, symbol) if symbol == "7203" else None

        result = compute_weekly_results(
            orders, loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )
        jp = result["markets"]["jp"]
        # 7203 の +10,000 だけ。9999 は約定の 100 円で評価し、代金 100,000 を損に数えない
        self.assertAlmostEqual(jp["weekly_return"], 0.01)
        self.assertAlmostEqual(jp["excess_return"], -0.01)
        self.assertEqual(jp["valued_at_fill_price"], ["9999"])
        self.assertEqual(result["markets"]["us"]["valued_at_fill_price"], [])

    def test_split_in_week_follows_actual_move(self):
        """週の中に 1:2 の分割があり、終値が全期間で調整し直してあっても、週の騰落率が実際の値動きどおりになること（#899）"""
        orders = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))])

        def adjusted_loader(market, symbol):
            # 実際は前週金曜 100 円 → 今週金曜 110 円。調整し直した足は全期間を 2 で割る
            return _price_loader(market, symbol) / 2.0

        def splits(market, symbol):
            return pd.Series([2.0], index=[pd.Timestamp("2026-10-14")])

        result = compute_weekly_results(
            orders,
            adjusted_loader,
            _INITIAL,
            date(2026, 10, 17),
            _benchmark_fetcher,
            split_loader=splits,
        )
        # 1,000 株 × 10 円 = +10,000
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], 0.01)


class TestClosedWeek(unittest.TestCase):
    """週の途中に動いても、直前に終わった週（前の ISO 週の月〜金）を書くこと（#919）"""

    def setUp(self):
        self.orders = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))])

    def test_wednesday_utc_of_thursday_jst_run_gives_previous_week(self):
        # 木曜 02:00 JST は UTC では水曜 2026-10-21
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 21), _benchmark_fetcher
        )
        self.assertEqual(result["week"], "2026-W42")
        self.assertEqual(result["start"], "2026-10-12")
        self.assertEqual(result["end"], "2026-10-16")
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], 0.01)

    def test_thursday_jst_gives_previous_week(self):
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 22), _benchmark_fetcher
        )
        self.assertEqual(result["week"], "2026-W42")
        self.assertEqual(result["end"], "2026-10-16")

    def test_friday_does_not_write_the_unfinished_week(self):
        """金曜の当日は、その週の終値がまだ無いため前の週を書くこと"""
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 16), _benchmark_fetcher
        )
        self.assertEqual(result["week"], "2026-W41")
        self.assertEqual(result["end"], "2026-10-09")

    def test_missing_week_end_close_is_marked_partial(self):
        """週の終わりの日の終値がまだ無い保有があれば、部分の週と分かる印を残すこと"""
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 24), _benchmark_fetcher
        )
        # 終値は 2026-10-16 までしか無い
        self.assertEqual(result["week"], "2026-W43")
        self.assertTrue(result["markets"]["jp"]["partial_week"])
        self.assertEqual(result["markets"]["jp"]["missing_week_end_close"], ["7203"])
        self.assertFalse(result["markets"]["us"]["partial_week"])

    def test_week_with_week_end_close_is_not_partial(self):
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )
        self.assertFalse(result["markets"]["jp"]["partial_week"])
        self.assertEqual(result["markets"]["jp"]["missing_week_end_close"], [])

    def test_job_passes_japan_date(self):
        """週の道は、UTC でなく日本時間の日付を as_of に渡すこと"""
        from src.orchestration.jobs.weekly import run_weekly_paper_results
        from src.utils.japan_time import JAPAN_TIMEZONE

        thursday_2am_jst = datetime(2026, 10, 22, 2, 0, tzinfo=JAPAN_TIMEZONE)
        with patch("src.utils.japan_time.now_jst", return_value=thursday_2am_jst), patch(
            "src.trading.paper_equity.write_paper_weekly_results"
        ) as mock_write:
            run_weekly_paper_results()
        self.assertEqual(mock_write.call_args[0][0], date(2026, 10, 22))


class TestBenchmarkNullReason(unittest.TestCase):
    """ベンチマークが取れないときは WARNING を出して null、取れれば超過の騰落率を出すこと（#919）"""

    def setUp(self):
        self.orders = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))])

    def test_empty_benchmark_warns_and_gives_null(self):
        def empty(ticker, start, end):
            return {"ticker": ticker, "total_return": None, "start": start, "end": end}

        with self.assertLogs("src.trading.paper_equity", level="WARNING") as logs:
            result = compute_weekly_results(
                self.orders, _price_loader, _INITIAL, date(2026, 10, 17), empty
            )
        self.assertIsNone(result["markets"]["jp"]["benchmark_return"])
        self.assertIsNone(result["markets"]["jp"]["excess_return"])
        self.assertTrue(any("1306.T" in line for line in logs.output))

    @patch("yfinance.download")
    def test_multiindex_columns_give_return_and_excess(self, mock_download):
        """yfinance が列を多重の見出しで返しても、ベンチマークの騰落率が取れること"""
        idx = pd.to_datetime(["2026-10-09", "2026-10-12", "2026-10-16"])

        def download(ticker, **kwargs):
            columns = pd.MultiIndex.from_tuples([("Close", ticker), ("Open", ticker)])
            return pd.DataFrame(
                [[100.0, 100.0], [101.0, 101.0], [104.0, 104.0]], index=idx, columns=columns
            )

        mock_download.side_effect = download
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), fetch_benchmark_returns
        )
        jp = result["markets"]["jp"]
        self.assertAlmostEqual(jp["benchmark_return"], 0.04)
        self.assertAlmostEqual(jp["excess_return"], jp["weekly_return"] - jp["benchmark_return"])
        self.assertAlmostEqual(jp["excess_return"], -0.03)

    @patch("yfinance.download", return_value=pd.DataFrame())
    def test_empty_download_warns_and_gives_null(self, _mock_download):
        with self.assertLogs("src.trading.paper_equity", level="WARNING"):
            result = compute_weekly_results(
                self.orders, _price_loader, _INITIAL, date(2026, 10, 17), fetch_benchmark_returns
            )
        self.assertIsNone(result["markets"]["us"]["benchmark_return"])
        self.assertIsNone(result["markets"]["us"]["excess_return"])


def _download_from(closes):
    """yfinance と同じく start 以上・end 未満の行だけを返す download の偽物。"""
    series = pd.Series(closes)
    series.index = pd.to_datetime(series.index)

    def download(ticker, start, end, **kwargs):
        rows = series[(series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))]
        return pd.DataFrame({"Close": rows})

    return download


class TestBenchmarkBaseOnHoliday(unittest.TestCase):
    """ベンチマークの起点は、紙の売買の評価額と同じ base_day 以前の最後の終値にすること（#1076）"""

    def setUp(self):
        self.orders = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))])

    @patch("yfinance.download")
    def test_holiday_base_day_uses_previous_close(self, mock_download):
        # base_day の 2026-10-09（金）が休場。行は木曜と月曜〜金曜にある
        mock_download.side_effect = _download_from(
            {
                "2026-10-08": 100.0,
                "2026-10-12": 102.0,
                "2026-10-13": 103.0,
                "2026-10-14": 104.0,
                "2026-10-15": 104.0,
                "2026-10-16": 105.0,
            }
        )
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), fetch_benchmark_returns
        )
        # 木曜 100 → 金曜 105（月曜起点なら 105 / 102 - 1）
        self.assertAlmostEqual(result["markets"]["jp"]["benchmark_return"], 0.05)
        self.assertAlmostEqual(result["markets"]["us"]["benchmark_return"], 0.05)

    @patch("yfinance.download")
    def test_trading_base_day_keeps_previous_value(self, mock_download):
        mock_download.side_effect = _download_from(
            {
                "2026-10-08": 90.0,
                "2026-10-09": 100.0,
                "2026-10-12": 102.0,
                "2026-10-16": 103.0,
            }
        )
        expected = fetch_benchmark_returns("1306.T", "2026-10-09", "2026-10-17")["total_return"]
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), fetch_benchmark_returns
        )
        self.assertEqual(result["markets"]["jp"]["benchmark_return"], expected)
        self.assertAlmostEqual(result["markets"]["jp"]["benchmark_return"], 0.03)


class TestWritePaperWeeklyResults(unittest.TestCase):
    """週の道からファイルに書くこと"""

    @patch("src.trading.paper_equity._default_split_loader", return_value=None)
    @patch("src.trading.paper_equity._default_price_loader", side_effect=_price_loader)
    @patch("src.trading.paper_equity._load_filled_orders")
    def test_writes_iso_week_json(self, mock_load, _mock_prices, _mock_splits):
        mock_load.return_value = _orders(
            [("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_paper_weekly_results(date(2026, 10, 17), _benchmark_fetcher, tmp)
            self.assertEqual(path, os.path.join(tmp, "2026-W42.json"))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertAlmostEqual(data["markets"]["jp"]["weekly_return"], 0.01)
        self.assertEqual(data["markets"]["jp"]["fills"], 0)

    @patch("src.trading.paper_equity._load_filled_orders")
    def test_no_orders_writes_nothing(self, mock_load):
        mock_load.return_value = _orders([])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_paper_weekly_results(date(2026, 10, 17), _benchmark_fetcher, tmp)
            self.assertIsNone(path)
            self.assertEqual(os.listdir(tmp), [])


class TestWeeklyReportCallsPaperResults(unittest.TestCase):
    """週のレポートの道が週の成績を書き、失敗しても止まらないこと"""

    @patch("src.trading.paper_equity.write_paper_weekly_results", side_effect=RuntimeError("x"))
    def test_failure_does_not_propagate(self, mock_write):
        from src.orchestration.jobs.weekly import run_weekly_paper_results

        run_weekly_paper_results()  # 例外が外に出ないこと
        mock_write.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    @patch("src.trading.paper_equity.write_paper_weekly_results", side_effect=RuntimeError("x"))
    def test_failure_sends_notice(self, _mock_write, mock_send):
        """書き出しが落ちた回は、失敗の知らせを送ること（#999）"""
        from src.orchestration.jobs.weekly import run_weekly_paper_results

        run_weekly_paper_results()
        mock_send.assert_called_once()
        text = mock_send.call_args[0][0]
        self.assertIn("❌ 紙の売買の週の成績の書き出し失敗", text)
        self.assertIn("x", text)

    @patch("src.reporting.discord.discord_utils.send_webhook_text", side_effect=RuntimeError("y"))
    @patch("src.trading.paper_equity.write_paper_weekly_results", side_effect=RuntimeError("x"))
    def test_notice_failure_does_not_propagate(self, _mock_write, mock_send):
        """知らせの送信が落ちても、例外が外に出ないこと（#999）"""
        from src.orchestration.jobs.weekly import run_weekly_paper_results

        run_weekly_paper_results()
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    @patch("src.trading.paper_equity.write_paper_weekly_results", return_value="2026-W43.json")
    def test_success_sends_no_notice(self, mock_write, mock_send):
        """書き出しが終わった回は、失敗の知らせを送らないこと（#999）"""
        from src.orchestration.jobs.weekly import run_weekly_paper_results

        run_weekly_paper_results()
        mock_write.assert_called_once()
        mock_send.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    @patch("src.trading.paper_equity.write_paper_weekly_results", side_effect=RuntimeError("x"))
    @patch("src.reporting.discord.discord_utils.send_miss_analysis_summary")
    @patch("src.prediction.miss_analysis.run_miss_analysis_batch")
    @patch("src.prediction.db.load_top_prediction_misses")
    @patch("src.reporting.discord.discord_utils.send_weekly_report")
    @patch("src.reporting.llm_review.generate_weekly_review", return_value=None)
    @patch("src.infrastructure.llm.factory.get_text_review_port")
    @patch("src.prediction.db.save_weekly_accuracy_snapshot")
    @patch("src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery")
    def test_weekly_report_runs_all_stages_and_notifies(
        self,
        _mock_analytics,
        _mock_snapshot,
        _mock_port,
        _mock_review,
        mock_report,
        _mock_misses,
        _mock_batch,
        mock_miss_summary,
        mock_write,
        mock_send,
    ):
        """書き出しが落ちても、週のレポートと外れの分析が走り、ジョブは例外を上げないこと（#999）"""
        from src.orchestration.jobs.weekly import run_weekly_report

        run_weekly_report()
        mock_report.assert_called_once()
        mock_miss_summary.assert_called_once()
        mock_write.assert_called_once()
        mock_send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
