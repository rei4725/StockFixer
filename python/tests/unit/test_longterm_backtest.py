"""長期コホート・バックテスト(#431)の単体テスト。

合成の生 OHLCV（既知の価格軌道）を load_raw_closes モックで与え、
fetch_benchmark_returns もモックして検証する。
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest import longterm_backtest as lb
from src.backtest.longterm import engine as lb_engine
from src.backtest.longterm import prices as lb_prices
from src.screening.types import TrendCandidate

_PHASE_A = 400  # トレンド形成期間（営業日）
_PHASE_B = 500  # ホールド期間（営業日）
_N = _PHASE_A + _PHASE_B
_DATES = pd.date_range("2020-01-01", periods=_N, freq="B").strftime("%Y-%m-%d").tolist()
_ENTRY_DATE = _DATES[_PHASE_A]  # 最初のリスクリーン日＝Phase B 開始日
_END = _DATES[-1]


def _features(closes) -> pd.DataFrame:
    """load_raw_ohlcv 風（Date index・Close 列）の合成 OHLCV を作る。"""
    closes = np.asarray(closes, dtype=float)
    idx = pd.to_datetime(_DATES)
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(_N, 1_000_000.0),
        },
        index=idx,
    )


def _phase_a():
    """Phase A: 50→100 の緩やかな上昇（トレンド形成・100で着地）。"""
    return np.linspace(50.0, 100.0, _PHASE_A)


def _mid():
    """Phase B で 100→300（3倍到達）。"""
    return np.concatenate([_phase_a(), np.linspace(100.0, 300.0, _PHASE_B)])


def _high():
    """Phase B で 100→800（8倍到達）。"""
    return np.concatenate([_phase_a(), np.linspace(100.0, 800.0, _PHASE_B)])


def _low():
    """Phase B で 100→150（1.5倍・2倍未到達、最終日までホールド）。"""
    return np.concatenate([_phase_a(), np.linspace(100.0, 150.0, _PHASE_B)])


def _crash():
    """Phase B で 100→105 のあと 105→40 へ崩落（撤退・実現損）。"""
    up = np.linspace(100.0, 105.0, 50)
    down = np.linspace(105.0, 40.0, _PHASE_B - 50)
    return np.concatenate([_phase_a(), up, down])


_DF_MAP = {
    "MID": _features(_mid()),
    "HIGH": _features(_high()),
    "LOW": _features(_low()),
    "CRASH": _features(_crash()),
}
_SCORES = {"HIGH": 0.9, "MID": 0.8, "LOW": 0.6, "CRASH": 0.7}


def _make_screen(symbols, call_log=None):
    """指定 symbols を score 降順で返す擬似スクリーン。as_of を記録する。"""

    def _screen(market="us", top_n=30, as_of=None, **kwargs):
        if call_log is not None:
            call_log.append(as_of)
        ordered = sorted(symbols, key=lambda s: _SCORES.get(s, 0.0), reverse=True)
        return [
            TrendCandidate(
                market=market,
                symbol=s,
                score=_SCORES.get(s, 0.5),
                close=100.0,
                dist_from_52w_high=0.0,
                above_200dma=True,
                sma200_rising=True,
                return_6m=0.5,
                return_12m=1.0,
                avg_volume=1_000_000.0,
            )
            for s in ordered
        ][:top_n]

    return _screen


def _closes_frame(symbols):
    """load_raw_closes 風（columns=[symbol, ts, close]）の合成フレームを作る。"""
    frames = [
        pd.DataFrame(
            {
                "symbol": s,
                "ts": _DF_MAP[s].index,
                "close": _DF_MAP[s]["Close"].to_numpy(),
            }
        )
        for s in symbols
        if s in _DF_MAP
    ]
    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "close"])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"])


def _run(symbols, screen=None, call_log=None, **kwargs):
    screen = screen or _make_screen(symbols, call_log)

    def _load_closes(market, start_date=None, end_date=None, timeframe="1d"):
        df = _closes_frame(symbols)
        if end_date is not None and not df.empty:
            df = df[df["ts"] <= pd.Timestamp(end_date)]
        return df.reset_index(drop=True)

    with patch.object(lb_prices, "load_raw_closes", side_effect=_load_closes), patch.object(
        lb_engine, "screen_trend_candidates", side_effect=screen
    ), patch.object(
        lb_engine,
        "fetch_benchmark_returns",
        return_value={"ticker": "^GSPC", "total_return": 0.5, "start": "", "end": ""},
    ):
        return lb.run_longterm_backtest(
            market="us",
            start=_ENTRY_DATE,
            end=_END,
            rescreen_freq="quarterly",
            **kwargs,
        )


class TestLongtermBacktest(unittest.TestCase):
    def test_multiple_distribution(self):
        """既知の 1.5倍・3倍・8倍軌道（全て満期保有）で到達倍率カウントが正しい。

        execution_lag（既定 1）導入により、エントリーはスクリーン日
        （Phase B 開始日＝価格 100.0 の日）の翌営業日に 1 日ずれる。
        Phase B は linspace(start, 300/800/150, 500) の等差数列なので
        1 ステップ = (終値-100)/499 だけ entry_price が始値より高くなり、
        期待倍率は「区間の最終日（=最高値、単調増加のため）÷ entry_price」
        として厳密に計算し直せる:
            MID:  300 / (100 + 200/499) = 1497/501  ≈ 2.9880（旧 3.0）
            HIGH: 800 / (100 + 700/499) = 399200/50600 ≈ 7.8893（旧 8.0）
            LOW:  150 / (100 + 50/499)  = 74850/49950 ≈ 1.4985（旧 1.5）
        いずれも 2倍/5倍/10倍の到達判定（n_2x/n_5x/n_10x）は変えないので
        カウント自体は従来のまま。execution_lag=0 で旧実装と完全一致する
        ことは別途スクリプトで検証済み（PR 本文参照）。
        """
        _, metrics, trades = _run(["MID", "HIGH", "LOW"], max_positions=10)
        self.assertEqual(metrics["num_trades"], 3)
        # MID(~2.99x), HIGH(~7.89x) が 2倍到達。LOW(~1.50x) は未到達。
        self.assertEqual(metrics["n_2x"], 2)
        self.assertEqual(metrics["n_5x"], 1)  # HIGH のみ
        self.assertEqual(metrics["n_10x"], 0)
        mm = dict(zip(trades["symbol"], trades["max_multiple"]))
        self.assertAlmostEqual(mm["MID"], 1497 / 501, places=4)
        self.assertAlmostEqual(mm["HIGH"], 399200 / 50600, places=4)
        self.assertAlmostEqual(mm["LOW"], 74850 / 49950, places=4)

    def test_exit_reasons_recorded(self):
        """撤退理由が記録される（上昇銘柄=end_of_data, 崩落=stop 系）。"""
        _, _, trades = _run(["MID", "HIGH", "CRASH"], max_positions=10)
        reasons = dict(zip(trades["symbol"], trades["exit_reason"]))
        self.assertEqual(reasons["MID"], "end_of_data")
        self.assertEqual(reasons["HIGH"], "end_of_data")
        # CRASH は再投資で複数回リサイクルされ得るが、少なくとも1回は stop 撤退＋実現損
        crash = trades[trades["symbol"] == "CRASH"]
        self.assertTrue((crash["exit_reason"].isin({"ma_break", "trail_stop"})).any())
        self.assertTrue((crash["multiple"] < 1.0).any())

    def test_max_positions_not_exceeded(self):
        """max_positions を超えてエントリーされない。"""
        # 5 銘柄を候補にするが上限 2 → 上位 2（HIGH, MID）のみ保有
        symbols = ["HIGH", "MID", "CRASH"]
        _, metrics, trades = _run(symbols, max_positions=2)
        self.assertLessEqual(metrics["num_trades"], 2)
        # スコア上位 2 銘柄が選ばれる
        self.assertEqual(set(trades["symbol"]), {"HIGH", "MID"})

    def test_no_lookahead_as_of(self):
        """各リスクリーンの as_of がリスクリーン日に一致し、end を超えない。"""
        call_log: list = []
        _run(["MID", "HIGH"], call_log=call_log, max_positions=10)
        self.assertTrue(call_log)
        for as_of in call_log:
            self.assertIsNotNone(as_of)
            self.assertGreaterEqual(as_of, _ENTRY_DATE)
            self.assertLessEqual(as_of, _END)
        # 昇順（過去→未来の順でスクリーンされる）
        self.assertEqual(call_log, sorted(call_log))

    def test_equity_monotonic_and_consistent(self):
        """equity_df は日付昇順で、最終評価額が total_return と整合する。"""
        equity, metrics, _ = _run(["MID", "HIGH"], max_positions=10, initial_cash=1_000_000.0)
        dates = list(equity["date"])
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(len(dates), len(set(dates)))  # 重複なし
        final = float(equity["portfolio_value"].iloc[-1])
        self.assertEqual(metrics["final_cash"], round(final, 2))
        expected = 1_000_000.0 * (1.0 + metrics["total_return"])
        self.assertAlmostEqual(final, expected, delta=1.0)

    def test_alpha_vs_benchmark(self):
        """ベンチマーク超過(alpha)が total_return - benchmark_return で出る。"""
        _, metrics, _ = _run(["MID", "HIGH"], max_positions=10)
        self.assertEqual(metrics["benchmark_return"], 0.5)
        self.assertAlmostEqual(metrics["alpha"], round(metrics["total_return"] - 0.5, 6), places=6)

    def test_empty_universe(self):
        """対象データなしでも例外なく空の結果を返す。"""
        with patch.object(
            lb_prices,
            "load_raw_closes",
            return_value=pd.DataFrame(columns=["symbol", "ts", "close"]),
        ), patch.object(
            lb_engine,
            "fetch_benchmark_returns",
            return_value={"ticker": "^GSPC", "total_return": None},
        ):
            equity, metrics, trades = lb.run_longterm_backtest(
                market="us", start=_ENTRY_DATE, end=_END
            )
        self.assertTrue(equity.empty)
        self.assertTrue(trades.empty)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertEqual(metrics["final_cash"], 1_000_000.0)

    def test_conclusion_contains_survivorship_note(self):
        """結論文に生存者バイアスの注意書きが含まれる。"""
        _, metrics, _ = _run(["MID", "HIGH"], max_positions=10)
        text = lb.build_conclusion(metrics, "us", _ENTRY_DATE, _END, 10)
        self.assertIn("生存者バイアス", text)
        self.assertIn("到達", text)


if __name__ == "__main__":
    unittest.main()


class TestMakeRescreenDates(unittest.TestCase):
    """make_rescreen_dates の二分探索化（PR-3）が旧実装と一致することの回帰テスト。"""

    @staticmethod
    def _reference(calendar, start, freq):
        """bisect 化する前の線形走査による実装（比較用の正解）。"""
        if not calendar:
            return []
        offset = lb_prices.FREQ_OFFSETS.get(freq, lb_prices.FREQ_OFFSETS["quarterly"])
        last = pd.Timestamp(calendar[-1])
        target = pd.Timestamp(max(start, calendar[0]))
        out, seen = [], set()
        while target <= last:
            target_str = target.strftime("%Y-%m-%d")
            nxt = next((d for d in calendar if d >= target_str), None)
            if nxt is not None and nxt not in seen:
                out.append(nxt)
                seen.add(nxt)
            target = target + offset
        return out

    def _assert_matches(self, calendar, start, freq):
        self.assertEqual(
            lb_prices.make_rescreen_dates(calendar, start, freq),
            self._reference(calendar, start, freq),
            f"freq={freq} start={start}",
        )

    def test_matches_reference_for_each_freq(self):
        cal = pd.date_range("2020-01-01", periods=900, freq="B").strftime("%Y-%m-%d").tolist()
        for freq in ("weekly", "monthly", "quarterly", "yearly", "annual"):
            self._assert_matches(cal, cal[0], freq)

    def test_matches_reference_with_start_before_calendar(self):
        cal = pd.date_range("2022-03-01", periods=300, freq="B").strftime("%Y-%m-%d").tolist()
        self._assert_matches(cal, "2019-01-01", "quarterly")

    def test_matches_reference_with_long_gaps(self):
        """長期休場を跨ぐと隣接ターゲットが同じ取引日に丸まる経路を踏ませる。"""
        cal = ["2024-01-02", "2024-01-03", "2024-06-03", "2024-06-04", "2024-12-02"]
        for freq in ("weekly", "monthly"):
            self._assert_matches(cal, "2024-01-02", freq)

    def test_unknown_freq_falls_back_to_quarterly(self):
        cal = pd.date_range("2024-01-01", periods=400, freq="B").strftime("%Y-%m-%d").tolist()
        self.assertEqual(
            lb_prices.make_rescreen_dates(cal, cal[0], "fortnightly"),
            lb_prices.make_rescreen_dates(cal, cal[0], "quarterly"),
        )

    def test_empty_calendar_returns_empty(self):
        self.assertEqual(lb_prices.make_rescreen_dates([], "2024-01-01", "weekly"), [])

    def test_no_duplicate_dates(self):
        cal = ["2024-01-02", "2024-01-03", "2024-06-03", "2024-06-04", "2024-12-02"]
        out = lb_prices.make_rescreen_dates(cal, "2024-01-02", "weekly")
        self.assertEqual(len(out), len(set(out)))
