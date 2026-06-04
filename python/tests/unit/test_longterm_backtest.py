"""長期コホート・バックテスト(#431)の単体テスト。

合成の stock_features（既知の価格軌道）を load_stock_features / get_all_symbols
モックで与え、fetch_benchmark_returns もモックして検証する。
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest import longterm_backtest as lb
from src.screening.types import TrendCandidate

_PHASE_A = 400  # トレンド形成期間（営業日）
_PHASE_B = 500  # ホールド期間（営業日）
_N = _PHASE_A + _PHASE_B
_DATES = pd.date_range("2020-01-01", periods=_N, freq="B").strftime("%Y-%m-%d").tolist()
_ENTRY_DATE = _DATES[_PHASE_A]  # 最初のリスクリーン日＝Phase B 開始日
_END = _DATES[-1]


def _features(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": _DATES,
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(_N, 1_000_000.0),
        }
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


def _loader(market, symbol):
    return _DF_MAP.get(symbol)


def _run(symbols, screen=None, call_log=None, **kwargs):
    syms = [("us", s) for s in symbols]
    screen = screen or _make_screen(symbols, call_log)
    with patch.object(lb, "get_all_symbols", return_value=syms), patch.object(
        lb, "load_stock_features", side_effect=_loader
    ), patch.object(lb, "screen_trend_candidates", side_effect=screen), patch.object(
        lb,
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
        """既知の 1.5倍・3倍・8倍軌道（全て満期保有）で到達倍率カウントが正しい。"""
        _, metrics, trades = _run(["MID", "HIGH", "LOW"], max_positions=10)
        self.assertEqual(metrics["n_trades"], 3)
        # MID(3x), HIGH(8x) が 2倍到達。LOW(1.5x) は未到達。
        self.assertEqual(metrics["n_2x"], 2)
        self.assertEqual(metrics["n_5x"], 1)  # HIGH のみ
        self.assertEqual(metrics["n_10x"], 0)
        mm = dict(zip(trades["symbol"], trades["max_multiple"]))
        self.assertAlmostEqual(mm["MID"], 3.0, places=2)
        self.assertAlmostEqual(mm["HIGH"], 8.0, places=2)
        self.assertAlmostEqual(mm["LOW"], 1.5, places=2)

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
        self.assertLessEqual(metrics["n_trades"], 2)
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
        with patch.object(lb, "get_all_symbols", return_value=[]), patch.object(
            lb, "load_stock_features", return_value=None
        ), patch.object(
            lb,
            "fetch_benchmark_returns",
            return_value={"ticker": "^GSPC", "total_return": None},
        ):
            equity, metrics, trades = lb.run_longterm_backtest(
                market="us", start=_ENTRY_DATE, end=_END
            )
        self.assertTrue(equity.empty)
        self.assertTrue(trades.empty)
        self.assertEqual(metrics["n_trades"], 0)
        self.assertEqual(metrics["final_cash"], 1_000_000.0)

    def test_conclusion_contains_survivorship_note(self):
        """結論文に生存者バイアスの注意書きが含まれる。"""
        _, metrics, _ = _run(["MID", "HIGH"], max_positions=10)
        text = lb.build_conclusion(metrics, "us", _ENTRY_DATE, _END, 10)
        self.assertIn("生存者バイアス", text)
        self.assertIn("到達", text)


if __name__ == "__main__":
    unittest.main()
