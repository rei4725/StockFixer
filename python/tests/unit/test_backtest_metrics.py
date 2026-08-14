"""backtest/metrics.py のユニットテスト"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.metrics import (
    _extract_trade_pnl,
    _max_drawdown,
    _sharpe_per_trade,
    compute_metrics,
    deflated_sharpe_ratio,
    fetch_benchmark_returns,
    monte_carlo_equity,
    plot_backtest,
)


def _trade_log(entries: list[dict]) -> pd.DataFrame:
    """テスト用の取引ログ DataFrame を生成する"""
    return pd.DataFrame(entries)


class TestComputeMetricsEmpty(unittest.TestCase):
    """取引なしのケース"""

    def test_empty_dataframe_returns_zero_metrics(self):
        metrics = compute_metrics(pd.DataFrame(), initial_cash=1_000_000)
        self.assertEqual(metrics["total_return"], 0.0)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["max_drawdown"], 0.0)
        self.assertIsNone(metrics["profit_factor"])

    def test_none_returns_empty_metrics(self):
        metrics = compute_metrics(None, initial_cash=500_000)
        self.assertEqual(metrics["final_cash"], 500_000)
        self.assertEqual(metrics["num_trades"], 0)


class TestComputeMetricsProfitable(unittest.TestCase):
    """利益トレードのケース"""

    def setUp(self):
        # 100株 @ 100円で買い、110円で売り → 利益 1000円
        self.initial_cash = 1_000_000
        self.log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 110.0,
                    "qty": 100,
                    "cash": 1_001_000,
                },
            ]
        )

    def test_num_trades(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertEqual(metrics["num_trades"], 1)

    def test_win_rate_one(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertAlmostEqual(metrics["win_rate"], 1.0)

    def test_profit_factor_infinite_or_positive(self):
        """勝ちトレードのみ → profit_factor は None(inf) または正の値"""
        metrics = compute_metrics(self.log, self.initial_cash)
        # 損失ゼロなので inf → None に変換
        self.assertIsNone(metrics["profit_factor"])

    def test_total_return_positive(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertGreater(metrics["total_return"], 0.0)


class TestComputeMetricsLoss(unittest.TestCase):
    """損失トレードのケース"""

    def setUp(self):
        self.initial_cash = 1_000_000
        self.log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 90.0,
                    "qty": 100,
                    "cash": 999_000,
                },
            ]
        )

    def test_win_rate_zero(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertAlmostEqual(metrics["win_rate"], 0.0)

    def test_num_trades_one(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertEqual(metrics["num_trades"], 1)


class TestComputeMetricsMixed(unittest.TestCase):
    """勝ち負け混在のケース"""

    def setUp(self):
        self.initial_cash = 1_000_000
        # 1回目: 100→120 (+2000), 2回目: 100→80 (-2000)
        self.log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-05",
                    "action": "sell",
                    "price": 120.0,
                    "qty": 100,
                    "cash": 1_002_000,
                },
                {
                    "date": "2024-01-07",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 992_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 80.0,
                    "qty": 100,
                    "cash": 1_000_000,
                },
            ]
        )

    def test_num_trades_two(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertEqual(metrics["num_trades"], 2)

    def test_win_rate_half(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_profit_factor_positive(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertIsNotNone(metrics["profit_factor"])
        self.assertGreater(metrics["profit_factor"], 0)

    def test_sharpe_ratio_is_float(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertIsInstance(metrics["sharpe_ratio"], float)


class TestExtractTradePnl(unittest.TestCase):
    """_extract_trade_pnl のテスト"""

    def test_single_win(self):
        log = _trade_log(
            [
                {"action": "buy", "price": 100.0, "qty": 10, "cash": 900},
                {"action": "sell", "price": 110.0, "qty": 10, "cash": 1000},
            ]
        )
        wins, losses = _extract_trade_pnl(log)[:2]
        self.assertEqual(len(wins), 1)
        self.assertEqual(len(losses), 0)
        self.assertAlmostEqual(wins[0], 100.0)

    def test_single_loss(self):
        log = _trade_log(
            [
                {"action": "buy", "price": 100.0, "qty": 10, "cash": 900},
                {"action": "sell", "price": 90.0, "qty": 10, "cash": 800},
            ]
        )
        wins, losses = _extract_trade_pnl(log)[:2]
        self.assertEqual(len(wins), 0)
        self.assertEqual(len(losses), 1)
        self.assertAlmostEqual(losses[0], -100.0)

    def test_no_trades(self):
        wins, losses = _extract_trade_pnl(pd.DataFrame())[:2]
        self.assertEqual(wins, [])
        self.assertEqual(losses, [])

    def test_final_sell_counted(self):
        log = _trade_log(
            [
                {"action": "buy", "price": 100.0, "qty": 5, "cash": 950},
                {"action": "final_sell", "price": 120.0, "qty": 5, "cash": 1050},
            ]
        )
        wins, losses = _extract_trade_pnl(log)[:2]
        self.assertEqual(len(wins), 1)


class TestSharpeRatio(unittest.TestCase):
    """_sharpe_per_trade のテスト（非年率の取引単位 Sharpe）"""

    def test_empty_list_returns_zero(self):
        self.assertEqual(_sharpe_per_trade([]), 0.0)

    def test_single_item_returns_zero(self):
        self.assertEqual(_sharpe_per_trade([100.0]), 0.0)

    def test_zero_std_returns_zero(self):
        self.assertEqual(_sharpe_per_trade([100.0, 100.0, 100.0]), 0.0)

    def test_positive_returns_positive_sharpe(self):
        pnl = [200.0, 100.0, 300.0, 150.0, 250.0]
        sharpe = _sharpe_per_trade(pnl)
        self.assertGreater(sharpe, 0.0)

    def test_negative_returns_negative_sharpe(self):
        pnl = [-200.0, -100.0, -300.0, -150.0, -250.0]
        sharpe = _sharpe_per_trade(pnl)
        self.assertLess(sharpe, 0.0)


class TestMaxDrawdown(unittest.TestCase):
    """_max_drawdown のテスト"""

    def test_monotone_increase_no_drawdown(self):
        equity = pd.Series([1000, 1100, 1200, 1300])
        dd = _max_drawdown(equity)
        self.assertAlmostEqual(dd, 0.0)


class TestPlotBacktestAndBenchmark(unittest.TestCase):
    """plot_backtest / fetch_benchmark_returns のテスト"""

    _MPL_MODULES = ("matplotlib", "matplotlib.pyplot", "matplotlib.dates")

    def setUp(self):
        import sys

        import yfinance as _real_yf

        self._real_yf = _real_yf
        # 偽 matplotlib を sys.modules に挿す前に実物を退避する
        # （復元しないと後続テストの実 matplotlib 利用が壊れる）
        self._real_mpl = {name: sys.modules.get(name) for name in self._MPL_MODULES}

    def tearDown(self):
        import sys

        sys.modules["yfinance"] = self._real_yf
        for name, module in self._real_mpl.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)

    def _install_fake_matplotlib(self):
        class _Axis:
            def __init__(self):
                self.xaxis = types.SimpleNamespace(
                    set_major_formatter=lambda *_a, **_k: None,
                    get_majorticklabels=lambda: [],
                )

            def plot(self, *_a, **_k):
                return None

            def axhline(self, *_a, **_k):
                return None

            def set_ylabel(self, *_a, **_k):
                return None

            def set_xlabel(self, *_a, **_k):
                return None

            def legend(self, *_a, **_k):
                return None

            def grid(self, *_a, **_k):
                return None

            def fill_between(self, *_a, **_k):
                return None

        class _Fig:
            def suptitle(self, *_a, **_k):
                return None

            def text(self, *_a, **_k):
                return None

        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda *a, **k: (_Fig(), (_Axis(), _Axis()))
        fake_pyplot.setp = lambda *_a, **_k: None
        fake_pyplot.tight_layout = lambda *_a, **_k: None
        fake_pyplot.savefig = lambda path, **_k: Path(path).write_bytes(b"png")
        fake_pyplot.close = lambda *_a, **_k: None

        fake_mdates = types.ModuleType("matplotlib.dates")
        fake_mdates.DateFormatter = lambda fmt: fmt

        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.__path__ = []
        fake_matplotlib.use = lambda *_a, **_k: None
        fake_matplotlib.pyplot = fake_pyplot
        fake_matplotlib.dates = fake_mdates

        sys.modules["matplotlib"] = fake_matplotlib
        sys.modules["matplotlib.pyplot"] = fake_pyplot
        sys.modules["matplotlib.dates"] = fake_mdates

    def test_plot_backtest_saves_png(self):
        self._install_fake_matplotlib()
        trade_log = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-10"]),
                "action": ["buy", "sell"],
                "price": [100.0, 110.0],
                "qty": [10, 10],
                "cash": [999_000, 1_001_000],
            }
        )
        metrics = {
            "total_return": 0.01,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.03,
            "num_trades": 1,
            "win_rate": 1.0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = plot_backtest(trade_log, metrics, str(tmp_dir), market="jp", symbol="7203")
            self.assertTrue(out.endswith(".png"))
            self.assertTrue(Path(out).exists())

    def test_fetch_benchmark_returns_success(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=idx)
        fake_yf = types.SimpleNamespace(download=lambda *_a, **_k: df)
        sys.modules["yfinance"] = fake_yf

        result = fetch_benchmark_returns("^N225", "2024-01-01", "2024-01-10")
        self.assertEqual(result["ticker"], "^N225")
        self.assertIsNotNone(result["total_return"])

    def test_fetch_benchmark_returns_failure(self):
        class _ErrYf:
            @staticmethod
            def download(*_a, **_k):
                raise RuntimeError("network")

        sys.modules["yfinance"] = _ErrYf
        result = fetch_benchmark_returns("^N225", "2024-01-01", "2024-01-10")
        self.assertIsNone(result["total_return"])

    def test_monotone_decrease_full_drawdown(self):
        equity = pd.Series([1000, 900, 800, 700])
        dd = _max_drawdown(equity)
        self.assertLess(dd, 0.0)

    def test_drawdown_peak_then_recover(self):
        # 1000→1200→900→1100: max DD = (900-1200)/1200 = -0.25
        equity = pd.Series([1000.0, 1200.0, 900.0, 1100.0])
        dd = _max_drawdown(equity)
        self.assertAlmostEqual(dd, (900 - 1200) / 1200, places=8)

    def test_empty_series_returns_zero(self):
        dd = _max_drawdown(pd.Series([], dtype=float))
        self.assertEqual(dd, 0.0)


# ===========================================================================
# 境界値テスト（pytest parametrize スタイル）
# ===========================================================================


class TestComputeMetricsBoundary:
    """境界値・パラメータ化テスト"""

    @pytest.mark.parametrize(
        "buy_price, sell_price, expected_win_rate",
        [
            (100, 110, 1.0),  # 利益トレード → 勝率 1.0
            (100, 90, 0.0),  # 損失トレード → 勝率 0.0
        ],
    )
    def test_win_rate_single_trade(self, buy_price, sell_price, expected_win_rate):
        """単一取引の勝率は利益/損失に応じて 1.0 / 0.0"""
        initial_cash = 1_000_000
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": float(buy_price),
                    "qty": 100,
                    "cash": initial_cash - buy_price * 100,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": float(sell_price),
                    "qty": 100,
                    "cash": initial_cash + (sell_price - buy_price) * 100,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash)
        assert metrics["win_rate"] == pytest.approx(expected_win_rate)

    @pytest.mark.parametrize(
        "pnl, expected_sign",
        [
            ([200.0, 100.0, 150.0], "positive"),
            ([-200.0, -100.0, -150.0], "negative"),
        ],
    )
    def test_sharpe_sign(self, pnl, expected_sign):
        """平均プラスの PnL → シャープ正 / マイナスの PnL → シャープ負"""
        sharpe = _sharpe_per_trade(pnl)
        if expected_sign == "positive":
            assert sharpe > 0.0
        else:
            assert sharpe < 0.0

    @pytest.mark.parametrize("initial_cash", [100_000, 500_000, 1_000_000, 5_000_000])
    def test_empty_metrics_respects_initial_cash(self, initial_cash):
        """取引なし時の final_cash が initial_cash に一致する"""
        metrics = compute_metrics(pd.DataFrame(), initial_cash=initial_cash)
        assert metrics["final_cash"] == initial_cash
        assert metrics["total_return"] == 0.0

    @pytest.mark.parametrize(
        "equity, expected_dd",
        [
            ([1000, 1100, 1200, 1300], 0.0),  # 単調増加 → DDなし
            (
                [1000, 1200, 900, 1100],
                (900 - 1200) / 1200,
            ),  # ピークから1200→9003々0  # ピークから 25% 褶辺
        ],
    )
    def test_max_drawdown_scenarios(self, equity, expected_dd):
        """max_drawdown の各シナリオ検証"""
        dd = _max_drawdown(pd.Series(equity, dtype=float))
        assert dd == pytest.approx(expected_dd, abs=1e-8)


class TestComputeMetricsAvgWinLoss:
    """avg_win / avg_loss の数値精度テスト"""

    def test_single_win_avg_win(self):
        """100→110 の勝ちトレード1件: avg_win == 0.1"""
        initial_cash = 1_000_000
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 110.0,
                    "qty": 100,
                    "cash": 1_001_000,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash)
        assert metrics["avg_win"] == pytest.approx(0.1)

    def test_single_loss_avg_loss(self):
        """100→95 の負けトレード1件: avg_loss == 0.05"""
        initial_cash = 1_000_000
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 95.0,
                    "qty": 100,
                    "cash": 999_500,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash)
        assert metrics["avg_loss"] == pytest.approx(0.05)


class TestComputeMetricsTradeReturns:
    """trade_returns（符号付き取引リターン率）のテスト（#630: DSRを複数銘柄で
    プールする際、価格スケールに依存しない%リターンを使うための出力）。
    """

    def test_returns_signed_percentage_returns(self):
        initial_cash = 1_000_000
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 110.0,
                    "qty": 100,
                    "cash": 1_001_000,
                },
                {
                    "date": "2024-01-12",
                    "action": "buy",
                    "price": 200.0,
                    "qty": 50,
                    "cash": 991_000,
                },
                {
                    "date": "2024-01-20",
                    "action": "sell",
                    "price": 190.0,
                    "qty": 50,
                    "cash": 995_500,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash)
        assert sorted(metrics["trade_returns"]) == pytest.approx(sorted([0.10, -0.05]))

    def test_empty_when_no_trades(self):
        metrics = compute_metrics(pd.DataFrame(), initial_cash=1_000_000)
        assert metrics["trade_returns"] == []

    def test_all_wins_avg_loss_zero(self):
        """全トレードが勝ちの場合、avg_loss == 0.0"""
        initial_cash = 1_000_000
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 110.0,
                    "qty": 100,
                    "cash": 1_001_000,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash)
        assert metrics["avg_loss"] == 0.0

    def test_all_losses_avg_win_zero(self):
        """全トレードが負けの場合、avg_win == 0.0"""
        initial_cash = 1_000_000
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 90.0,
                    "qty": 100,
                    "cash": 999_000,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash)
        assert metrics["avg_win"] == 0.0

    def test_no_trades_avg_win_avg_loss_zero(self):
        """トレードなし: avg_win == 0.0、avg_loss == 0.0"""
        metrics = compute_metrics(pd.DataFrame(), initial_cash=1_000_000)
        assert metrics["avg_win"] == 0.0
        assert metrics["avg_loss"] == 0.0


class TestMonteCarloEquity(unittest.TestCase):
    """monte_carlo_equity のユニットテスト"""

    def test_empty_list_returns_all_zeros(self):
        """空リストを渡しても例外が出ず、全フィールドが 0.0 であること"""
        result = monte_carlo_equity([], initial_cash=1_000_000)
        for key in (
            "max_drawdown_mean",
            "max_drawdown_p95",
            "final_cash_p05",
            "final_cash_p50",
            "final_cash_p95",
        ):
            self.assertEqual(result[key], 0.0)

    def test_positive_pnl_list_final_cash_p05_lt_p95(self):
        """[100.0, -50.0, 200.0] を渡したとき final_cash_p05 < final_cash_p95 であること"""
        result = monte_carlo_equity([100.0, -50.0, 200.0], initial_cash=1_000_000)
        self.assertLess(result["final_cash_p05"], result["final_cash_p95"])

    def test_reproducibility_with_same_seed(self):
        """seed=42 で 2 回呼んだとき同じ結果が返ること"""
        pnl = [100.0, -50.0, 200.0, -30.0, 80.0]
        result1 = monte_carlo_equity(pnl, initial_cash=1_000_000, seed=42)
        result2 = monte_carlo_equity(pnl, initial_cash=1_000_000, seed=42)
        self.assertEqual(result1, result2)


class TestComputeMetricsMonteCarlo(unittest.TestCase):
    """compute_metrics の include_monte_carlo オプションのテスト（#374）"""

    def setUp(self):
        self.initial_cash = 1_000_000
        self.log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-05",
                    "action": "sell",
                    "price": 120.0,
                    "qty": 100,
                    "cash": 1_002_000,
                },
                {
                    "date": "2024-01-07",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 992_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 80.0,
                    "qty": 100,
                    "cash": 1_000_000,
                },
            ]
        )

    def test_default_omits_mc_keys(self):
        """include_monte_carlo 未指定（デフォルト False）では mc_ キーが含まれない"""
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertNotIn("mc_final_cash_p50", metrics)

    def test_include_monte_carlo_adds_mc_keys(self):
        """include_monte_carlo=True で mc_ プレフィックスの統計が追加される"""
        metrics = compute_metrics(self.log, self.initial_cash, include_monte_carlo=True)
        for key in (
            "mc_max_drawdown_mean",
            "mc_max_drawdown_p95",
            "mc_final_cash_p05",
            "mc_final_cash_p50",
            "mc_final_cash_p95",
        ):
            self.assertIn(key, metrics)

    def test_include_monte_carlo_with_no_trades_returns_zeros(self):
        """取引なしで include_monte_carlo=True を指定しても例外にならない"""
        metrics = compute_metrics(pd.DataFrame(), self.initial_cash, include_monte_carlo=True)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertNotIn("mc_final_cash_p50", metrics)


class TestDeflatedSharpeRatio(unittest.TestCase):
    """deflated_sharpe_ratio と compute_metrics の n_trials 拡張テスト"""

    def test_n_trials_1_returns_approx_half(self):
        """n_trials=1, sharpe=0.0 のとき DSR ≈ norm.cdf(0) ≈ 0.5"""
        dsr = deflated_sharpe_ratio(sharpe=0.0, n_trials=1, n_obs=50)
        self.assertAlmostEqual(dsr, 0.5, places=6)

    def test_range_0_to_1(self):
        """sharpe=3.0, n_trials=100, n_obs=50 のとき DSR が [0, 1] の範囲内"""
        dsr = deflated_sharpe_ratio(sharpe=3.0, n_trials=100, n_obs=50)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_compute_metrics_no_dsr_when_n_trials_zero(self):
        """n_trials=0 を渡したとき compute_metrics 結果に 'dsr' キーが含まれない"""
        log = _trade_log(
            [
                {
                    "date": "2024-01-02",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 100,
                    "cash": 990_000,
                },
                {
                    "date": "2024-01-10",
                    "action": "sell",
                    "price": 110.0,
                    "qty": 100,
                    "cash": 1_001_000,
                },
            ]
        )
        metrics = compute_metrics(log, initial_cash=1_000_000, n_trials=0)
        self.assertNotIn("dsr", metrics)


if __name__ == "__main__":
    unittest.main()
