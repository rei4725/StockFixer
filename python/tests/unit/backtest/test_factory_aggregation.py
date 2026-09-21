"""factory_aggregation の集計ロジックのテスト（#625）。"""

from __future__ import annotations

import math
import unittest

from src.backtest.factory_aggregation import SymbolMetrics, aggregate_symbol_metrics
from src.backtest.metrics import _annualize_sharpe, _sharpe_per_trade


def _row(symbol: str, num_trades: int, sharpe: float, **kwargs) -> SymbolMetrics:
    defaults = dict(
        sharpe_per_trade=sharpe / 10,
        win_rate=0.6,
        total_return=0.1,
        max_drawdown=-0.05,
    )
    defaults.update(kwargs)
    return SymbolMetrics(
        symbol=symbol,
        num_trades=num_trades,
        sharpe_ratio=sharpe,
        **defaults,
    )


class TestAggregateSymbolMetrics(unittest.TestCase):
    def test_excludes_symbols_below_min_trades(self):
        # 2取引の発散 Sharpe(25.0) は除外され、5取引の 0.5 だけが残る
        rows = [_row("AAA", 2, 25.0), _row("BBB", 5, 0.5)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.n_effective_symbols, 1)
        self.assertAlmostEqual(result.sharpe_ratio, 0.5)

    def test_num_trades_counts_only_effective_symbols(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0), _row("CCC", 8, 0.4)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.num_trades, 8)

    def test_reports_signal_and_effective_symbol_counts(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0), _row("CCC", 8, 0.4)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.n_symbols_with_signal, 3)
        self.assertEqual(result.n_effective_symbols, 1)
        # 平均取引数はフィルタ前の母数で割る: (1 + 2 + 8) / 3
        self.assertAlmostEqual(result.avg_trades_per_symbol, 11 / 3)

    def test_healthy_distribution_is_unchanged_by_filter(self):
        # 全銘柄が閾値以上なら、フィルタの有無で結果が一致する（回帰テスト）
        rows = [_row("AAA", 5, 0.8), _row("BBB", 9, 0.4), _row("CCC", 12, 0.6)]

        # NaN 同士は等価比較できないため、span_years を与えて全フィールドを実数にする
        filtered = aggregate_symbol_metrics(rows, min_trades_per_symbol=3, span_years=2.0)
        unfiltered = aggregate_symbol_metrics(rows, min_trades_per_symbol=1, span_years=2.0)

        self.assertEqual(filtered, unfiltered)
        self.assertAlmostEqual(filtered.sharpe_ratio, 0.6)
        self.assertEqual(filtered.n_effective_symbols, 3)

    def test_no_effective_symbols_returns_zeros_without_error(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.n_effective_symbols, 0)
        self.assertEqual(result.sharpe_ratio, 0.0)
        self.assertEqual(result.num_trades, 0)
        self.assertEqual(result.max_drawdown, 0.0)
        # 診断用の値はフィルタ前の実態を残す
        self.assertEqual(result.n_symbols_with_signal, 2)
        self.assertAlmostEqual(result.avg_trades_per_symbol, 1.5)

    def test_empty_rows_returns_zeros(self):
        result = aggregate_symbol_metrics([], min_trades_per_symbol=3)

        self.assertEqual(result.n_symbols_with_signal, 0)
        self.assertEqual(result.avg_trades_per_symbol, 0.0)
        self.assertEqual(result.sharpe_ratio, 0.0)

    def test_sharpe_per_trade_pools_trade_returns_across_effective_symbols(self):
        """DSR入力のsharpe_per_tradeは、有効銘柄の取引リターンを1系列にプールして
        算出する。銘柄別sharpe_per_tradeの単純平均ではない（#630）。
        """
        rows = [
            _row("AAA", 5, 0.8, trade_returns=[0.02, -0.01, 0.03, 0.01, -0.02]),
            _row("BBB", 4, 0.6, trade_returns=[0.01, 0.02, -0.01, 0.04]),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        pooled = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.04]
        self.assertAlmostEqual(result.sharpe_per_trade, _sharpe_per_trade(pooled))
        # 単純平均（旧実装）とは一致しないことを確認する
        naive_average = (
            _sharpe_per_trade([0.02, -0.01, 0.03, 0.01, -0.02])
            + _sharpe_per_trade([0.01, 0.02, -0.01, 0.04])
        ) / 2
        self.assertNotAlmostEqual(result.sharpe_per_trade, naive_average)

    def test_sharpe_per_trade_pool_excludes_ineffective_symbols(self):
        """最低取引数を満たさない銘柄の取引リターンはプールに含めない。"""
        rows = [
            _row("AAA", 2, 25.0, trade_returns=[0.50, -0.40]),  # 除外対象
            _row("BBB", 5, 0.5, trade_returns=[0.01, 0.02, -0.01, 0.01, 0.00]),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        expected = _sharpe_per_trade([0.01, 0.02, -0.01, 0.01, 0.00])
        self.assertAlmostEqual(result.sharpe_per_trade, expected)

    def test_max_drawdown_is_worst_of_effective_symbols(self):
        # 除外される銘柄(-0.90)の DD は採用されない
        rows = [
            _row("AAA", 2, 25.0, max_drawdown=-0.90),
            _row("BBB", 5, 0.5, max_drawdown=-0.08),
            _row("CCC", 6, 0.5, max_drawdown=-0.20),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertAlmostEqual(result.max_drawdown, -0.20)

    def test_effective_symbols_lists_adopted_symbols_in_input_order(self):
        """ポートフォリオDDの合成対象を呼び出し側が特定できるよう銘柄名を返す。

        集計に採用した銘柄と、ポートフォリオ equity を合成する銘柄が一致していないと
        Sharpe と DD が別母集団の数字になってしまう。
        """
        rows = [
            _row("AAA", 2, 25.0),  # 除外
            _row("BBB", 5, 0.5),
            _row("CCC", 6, 0.5),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.effective_symbols, ["BBB", "CCC"])


class TestPortfolioSharpe(unittest.TestCase):
    """ゲート用 Sharpe はプール済み per-trade Sharpe を1回だけ年率化したもの。

    従来の sharpe_ratio は「銘柄別・年率化 Sharpe の単純平均」で、銘柄あたり
    3取引で採用される（MIN_TRADES_PER_SYMBOL=3）ため発散した値が平均に混ざる。
    台帳の再現ペア130組では sharpe_ratio の自己相関が 0.446 しかなく、
    取引数(0.994)・DD(0.958)・リターン(0.898)に比べて著しく不安定だった。
    """

    def test_annualizes_pooled_per_trade_sharpe_with_portfolio_frequency(self):
        rows = [
            _row("AAA", 5, 0.8, trade_returns=[0.02, -0.01, 0.03, 0.01, -0.02]),
            _row("BBB", 4, 0.6, trade_returns=[0.01, 0.02, -0.01, 0.04]),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3, span_years=2.0)

        pooled = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.04]
        expected = _annualize_sharpe(_sharpe_per_trade(pooled), 9 / 2.0)
        self.assertAlmostEqual(result.portfolio_sharpe_ratio, expected)

    def test_is_not_distorted_by_a_diverging_few_trade_symbol(self):
        """発散銘柄を混ぜても、プール後の Sharpe は穏当な水準にとどまる。

        従来の単純平均では除外閾値(3取引)をわずかに超えた銘柄の発散 Sharpe が
        そのまま平均に効いてしまう。
        """
        calm = [0.01, 0.012, 0.009, 0.011, 0.010]
        rows = [
            # 3取引でほぼ同値 → 銘柄別 Sharpe は発散するが、プールでは3標本の寄与しかない
            _row("WILD", 3, 200.0, trade_returns=[0.010, 0.0101, 0.0099]),
            _row("CALM", 5, 0.5, trade_returns=calm),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3, span_years=2.0)

        self.assertAlmostEqual(result.sharpe_ratio, (200.0 + 0.5) / 2)  # 旧指標は発散したまま
        self.assertLess(result.portfolio_sharpe_ratio, 40.0)

    def test_is_nan_when_span_is_unknown(self):
        rows = [_row("AAA", 5, 0.8, trade_returns=[0.02, -0.01, 0.03, 0.01, -0.02])]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertTrue(math.isnan(result.portfolio_sharpe_ratio))

    def test_is_nan_when_no_effective_symbols(self):
        result = aggregate_symbol_metrics(
            [_row("AAA", 1, 0.0)], min_trades_per_symbol=3, span_years=2.0
        )

        self.assertTrue(math.isnan(result.portfolio_sharpe_ratio))


class TestAggregateSymbolMetricsExtra(unittest.TestCase):
    def test_effective_symbols_is_empty_when_no_symbol_qualifies(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.effective_symbols, [])


if __name__ == "__main__":
    unittest.main()
