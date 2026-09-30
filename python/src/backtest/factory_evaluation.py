"""戦略ファクトリー: 銘柄別シミュレーション結果から仮説単位の FactoryEvaluation を組み立てる。

evaluate_hypothesis（本番の夜間バッチ・サンドボックス）と、合成戦略によるゲート分類テスト
（tests/unit/backtest/test_factory_gate_classification.py）が同じ組み立てを通るように
切り出したもの。テスト側に写しを持つと本番とずれていくため、ここを唯一の実装とする。
"""

from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from src.backtest.factory_aggregation import SymbolMetrics, aggregate_symbol_metrics
from src.backtest.factory_portfolio import build_portfolio_equity
from src.backtest.factory_significance import annualized_sharpe, daily_returns
from src.backtest.metrics import _max_drawdown
from src.backtest.types import FactoryEvaluation, FactoryHypothesis


def build_evaluation(
    hypothesis: FactoryHypothesis,
    symbol_rows: list[SymbolMetrics],
    equity_by_symbol: dict[str, pd.Series],
    *,
    initial_cash: float,
    min_trades_per_symbol: int,
    n_symbols: int,
    window_returns: Sequence[float] = (),
) -> FactoryEvaluation:
    """銘柄別の結果を集計し、ゲートが使うポートフォリオ指標を付けて返す。

    - 銘柄別メトリクス: 銘柄あたり最低取引数を満たす銘柄のみで集計する（#625）
    - ポートフォリオ指標: 同じ有効銘柄を等金額で保有した日次 equity から、年率 Sharpe・
      最大ドローダウン・日次リターンを算出する。equity が得られなければ NaN / None
    """
    aggregated = aggregate_symbol_metrics(symbol_rows, min_trades_per_symbol)
    equity = build_portfolio_equity(
        [equity_by_symbol[s] for s in aggregated.effective_symbols if s in equity_by_symbol],
        initial_cash,
    )
    if equity.empty:
        returns = None
        portfolio_sharpe = math.nan
        portfolio_dd = math.nan
    else:
        returns = daily_returns(equity)
        portfolio_sharpe = annualized_sharpe(returns)
        portfolio_dd = _max_drawdown(equity)

    return FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=aggregated.sharpe_ratio,
        sharpe_per_trade=aggregated.sharpe_per_trade,
        portfolio_sharpe_ratio=portfolio_sharpe,
        win_rate=aggregated.win_rate,
        num_trades=aggregated.num_trades,
        max_drawdown=aggregated.max_drawdown,
        portfolio_max_drawdown=portfolio_dd,
        total_return=aggregated.total_return,
        window_returns=list(window_returns),
        n_symbols=n_symbols,
        n_symbols_with_signal=aggregated.n_symbols_with_signal,
        n_effective_symbols=aggregated.n_effective_symbols,
        avg_trades_per_symbol=aggregated.avg_trades_per_symbol,
        portfolio_returns=returns,
    )
