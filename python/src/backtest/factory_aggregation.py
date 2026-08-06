"""
戦略ファクトリー（#369）: 銘柄別評価結果の集計（#625）

少取引銘柄の Sharpe は分母がほぼ 0 になり発散する（取引が 2 回なら
std = |a-b|/√2 であり、2 回のリターンが近いほど Sharpe が大きくなる）。
そのため銘柄あたり最低取引数を満たす銘柄だけを集計に採用する。

factory.py から切り出した純関数であり DataFrame に依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolMetrics:
    """1銘柄・全期間シミュレーションの評価結果。"""

    symbol: str
    num_trades: int
    sharpe_ratio: float
    sharpe_per_trade: float
    win_rate: float
    total_return: float
    max_drawdown: float


@dataclass
class AggregatedMetrics:
    """有効銘柄のみで再集計した仮説単位のメトリクス。

    n_symbols_with_signal / avg_trades_per_symbol はフィルタ「前」の母数で算出する。
    フィルタがどれだけ効いたかを診断するための値であるため。
    """

    sharpe_ratio: float = 0.0
    sharpe_per_trade: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    n_symbols_with_signal: int = 0
    n_effective_symbols: int = 0
    avg_trades_per_symbol: float = 0.0


def aggregate_symbol_metrics(
    rows: list[SymbolMetrics], min_trades_per_symbol: int
) -> AggregatedMetrics:
    """銘柄別の評価結果を、最低取引数を満たす銘柄だけで集計する。

    Args:
        rows: 買いシグナルが出た銘柄の評価結果（シグナル 0 の銘柄は含めない）
        min_trades_per_symbol: 集計に採用する銘柄あたり最低取引数

    Returns:
        AggregatedMetrics。有効銘柄が 0 件でも例外を投げず、集計値は 0 のまま
        診断用の n_symbols_with_signal / avg_trades_per_symbol だけを埋めて返す。
    """
    n_with_signal = len(rows)
    avg_trades = sum(r.num_trades for r in rows) / n_with_signal if n_with_signal else 0.0

    effective = [r for r in rows if r.num_trades >= min_trades_per_symbol]
    if not effective:
        return AggregatedMetrics(
            n_symbols_with_signal=n_with_signal,
            n_effective_symbols=0,
            avg_trades_per_symbol=avg_trades,
        )

    n = len(effective)
    return AggregatedMetrics(
        sharpe_ratio=sum(r.sharpe_ratio for r in effective) / n,
        sharpe_per_trade=sum(r.sharpe_per_trade for r in effective) / n,
        win_rate=sum(r.win_rate for r in effective) / n,
        total_return=sum(r.total_return for r in effective) / n,
        max_drawdown=min(r.max_drawdown for r in effective),
        num_trades=sum(r.num_trades for r in effective),
        n_symbols_with_signal=n_with_signal,
        n_effective_symbols=n,
        avg_trades_per_symbol=avg_trades,
    )
