"""
戦略ファクトリー（#369）: 銘柄別評価結果の集計（#625）

少取引銘柄の Sharpe は分母がほぼ 0 になり発散する（取引が 2 回なら
std = |a-b|/√2 であり、2 回のリターンが近いほど Sharpe が大きくなる）。
そのため銘柄あたり最低取引数を満たす銘柄だけを集計に採用する。

factory.py から切り出した純関数であり DataFrame に依存しない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.backtest.metrics import _annualize_sharpe, _sharpe_per_trade


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
    # 符号付き取引リターン率（%）。DSR算出のため複数銘柄でプールする（#630）。
    trade_returns: list[float] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """有効銘柄のみで再集計した仮説単位のメトリクス。

    n_symbols_with_signal / avg_trades_per_symbol はフィルタ「前」の母数で算出する。
    フィルタがどれだけ効いたかを診断するための値であるため。
    """

    # 銘柄別・年率化 Sharpe の単純平均（診断用。ゲート判定には使わない）
    sharpe_ratio: float = 0.0
    sharpe_per_trade: float = 0.0
    # プール済み per-trade Sharpe をポートフォリオの取引頻度で1回だけ年率化した値
    # （ゲート判定に使う。評価期間が不明・有効銘柄0なら NaN）
    portfolio_sharpe_ratio: float = float("nan")
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    n_symbols_with_signal: int = 0
    n_effective_symbols: int = 0
    avg_trades_per_symbol: float = 0.0
    # 集計に採用した銘柄名。ポートフォリオ equity を同じ母集団で合成するために返す。
    effective_symbols: list[str] = field(default_factory=list)


def aggregate_symbol_metrics(
    rows: list[SymbolMetrics], min_trades_per_symbol: int, span_years: float = 0.0
) -> AggregatedMetrics:
    """銘柄別の評価結果を、最低取引数を満たす銘柄だけで集計する。

    Args:
        rows: 買いシグナルが出た銘柄の評価結果（シグナル 0 の銘柄は含めない）
        min_trades_per_symbol: 集計に採用する銘柄あたり最低取引数
        span_years: 評価期間の年数。portfolio_sharpe_ratio の年率化に使う。
            0 以下（不明）の場合 portfolio_sharpe_ratio は NaN になる。

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
    # DSRの入力（sharpe_per_trade, num_trades）が同一母集団になるよう、有効銘柄の
    # 取引リターンを1系列にプールしてから算出する。銘柄別Sharpeの単純平均は
    # 「銘柄横断でプールしたnum_trades」と対応しないため使わない（#630）。
    pooled_returns = [r for row in effective for r in row.trade_returns]
    pooled_sharpe_per_trade = _sharpe_per_trade(pooled_returns)

    # ゲート用 Sharpe。銘柄別 Sharpe の単純平均は、3取引で採用された銘柄の発散値が
    # そのまま平均に効くため再現しない（台帳の再現ペア130組で自己相関 0.446）。
    # プール済み per-trade Sharpe を、ポートフォリオ全体の取引頻度で1回だけ年率化する。
    total_trades = sum(r.num_trades for r in effective)
    portfolio_sharpe = (
        _annualize_sharpe(pooled_sharpe_per_trade, total_trades / span_years)
        if span_years > 0
        else math.nan
    )

    return AggregatedMetrics(
        sharpe_ratio=sum(r.sharpe_ratio for r in effective) / n,
        sharpe_per_trade=pooled_sharpe_per_trade,
        portfolio_sharpe_ratio=portfolio_sharpe,
        win_rate=sum(r.win_rate for r in effective) / n,
        total_return=sum(r.total_return for r in effective) / n,
        max_drawdown=min(r.max_drawdown for r in effective),
        num_trades=sum(r.num_trades for r in effective),
        n_symbols_with_signal=n_with_signal,
        n_effective_symbols=n,
        avg_trades_per_symbol=avg_trades,
        effective_symbols=[r.symbol for r in effective],
    )
