"""長期コホート・バックテストの結果保存とサマリ文生成。

`longterm_backtest` から `save_results` / `build_conclusion` をそのまま移設した。
"""

from __future__ import annotations

import pandas as pd

from src.backtest.longterm.config import LongtermBacktestConfig
from src.utils.results_io import save_result_csvs

_FREQ_LABELS = {
    "weekly": "週次",
    "monthly": "月次",
    "quarterly": "四半期",
    "yearly": "年次",
    "annual": "年次",
}


def save_results(equity_df: pd.DataFrame, trades_df: pd.DataFrame, market: str) -> tuple[str, str]:
    """equity / trades を results/backtest/longterm/ に CSV 保存しパスを返す。"""
    paths = save_result_csvs(
        {f"equity_{market}": equity_df, f"trades_{market}": trades_df},
        "backtest/longterm",
    )
    return paths[f"equity_{market}"], paths[f"trades_{market}"]


def build_conclusion(metrics: dict, config: LongtermBacktestConfig) -> str:
    """metrics から結論文を組み立てる（サマリ出力用）。"""
    initial = metrics.get("initial_cash", 0.0)
    final = metrics.get("final_cash", 0.0)
    mult = final / initial if initial > 0 else 0.0
    total_ret = metrics.get("total_return", 0.0)
    cagr = metrics.get("cagr", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    bench = metrics.get("benchmark_return")
    alpha = metrics.get("alpha")

    bench_part = (
        f"ベンチマーク（{metrics.get('benchmark_ticker')}: {bench:+.1%}）に対し "
        f"{alpha:+.1%}pt。"
        if bench is not None and alpha is not None
        else "ベンチマーク比較はデータ取得失敗のため省略。"
    )
    freq_label = _FREQ_LABELS.get(config.rescreen_freq, config.rescreen_freq)
    ma_weeks = config.rules.trail_ma_weeks
    return (
        f"{config.market} を {config.start}〜{config.end} に{freq_label}スクリーン・"
        f"最大{config.max_positions}銘柄保有・{ma_weeks}週線割れまでホールドの規律で運用した場合、"
        f"資産は {initial:,.0f}円→{final:,.0f}円（{mult:.2f}倍, 総リターン {total_ret:+.1%}, "
        f"CAGR {cagr:+.1%}）、最大DD {max_dd:.1%}。"
        f"期間中に 2倍到達 {metrics.get('n_2x', 0)}件 / 3倍 {metrics.get('n_3x', 0)}件 / "
        f"5倍 {metrics.get('n_5x', 0)}件 / 10倍 {metrics.get('n_10x', 0)}件。"
        f"{bench_part}"
        " ※ユニバースは現存銘柄のみのため生存者バイアスで楽観方向に歪む点に注意。"
    )
