"""ポートフォリオ結果の保存・可視化・標準出力表示。

Issue #511: 肥大化した portfolio.py を責務分割。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_portfolio_results(
    equity_df: pd.DataFrame,
    metrics: dict[str, Any],
    holdings_df: pd.DataFrame,
    market: str,
    top_n: int,
    rebalance_freq: str,
) -> str:
    """結果を CSV に保存する。保存先パスを返す。"""
    from src.utils.data_path_utils import ensure_dir, get_results_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{get_results_dir()}/backtest/portfolio"
    ensure_dir(out_dir)

    prefix = f"portfolio_{market or 'all'}_{rebalance_freq}_top{top_n}_{ts}"
    equity_path = f"{out_dir}/{prefix}_equity.csv"
    holdings_path = f"{out_dir}/{prefix}_holdings.csv"

    equity_df.to_csv(equity_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    print(f"\nエクイティカーブ保存: {equity_path}")
    print(f"保有銘柄推移保存:     {holdings_path}")
    return equity_path


def plot_portfolio(
    equity_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Optional[str] = None,
    market: str = "",
    top_n: int = 5,
    rebalance_freq: str = "weekly",
) -> str:
    """ポートフォリオ結果グラフ（エクイティ + 銘柄寄与 + ターンオーバー）を PNG 保存する。"""
    if equity_df is None or equity_df.empty:
        return ""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib がインストールされていません。")

    equity_df = equity_df.copy()
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    title = f"ポートフォリオバックテスト ({market or 'ALL'} | Top-{top_n} | {rebalance_freq})"
    fig.suptitle(title, fontsize=13)

    # ── 1. エクイティカーブ ──
    ax1 = axes[0]
    pf_ret = (equity_df["portfolio_value"] / equity_df["portfolio_value"].iloc[0] - 1) * 100
    ew_ret = (equity_df["equal_weight_value"] / equity_df["equal_weight_value"].iloc[0] - 1) * 100
    ax1.plot(
        equity_df["date"], pf_ret, label=f"Top-{top_n} 予測比例", color="steelblue", linewidth=1.5
    )
    ax1.plot(
        equity_df["date"],
        ew_ret,
        label="等分ベンチマーク",
        color="orange",
        linewidth=1.2,
        linestyle="--",
    )
    ax1.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax1.set_ylabel("累積リターン (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 2. 銘柄寄与度（保有銘柄のユニーク出現頻度） ──
    ax2 = axes[1]
    if (
        not holdings_df.empty
        and "symbol" in holdings_df.columns
        and "weight" in holdings_df.columns
    ):
        contrib = (
            holdings_df.groupby("symbol")["weight"].sum().sort_values(ascending=False).head(15)
        )
        ax2.bar(contrib.index, contrib.values, color="mediumseagreen")
        ax2.set_ylabel("累積ウェイト合計")
        ax2.set_xlabel("銘柄")
        ax2.set_title("保有頻度（上位15銘柄）")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "銘柄データなし", ha="center", va="center", transform=ax2.transAxes)

    # ── 3. ターンオーバー率 ──
    ax3 = axes[2]
    if not holdings_df.empty and "turnover" in holdings_df.columns:
        to_df = holdings_df.drop_duplicates(subset=["rebalance_date"])[
            ["rebalance_date", "turnover"]
        ].copy()
        to_df["rebalance_date"] = pd.to_datetime(to_df["rebalance_date"])
        ax3.bar(to_df["rebalance_date"], to_df["turnover"] * 100, color="coral", width=3)
        ax3.set_ylabel("ターンオーバー率 (%)")
        ax3.set_xlabel("リバランス日")
        ax3.set_title("リバランスごとのターンオーバー")
        ax3.xaxis.set_major_formatter(  # type: ignore[no-untyped-call]
            mdates.DateFormatter("%Y-%m")
        )
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(
            0.5, 0.5, "ターンオーバーデータなし", ha="center", va="center", transform=ax3.transAxes
        )

    # 指標サマリー
    info_parts = [
        f"Total Return: {metrics.get('total_return', 0):.2%}",
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
        f"MaxDD: {metrics.get('max_drawdown', 0):.2%}",
    ]
    fig.text(0.5, 0.005, "  |  ".join(info_parts), ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))

    import os

    if output_dir is None:
        from src.utils.data_path_utils import get_results_dir

        output_dir = os.path.join(get_results_dir(), "backtest", "portfolio")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"portfolio_{market or 'all'}_{rebalance_freq}_top{top_n}_{ts}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_portfolio_metrics(
    metrics: dict[str, Any], market: str, top_n: int, rebalance_freq: str
) -> None:
    """ポートフォリオメトリクスを標準出力に表示する。"""
    if not metrics:
        return
    label = f"ポートフォリオ ({market or 'ALL'} | Top-{top_n} | {rebalance_freq})"
    print(f"\n{'='*55}")
    print(f" {label}")
    print(f"{'='*55}")
    regime_metrics = metrics.get("regime_metrics", {})
    for k, v in metrics.items():
        if k == "regime_metrics":
            continue
        print(f"  {k:25s}: {v}")
    if regime_metrics:
        print("  regime_metrics:")
        for label in ("all", "bull", "bear", "range"):
            leg = regime_metrics.get(label)
            if not leg:
                continue
            print(
                "    "
                f"{label:5s} days={leg['days']:3d} "
                f"return={leg['total_return']:.4f} "
                f"sharpe={leg['sharpe_ratio']:.4f} "
                f"hit={leg['hit_rate']:.4f}"
            )
    print(f"{'='*55}")
