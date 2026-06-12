"""
損益曲線チャート生成（月次レポート用）

ペーパートレードのエクイティ系列とベンチマーク（S&P500 等）を 100 起点に正規化して
重ね描きした PNG を生成する。Discord の send_webhook_file で送信する想定。
"""

from __future__ import annotations

from typing import Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # ヘッドレス環境（コンテナ・スケジューラー）用
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def build_equity_chart(
    equity: pd.Series,
    benchmark: Optional[pd.Series],
    out_path: str,
    title: str = "Paper Trading Equity vs S&P 500",
) -> str:
    """エクイティ vs ベンチマークの正規化チャートを PNG 保存してパスを返す。

    Args:
        equity: 日次評価額 Series（DatetimeIndex）
        benchmark: ベンチマーク終値 Series（None ならエクイティのみ描画）
        out_path: 保存先 PNG パス
        title: チャートタイトル

    Raises:
        ValueError: equity が空、または正規化基準が取れない場合
    """
    if equity is None or equity.empty:
        raise ValueError("エクイティ系列が空です")

    equity = equity.dropna()
    base = equity.iloc[0]
    if base == 0:
        raise ValueError("エクイティ初期値が 0 のため正規化できません")
    equity_norm = equity / base * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_norm.index, equity_norm.values, label="Paper Trading", linewidth=2)

    if benchmark is not None and not benchmark.dropna().empty:
        bench = benchmark.dropna()
        bench.index = pd.to_datetime(bench.index).tz_localize(None)
        bench = bench.reindex(equity_norm.index).ffill().bfill()
        if not bench.dropna().empty and bench.iloc[0] != 0:
            bench_norm = bench / bench.iloc[0] * 100.0
            ax.plot(
                bench_norm.index,
                bench_norm.values,
                label="S&P 500",
                linewidth=1.5,
                linestyle="--",
            )

    ax.set_title(title)
    ax.set_ylabel("Normalized (start = 100)")
    ax.axhline(100.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("損益曲線チャートを保存: %s", out_path)
    return out_path
