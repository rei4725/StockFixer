"""
損益曲線チャート生成（月次レポート用）

ペーパートレードのエクイティ系列とベンチマーク（S&P500 等）を 100 起点に正規化して
重ね描きした PNG を生成する。Discord の send_webhook_file で送信する想定。
"""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # ヘッドレス環境（コンテナ・スケジューラー）用
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


_DEFAULT_BENCHMARK_LABELS = frozenset({"S&P 500"})


def _is_benchmark(label: str, benchmark_labels: frozenset[str]) -> bool:
    """ラベルがベンチマーク系列かどうかを判定する。

    呼び出し元が系列ラベルに "(since YYYY-MM-DD)" のような開始日サフィックスを
    付与することがあるため、完全一致に加えて "<ラベル> " で始まる場合も一致とみなす。
    """
    return any(label == b or label.startswith(f"{b} ") for b in benchmark_labels)


def build_equity_chart(
    series: dict[str, pd.Series],
    out_path: str,
    title: str = "Paper Trading Equity Comparison",
    benchmark_labels: frozenset[str] = _DEFAULT_BENCHMARK_LABELS,
) -> str:
    """複数のエクイティ系列を正規化して重ね描きした PNG を保存しパスを返す。

    各系列は自身の先頭値を 100 として個別に正規化する（絶対額でなく相対推移で比較）。
    benchmark_labels に含まれるラベルの系列は点線で描画する。

    Args:
        series: {ラベル: 日次評価額 Series（DatetimeIndex）}
        out_path: 保存先 PNG パス
        title: チャートタイトル
        benchmark_labels: 点線で描くラベルの集合

    Raises:
        ValueError: series が空、または有効な（空でも0基準でもない）系列が1つも無い場合
    """
    if not series:
        raise ValueError("系列が空です")

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0

    for label, values in series.items():
        if values is None:
            continue
        cleaned = values.dropna()
        if cleaned.empty:
            logger.info("エクイティ系列が空のため描画から除外: %s", label)
            continue
        base = cleaned.iloc[0]
        if base == 0:
            logger.warning("エクイティ初期値が 0 のため描画から除外: %s", label)
            continue
        cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
        normalized = cleaned / base * 100.0
        is_benchmark = _is_benchmark(label, benchmark_labels)
        linestyle = "--" if is_benchmark else "-"
        linewidth = 1.5 if is_benchmark else 2.0
        ax.plot(
            normalized.index,
            normalized.values,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("描画可能な系列がありません")

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
