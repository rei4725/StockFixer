"""
週の Walk-Forward の流れの表を出すスクリプト（#1926。CLI ラッパー）

results/backtest/walk_forward_reports の wf_summary_*.csv を全部読み、回と市場ごとの
銘柄数・error の数・成績の平均と中央値の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.walk_forward_trend import (
    format_walk_forward_trend_csv,
    format_walk_forward_trend_markdown,
    load_walk_forward_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="週の Walk-Forward の成績の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="wf_summary の置き場（省略時は results/backtest/walk_forward_reports）",
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ出す（ALL 可）")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    report_dir = args.dir or os.path.join(get_results_dir(), "backtest", "walk_forward_reports")
    df = load_walk_forward_trend(report_dir)
    if args.market:
        df = df[df["market"] == args.market]

    if args.format == "csv":
        text = format_walk_forward_trend_csv(df)
    else:
        text = format_walk_forward_trend_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Walk-Forward の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
