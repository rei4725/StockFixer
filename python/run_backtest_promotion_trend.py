"""
夜間の仮の入れ替えの記録の流れの表を出すスクリプト（#1890。CLI ラッパー）

results/backtest_promotion の回ごとの JSON を読み、回ごとの結果・日ごとの勝ち越し・候補と
production の物差し・市場ごとの比べた行の数・負けの基準の有無の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.backtest_promotion_trend import (
    format_backtest_promotion_csv,
    format_backtest_promotion_markdown,
    load_backtest_promotion_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="夜間の仮の入れ替えの記録の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="記録の置き場（省略時は results/backtest_promotion）",
    )
    parser.add_argument("--tail", type=int, default=None, help="直近 N 回だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.dir or os.path.join(get_results_dir(), "backtest_promotion")
    df = load_backtest_promotion_trend(results_dir)
    if args.tail:
        df = df.tail(args.tail)

    text = (
        format_backtest_promotion_csv(df)
        if args.format == "csv"
        else format_backtest_promotion_markdown(df) + "\n"
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"仮の入れ替えの記録の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
