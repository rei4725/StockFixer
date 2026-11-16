"""
値動きに合わせた量と損切りの線の比べの流れの表を出すスクリプト（#1864。CLI ラッパー）

results/backtest_compare/volatility_sizing の週ごとの JSON を読み、週・市場・形・k ごとの
値と、同じ週・同じ市場の fixed との差の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.volatility_compare_trend import format_compare_markdown, load_compare_trend
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="値動きの量と損切りの線の比べの流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="比べの JSON の置き場（省略時は results/backtest_compare/volatility_sizing）",
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    compare_dir = args.dir or os.path.join(
        get_results_dir(), "backtest_compare", "volatility_sizing"
    )
    df = load_compare_trend(compare_dir)
    if args.market:
        df = df[df["market"] == args.market]

    text = df.to_csv(index=False) if args.format == "csv" else format_compare_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"比べの流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
