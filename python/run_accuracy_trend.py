"""
採点の JSON の流れの表を出すスクリプト（#1815。CLI ラッパー）

results/accuracy の日ごとの JSON を読み、日・モデル・ホライズン・市場ごとの的中率と
基準と rank_corr と predicted_up_rate と予想と実際の大きさと予想の散らばり
（mean_abs_predicted_ratio・mean_abs_actual_ratio・prediction_std、#2009）の表と
N 日の移動平均を CSV か Markdown で出す。
--four-week を付けると、各欄の four_week（4週の窓）とその by_market の表を出す（#2013）。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.accuracy_trend import (
    format_trend_markdown,
    load_accuracy_trend,
    load_four_week_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="採点の JSON の的中率の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="採点の JSON の置き場（省略時は results/accuracy）",
    )
    parser.add_argument("--window", type=int, default=5, help="移動平均の日数（既定 5）")
    parser.add_argument("--model", type=str, default=None, help="このモデルの行だけ出す")
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ出す（ALL 可）")
    parser.add_argument("--tail", type=int, default=None, help="組ごとの直近 N 日だけ出す")
    parser.add_argument(
        "--four-week",
        action="store_true",
        help="各欄の four_week（4週の窓）の表を出す（--window は使わない）",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy_dir = args.dir or os.path.join(get_results_dir(), "accuracy")
    if args.four_week:
        df = load_four_week_trend(accuracy_dir)
    else:
        df = load_accuracy_trend(accuracy_dir, window=args.window)
    if args.model:
        df = df[df["model"] == args.model]
    if args.market:
        df = df[df["market"] == args.market]
    if args.tail:
        df = df.groupby(["model", "horizon", "market"], sort=False).tail(args.tail)

    text = df.to_csv(index=False) if args.format == "csv" else format_trend_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"採点の JSON の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
