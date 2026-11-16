"""
スケジューラーのジョブの実行の記録を回ごとの表にするスクリプト（#1892。CLI ラッパー）

results/scheduler_queue_state.json の events を読み、ジョブ・期間・run_id ごとに
終わりの状態・所要時間・やり直しの回数・終わりの記録が無い印の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.scheduler_run_trend import (
    filter_scheduler_runs,
    format_scheduler_run_csv,
    format_scheduler_run_markdown,
    load_scheduler_run_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="スケジューラーのジョブの実行の記録を表にする")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="記録の JSON（省略時は results/scheduler_queue_state.json）",
    )
    parser.add_argument("--job", type=str, default=None, help="この job_id の行だけ出す")
    parser.add_argument(
        "--since", type=str, default=None, help="この日（YYYY-MM-DD）以後に始めた回だけ出す"
    )
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="この終わりの status の行だけ出す（unfinished で終わりの記録が無い回）",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path or os.path.join(get_results_dir(), "scheduler_queue_state.json")
    df = filter_scheduler_runs(
        load_scheduler_run_trend(path), job=args.job, since=args.since, status=args.status
    )

    text = (
        format_scheduler_run_csv(df)
        if args.format == "csv"
        else format_scheduler_run_markdown(df) + "\n"
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
        logger.critical(f"スケジューラーの実行の記録の表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
