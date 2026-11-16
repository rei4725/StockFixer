"""
戦略ファクトリーの台帳 factory_runs の流れの表を出すスクリプト（#1907。CLI ラッパー）

factory_runs を読み、評価の日・市場ごとに評価の数・合格の数・ゲートの理由ごとの落ちた数の表を
CSV か Markdown で出す。読むだけで、DB にもファイルにも書かない（--output を渡したときだけ
そのファイルに書く）。
"""

import argparse
import sys
from datetime import datetime

from src.reporting.factory_runs_trend import (
    build_factory_runs_trend,
    format_factory_runs_trend_markdown,
)
from src.utils.db.factory_runs import load_factory_runs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="戦略ファクトリーの評価の数・合格の数・ゲートの理由ごとの落ちた数の流れを表にする"
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ読む")
    parser.add_argument(
        "--since",
        type=datetime.fromisoformat,
        default=None,
        help="evaluated_at がこの日時（YYYY-MM-DD）以後の行だけ読む",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    df = build_factory_runs_trend(load_factory_runs(market=args.market, since=args.since))

    text = (
        df.to_csv(index=False)
        if args.format == "csv"
        else format_factory_runs_trend_markdown(df) + "\n"
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
        logger.critical(f"factory_runs の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
