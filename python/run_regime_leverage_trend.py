"""
紙のレジームレバレッジ戦略の記録 regime_leverage_log の流れの表を出すスクリプト（#2094。CLI ラッパー）

regime_leverage_log を読み、日本時間の日付ごとに最後の action と reason・shares・評価額・
建てたときからの増減率・維持率・損切りの値の表を CSV か Markdown で出す。読むだけで、DB にも
ファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import sys
from datetime import datetime

from src.reporting.regime_leverage_trend import (
    build_regime_leverage_trend,
    format_regime_leverage_trend_markdown,
)
from src.trading.regime_leverage_strategy.repository import load_regime_leverage_log
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="紙のレジームレバレッジ戦略の評価額・維持率・判定の理由の流れを表にする"
    )
    parser.add_argument(
        "--since",
        type=datetime.fromisoformat,
        default=None,
        help="executed_at（UTC）がこの日時（YYYY-MM-DD）以後の行だけ読む",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    df = build_regime_leverage_trend(load_regime_leverage_log(since=args.since))

    text = (
        df.to_csv(index=False)
        if args.format == "csv"
        else format_regime_leverage_trend_markdown(df) + "\n"
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
        logger.critical(f"regime_leverage_log の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
