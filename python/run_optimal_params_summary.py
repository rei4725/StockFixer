"""
最適パラメータの JSON を市場ごとに集計した表を出すスクリプト（#1950。CLI ラッパー）

results/optimal_params.json を読み、市場ごとに1行で、行の数・timestamp の古さ・
num_trades の区分・threshold・stop_loss_pct の null・position_sizing の表を CSV か Markdown で出す。
--stale で、N 日より古い銘柄の一覧も出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
ファイルが無いときは何も書かずに終わる。
"""

import argparse
import sys

from src.reporting.optimal_params_summary import (
    DEFAULT_STALE_DAYS,
    build_param_rows,
    format_summary_markdown,
    load_optimal_params,
    stale_rows,
    summarize_by_market,
)
from src.utils.logger import get_logger
from src.utils.optimal_params_loader import optimal_params_results_path

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="最適パラメータの JSON を市場ごとに集計する")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="最適パラメータの JSON（省略時は results/optimal_params.json）",
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ出す")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_STALE_DAYS, help="この日数より古い行を古いと数える"
    )
    parser.add_argument("--stale", action="store_true", help="古い銘柄の一覧も出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def _render(df, fmt: str) -> str:
    return df.to_csv(index=False) if fmt == "csv" else format_summary_markdown(df) + "\n"


def main():
    args = parse_args()
    path = args.path or str(optimal_params_results_path())
    params = load_optimal_params(path)
    if params is None:
        return

    rows = build_param_rows(params)
    if args.market:
        rows = rows[rows["market"] == args.market]

    text = _render(summarize_by_market(rows, stale_days=args.days), args.format)
    if args.stale:
        text += "\n" + _render(stale_rows(rows, stale_days=args.days), args.format)

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"最適パラメータの集計の表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
