"""
週ごとの銘柄リストの更新 CLI エントリポイント（GitHub Actions の weekly-watchlist-refresh から呼ぶ）

指数構成銘柄（S&P500 / 日経225）と watchlist.json を同期し、結果を Step Summary に書く。
指数銘柄の取得に失敗した市場があれば exit 1 で終える。

使い方:
  py run_watchlist_refresh.py [--summary-path PATH]
"""

import argparse
import os
import sys

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="watchlist.json を指数構成銘柄と同期する")
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="結果の Markdown を追記するファイル（既定: GITHUB_STEP_SUMMARY）",
    )
    args = parser.parse_args(argv)

    from src.watchlist import manager

    diffs = manager.run_watchlist_refresh()
    summary = manager.format_refresh_summary(diffs)
    print(summary)
    if args.summary_path:
        with open(args.summary_path, "a", encoding="utf-8") as f:
            f.write(summary)

    failed = manager.failed_markets(diffs)
    if failed:
        print(f"::error::指数銘柄の取得失敗: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
