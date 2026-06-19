#!/usr/bin/env python3
"""
バックテスト批判的レビュー CLI（B-6）

Claude にバックテストの方法論・設定を批判的レビューさせ、検出した欠陥を
factory-intake スキーマの JSON として results/factory/reports/ に書き出す。
IssueAgent の --factory-intake が GitHub Issue 草案として起票する。

定期実行は OS のタスクスケジューラ / cron から本スクリプトを呼ぶ想定。
有効化には .env に BACKTEST_REVIEW_ENABLED=true と ANTHROPIC_API_KEY が必要。

使い方:
    python run_backtest_review.py            # レビュー実行・JSON 書き込み
    python run_backtest_review.py --dry-run  # 検出のみ（書き込みなし）
"""

import argparse

from src.backtest.critical_review import run_backtest_review


def main() -> None:
    parser = argparse.ArgumentParser(description="バックテスト批判的レビュー（Claude）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="JSON を書き込まず検出した発見事項のみ表示する",
    )
    args = parser.parse_args()

    findings = run_backtest_review(dry_run=args.dry_run)

    if not findings:
        print("発見事項なし（または無効/失敗）。")
        return

    print(f"検出した発見事項: {len(findings)} 件")
    for i, f in enumerate(findings, 1):
        print(f"  {i}. [{f.get('severity', '?')}] {f.get('title', '')} ({f.get('category', '')})")
    if args.dry_run:
        print("（dry-run のため JSON は書き込んでいません）")


if __name__ == "__main__":
    main()
