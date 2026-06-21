#!/usr/bin/env python3
"""テスト穴埋めボット CLI（A-3）

coverage の未カバー行を Claude にレビューさせ、不足テストの追加案を
factory-intake スキーマの JSON として results/factory/reports/ に書き出す。
IssueAgent の --factory-intake が GitHub Issue 草案として起票する。

事前に coverage.json を生成しておくこと:
    cd python && python -m pytest tests/unit --cov=src --cov-report=json

有効化には .env に TEST_GAP_ENABLED=true が必要。バックエンドは LLM_BACKEND で
選択する（既定運用は cli=サブスク認証・CLAUDE_CODE_OAUTH_TOKEN、ANTHROPIC_API_KEY 不要）。
定期実行は OS のタスクスケジューラ / cron から本スクリプトを呼ぶ想定。

使い方:
    python run_test_gap.py                       # レビュー実行・JSON 書き込み
    python run_test_gap.py --dry-run             # 検出のみ（書き込みなし）
    python run_test_gap.py --coverage-json X.json
"""

import argparse

from src.quality.test_gap_review import run_test_gap_review


def main() -> None:
    parser = argparse.ArgumentParser(description="テスト穴埋めボット（Claude）")
    parser.add_argument(
        "--coverage-json",
        default="coverage.json",
        help="coverage JSON レポートのパス（既定: coverage.json）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="JSON を書き込まず得た提案のみ表示する",
    )
    args = parser.parse_args()

    suggestions = run_test_gap_review(coverage_json=args.coverage_json, dry_run=args.dry_run)

    if not suggestions:
        print("提案なし（または無効/対象なし/失敗）。")
        return

    print(f"テスト追加案: {len(suggestions)} 件")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. [{s.get('priority', '?')}] {s.get('target_path', '')}")
    if args.dry_run:
        print("（dry-run のため JSON は書き込んでいません）")


if __name__ == "__main__":
    main()
