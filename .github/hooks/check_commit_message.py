#!/usr/bin/env python3
"""
コミットメッセージフォーマット検証スクリプト

Conventional Commits スタイルに従っているかをチェックする。
形式: <type>(<scope>): <subject>

例:
  feat(data-pipeline): 株価取得エラーハンドリング追加
  fix(scheduler): タイムゾーン設定バグを修正
  docs: README更新
  refactor(models): 予測モデル構造を整理
"""

import sys
import re
from pathlib import Path

# 許可されるコミットタイプ
ALLOWED_TYPES = {
    "feat",       # 新機能
    "fix",        # バグ修正
    "docs",       # ドキュメント
    "style",      # コード整形 (意味なし)
    "refactor",   # リファクタリング
    "perf",       # パフォーマンス改善
    "test",       # テスト追加・修正
    "ci",         # CI/CD設定
    "chore",      # その他の変更
    "revert",     # コミット取り消し
}

# スコープの推奨例
RECOMMENDED_SCOPES = {
    "data-pipeline",
    "scheduler",
    "models",
    "prediction",
    "backtest",
    "api",
    "db",
    "docker",
    "tests",
    "docs",
    "deps",  # 依存関係更新
}


def validate_commit_message(msg: str) -> tuple[bool, str]:
    """
    コミットメッセージを検証する。
    
    Args:
        msg: コミットメッセージ（最初の行）
    
    Returns:
        (is_valid, message): 検証結果とメッセージ
    """
    # 空白行・コメント行を削除
    lines = [line for line in msg.strip().split('\n') if line and not line.startswith('#')]
    if not lines:
        return False, "コミットメッセージが空です。"
    
    first_line = lines[0]
    
    # Merge コミット・Revert コミットは許可
    if first_line.startswith("Merge ") or first_line.startswith("Revert "):
        return True, "OK"
    
    # 形式: <type>(<scope>): <subject> または <type>: <subject>
    pattern = r'^(\w+)(?:\(([^)]+)\))?: (.+)$'
    match = re.match(pattern, first_line)
    
    if not match:
        return False, (
            f"✗ コミットメッセージ形式が正しくありません。\n"
            f"  形式: <type>(<scope>): <subject> または <type>: <subject>\n"
            f"  例: feat(data-pipeline): 株価取得エラーハンドリング追加\n"
            f"  例: fix: タイムゾーン設定バグを修正\n"
            f"  入力: {first_line}"
        )
    
    commit_type, scope, subject = match.groups()
    
    # タイプをチェック
    if commit_type not in ALLOWED_TYPES:
        return False, (
            f"✗ コミットタイプ '{commit_type}' は許可されていません。\n"
            f"  許可されるタイプ: {', '.join(sorted(ALLOWED_TYPES))}\n"
            f"  入力: {first_line}"
        )
    
    # サブジェクトをチェック
    if not subject:
        return False, "✗ サブジェクトが空です。"
    
    if len(subject) > 100:
        return False, (
            f"✗ サブジェクトが長すぎます（最大100文字、現在{len(subject)}文字）。\n"
            f"  入力: {subject}"
        )
    
    # 最初の文字を確認（大文字推奨）
    if subject[0].islower():
        # これは warning レベルなので許可する
        pass
    
    # スコープが指定されている場合、推奨スコープをチェック
    if scope and scope not in RECOMMENDED_SCOPES:
        # これは warning レベルなので許可する
        pass
    
    return True, "OK"


def main():
    """pre-commit から呼ばれるエントリポイント"""
    if len(sys.argv) < 2:
        print("使用方法: check_commit_message.py <commit_msg_file>")
        sys.exit(1)
    
    commit_msg_file = sys.argv[1]
    
    try:
        with open(commit_msg_file, 'r', encoding='utf-8') as f:
            msg = f.read()
    except Exception as e:
        print(f"✗ ファイル読み込みエラー: {e}")
        sys.exit(1)
    
    is_valid, msg_result = validate_commit_message(msg)
    
    if is_valid:
        print("✓ コミットメッセージ: OK")
        sys.exit(0)
    else:
        print(msg_result)
        sys.exit(1)


if __name__ == "__main__":
    main()
