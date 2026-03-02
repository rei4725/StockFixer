#!/usr/bin/env python3
"""
ファイルロック問題を検出する静的コード分析スクリプト

検出対象：
1. 複数スレッド/プロセスからの同時ファイルアクセス（Lock/Semaphore未使用）
2. to_csv + read_csv の並列実行
3. joblib.dump + joblib.load の競合
4. CSVファイルの O_WRONLY 排他オープンなし
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class FileLockChecker:
    """ファイルロック関連の問題を検査する"""

    def __init__(self):
        self.danger_patterns = {
            # 複数スレッドでの同時ファイル操作
            "thread_pool_csv_write": {
                "pattern": r"ThreadPoolExecutor.*\.to_csv",
                "message": "警告: ThreadPoolExecutor内で.to_csv()を実行。同時ファイル書き込みでファイルロック競合。",
                "severity": "high",
            },
            # joblib でのモデル保存競合
            "thread_pool_joblib_dump": {
                "pattern": r"(ThreadPoolExecutor|ProcessPoolExecutor).*joblib\.dump",
                "message": "警告: ThreadPoolExecutor内でjoblib.dump()を実行。同時ファイル書き込みでロック競合。",
                "severity": "high",
            },
            # open() で 'w' モード（排他性なし）
            "open_w_mode": {
                "pattern": r"open\(['\"].*?['\"],\s*['\"]w['\"]",
                "message": "注意: open(..., 'w') で排他ロック指定なし。複数プロセスからの追記時に破損リスク。"
                "fcntl.flock() または threading.Lock 使用を推奨。",
                "severity": "medium",
            },
            # to_csv() + read_csv() の同時実行
            "csv_concurrent_rw": {
                "pattern": r"\.to_csv\(.*?\).*?pd\.read_csv",
                "message": "注意: to_csv()直後のread_csv()は同期化されていない。"
                "複数スレッドでこのシーケンスが実行される場合、read_csv()が不完全なファイルを読む可能性。",
                "severity": "medium",
            },
            # shutil.move/copy での排他性不保証
            "shutil_without_lock": {
                "pattern": r"shutil\.(move|copy|copyfile)",
                "message": "注意: shutil.move()/copy()は排他性を保証しない。"
                "同時実行される環境では fcntl.flock() またはファイルロック併用を推奨。",
                "severity": "low",
            },
        }

        self.violation_severity_map = {
            "critical": 3,
            "high": 2,
            "medium": 1,
            "low": 0,
        }

    def check_file(self, filepath: str) -> List[Tuple[int, str, str]]:
        """
        ファイルをチェックする

        Returns:
            [(line_num, message, severity), ...]
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            return [(0, f"ファイル読込エラー: {e}", "critical")]

        violations = []

        for pattern_name, pattern_info in self.danger_patterns.items():
            pattern = pattern_info["pattern"]
            message = pattern_info["message"]
            severity = pattern_info["severity"]

            # 複数行パターンの場合は content で検索
            if "\n" in pattern:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    violations.append((line_num, message, severity))
            else:
                # 単一行パターンの場合は行ごとに検索
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        violations.append((i, message, severity))

        return violations

    def check_lock_usage(self, filepath: str) -> List[Tuple[int, str, str]]:
        """
        ファイルロック機構の適切な使用を確認

        並列ファイル操作がある場合、Lock/fcntl.flock が使用されているか確認
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception:
            return []

        violations = []

        # ThreadPoolExecutor + ファイル操作があるがLock未使用
        if "ThreadPoolExecutor" in content or "ProcessPoolExecutor" in content:
            has_file_op = any(
                keyword in content
                for keyword in [".to_csv(", ".to_json(", "joblib.dump", "open("]
            )
            has_lock = any(
                keyword in content
                for keyword in ["threading.Lock", "Lock(", "fcntl.flock"]
            )

            if has_file_op and not has_lock:
                for i, line in enumerate(lines, 1):
                    if "ThreadPoolExecutor" in line or "ProcessPoolExecutor" in line:
                        violations.append(
                            (
                                i,
                                "⚠️  並列処理がありますがロック機構(Lock/fcntl.flock)が見当たりません。"
                                "ファイルロック競合の可能性があります。",
                                "medium",
                            )
                        )
                        break

        return violations


def main():
    """エントリポイント"""
    if len(sys.argv) < 2:
        print("使用方法: check_file_lock.py <file1> [file2] ...")
        sys.exit(0)

    checker = FileLockChecker()
    all_violations = []
    exit_code = 0

    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            continue

        # 標準チェック
        violations = checker.check_file(filepath)

        # ロック機構チェック
        violations.extend(checker.check_lock_usage(filepath))

        # 違反をソート（重大度順）
        violations.sort(
            key=lambda x: checker.violation_severity_map.get(x[2], -1),
            reverse=True,
        )

        for line_num, message, severity in violations:
            print(f"{filepath}:{line_num}: [{severity.upper()}] {message}")
            all_violations.append((severity, filepath, line_num))
            if severity == "critical":
                exit_code = 1

    if all_violations:
        # サマリー表示
        critical = sum(1 for s, _, _ in all_violations if s == "critical")
        high = sum(1 for s, _, _ in all_violations if s == "high")
        medium = sum(1 for s, _, _ in all_violations if s == "medium")
        low = sum(1 for s, _, _ in all_violations if s == "low")

        print("\n" + "=" * 70)
        print("ファイルロック チェック結果")
        print("=" * 70)
        print(f"Critical: {critical}")
        print(f"High:     {high}")
        print(f"Medium:   {medium}")
        print(f"Low:      {low}")

        if critical > 0:
            exit_code = 1
        elif high > 0:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
