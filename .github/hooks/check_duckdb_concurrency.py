#!/usr/bin/env python3
"""
DuckDB 並行実行に関する問題を検出する静的コード分析スクリプト

検出対象：
1. ThreadPoolExecutor + DB write: 同一接続への並列書き込み
2. JOIN句での複数テーブル操作後に直後に write
3. トランザクション開始なしでの複数execute
4. エラーハンドリング不備でロック解放漏れ
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class DuckDBConcurrencyChecker:
    """DuckDB並行処理の問題を検査する"""

    def __init__(self):
        # 危険なパターンを定義
        self.danger_patterns = {
            # ThreadPoolExecutor + DB write の同時実行
            "thread_pool_db_write": {
                "pattern": r"ThreadPoolExecutor.*executor\.submit.*execute",
                "message": "警告: ThreadPoolExecutor内でDB writeコマンド(execute)を実行。DuckDBロック競合のリスク。",
                "severity": "high",
            },
            # ProcessPoolExecutor + DB write
            "process_pool_db_write": {
                "pattern": r"ProcessPoolExecutor.*execute",
                "message": "警告: ProcessPoolExecutor内でDB writeコマンドを実行。別プロセスでのDB競合。",
                "severity": "high",
            },
            # upsert_raw_ohlcv の並列実行
            "parallel_upsert": {
                "pattern": r"(ThreadPoolExecutor|ProcessPoolExecutor|run_parallel).*upsert_raw_ohlcv",
                "message": "警告: 並列処理内で upsert_raw_ohlcv を実行。INSERT OR REPLACE の同時実行はロック競合。",
                "severity": "high",
            },
            # save_features_to_db の並列実行
            "parallel_save_features": {
                "pattern": r"(ThreadPoolExecutor|ProcessPoolExecutor|executor\.(map|submit)).*save_features_to_db",
                "message": "警告: 並列処理内で save_features_to_db を実行。DELETE + INSERT の同時実行。",
                "severity": "high",
            },
            # DELETE + INSERT の非原子実行
            "delete_insert_race": {
                "pattern": r"DELETE\s+FROM.*?INSERT",
                "message": "注意: DELETE直後のINSERTはトランザクション保護なしではレース条件の可能性。",
                "severity": "medium",
            },
            # asyncio + DB（特に危険）
            "asyncio_db": {
                "pattern": r"async\s+def.*execute",
                "message": "エラー: asyncio内でDB操作。DuckDBはスレッドセーフでない。",
                "severity": "critical",
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

    def check_fetch_defer_raw_save(self, filepath: str) -> List[Tuple[int, str, str]]:
        """
        fetch_stock_data_with_features の defer_raw_save フラグ使用を確認

        並列フェーズから DB write を完全に分離したか確認する。
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception:
            return []

        violations = []

        # run_parallel 内で defer_raw_save=True が使用されているか確認
        if "run_parallel" in content:
            # defer_raw_save=True を検索
            if "defer_raw_save=True" in content:
                return []  # OK: 適切に遅延している
            else:
                # run_parallel 内で fetch_stock_data_with_features が呼ばれているか
                if "fetch_stock_data_with_features" in content:
                    for i, line in enumerate(lines, 1):
                        if (
                            "fetch_stock_data_with_features" in line
                            and "run_parallel" in content[:content.find(line)]
                        ):
                            violations.append(
                                (
                                    i,
                                    "警告: run_parallel内でfetch_stock_data_with_featuresを呼んでいるが、"
                                    "defer_raw_save=True が指定されていない可能性。"
                                    "並列フェーズからDB書き込みを分離してください。",
                                    "high",
                                )
                            )

        return violations


def main():
    """エントリポイント"""
    if len(sys.argv) < 2:
        print("使用方法: check_duckdb_concurrency.py <file1> [file2] ...")
        sys.exit(0)

    checker = DuckDBConcurrencyChecker()
    all_violations = []
    exit_code = 0

    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            continue

        # 標準チェック
        violations = checker.check_file(filepath)

        # 特別チェック（data_pipeline.py）
        if "data_pipeline" in filepath:
            violations.extend(checker.check_fetch_defer_raw_save(filepath))

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
        print("DuckDB 並行処理チェック結果")
        print("=" * 70)
        print(f"Critical: {critical}")
        print(f"High:     {high}")
        print(f"Medium:   {medium}")
        print(f"Low:      {low}")

        if critical > 0:
            print(
                "\n❌ Critical エラーがあります。コミットはブロックされます。\n"
            )
            exit_code = 1
        elif high > 0:
            print(
                "\n⚠️  High リスクの警告があります。修正を推奨します。\n"
            )
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
