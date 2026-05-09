#!/usr/bin/env python3
"""
アーキテクチャ違反（旧モジュール経由 import）を検出する静的コード分析スクリプト

検出対象：
1. from src.data.*       → src.market_data に移行済み
2. from src.models.*     → src.prediction に移行済み
3. from src.brokers.*    → src.trading.brokers に移行済み
4. from src.analysis.*   → src.market_data 推奨
5. from src.strategy.*   → src.trading 推奨
6. from src.services.*   → 各 BC 直下推奨

自動起票時ラベル: architecture-violation, auto-ok
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class ArchViolationChecker:
    """旧モジュール経由 import を検査する"""

    VIOLATION_PATTERNS = {
        "legacy_data": {
            "pattern": r"from\s+src\.data\.",
            "message": (
                "[arch-violation] from src.data.* は廃止パス。"
                "src.market_data を使用してください。"
            ),
            "severity": "high",
        },
        "legacy_models": {
            "pattern": r"from\s+src\.models\.",
            "message": (
                "[arch-violation] from src.models.* は廃止パス。"
                "src.prediction を使用してください。"
            ),
            "severity": "high",
        },
        "legacy_brokers": {
            "pattern": r"from\s+src\.brokers\.",
            "message": (
                "[arch-violation] from src.brokers.* は廃止パス。"
                "src.trading.brokers を使用してください。"
            ),
            "severity": "high",
        },
        "legacy_analysis": {
            "pattern": r"from\s+src\.analysis\.",
            "message": (
                "[arch-violation] from src.analysis.* は廃止パス。"
                "src.market_data を推奨します。"
            ),
            "severity": "high",
        },
        "legacy_strategy": {
            "pattern": r"from\s+src\.strategy\.",
            "message": (
                "[arch-violation] from src.strategy.* は廃止パス。"
                "src.trading を推奨します。"
            ),
            "severity": "high",
        },
        "legacy_services": {
            "pattern": r"from\s+src\.services\.",
            "message": (
                "[arch-violation] from src.services.* は廃止パス。"
                "各 BC 直下（src.prediction / src.trading 等）を推奨します。"
            ),
            "severity": "high",
        },
    }

    SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}

    def check_file(self, filepath: str) -> List[Tuple[int, str, str]]:
        """
        Returns:
            [(line_num, message, severity), ...]
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return [(0, f"ファイル読込エラー: {e}", "critical")]

        violations = []
        for pattern_name, info in self.VIOLATION_PATTERNS.items():
            regex = info["pattern"]
            message = info["message"]
            severity = info["severity"]
            for i, line in enumerate(lines, 1):
                if re.search(regex, line):
                    violations.append((i, message, severity))

        return violations


def main() -> None:
    """エントリポイント"""
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("使用方法: check_arch_violation.py <file1> [file2] ...")
        sys.exit(0)

    checker = ArchViolationChecker()
    all_violations = []
    exit_code = 0

    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            continue

        violations = checker.check_file(filepath)
        violations.sort(
            key=lambda x: checker.SEVERITY_RANK.get(x[2], -1),
            reverse=True,
        )

        for line_num, message, severity in violations:
            print(f"{filepath}:{line_num}: [{severity.upper()}] {message}")
            all_violations.append((severity, filepath, line_num))
            if severity in ("critical", "high"):
                exit_code = 1

    if all_violations:
        critical = sum(1 for s, _, _ in all_violations if s == "critical")
        high = sum(1 for s, _, _ in all_violations if s == "high")
        medium = sum(1 for s, _, _ in all_violations if s == "medium")
        low = sum(1 for s, _, _ in all_violations if s == "low")

        print("\n" + "=" * 70)
        print("アーキテクチャ違反チェック結果")
        print("=" * 70)
        print(f"Critical: {critical}")
        print(f"High:     {high}")
        print(f"Medium:   {medium}")
        print(f"Low:      {low}")
        print()
        print("違反が検出された場合は GitHub Issue を作成してください。")
        print("ラベル: architecture-violation, auto-ok")

        if exit_code != 0:
            print("\n[ERROR] アーキテクチャ違反があります。修正してからコミットしてください。\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
