#!/usr/bin/env python3
"""src/ 配下の Python ファイル行数ゲート (Issue #497 フォローアップ)。

1 ファイルあたり MAX_LINES 行以下を強制し、神クラス/肥大化の再発を防ぐ。

既存の超過ファイルは GRANDFATHERED に「現状行数」を上限として登録する（ratchet 方式）:
  - grandfathered ファイルは登録上限を超えて**増やせない**（これ以上の肥大化を阻止）。
  - それ以外のファイルは MAX_LINES を超えられない（新規肥大化を阻止）。
  - 責務分割して MAX_LINES 以下になったら **GRANDFATHERED から該当行を削除**すること。
    上限が恒久的に下がり、二度と肥大化しない（ratchet が締まる）。

ローカル実行:  python scripts/check_file_size.py
CI:            file-size-check.yml / check-ci スクリプトから呼ばれる。
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 600

# src/ からの相対パス -> 現状行数を上限とする grandfather 登録。
# 分割して MAX_LINES 以下にしたら、この辞書から該当行を削除すること。
GRANDFATHERED: dict[str, int] = {
    "backtest/optimizer.py": 944,
    "prediction/training_pipeline.py": 790,
    "backtest/metrics.py": 707,
    "backtest/pipeline.py": 645,
}

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> int:
    violations: list[tuple[str, int, int]] = []
    shrunk: list[tuple[str, int]] = []
    seen_grandfathered: set[str] = set()

    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        lines = _count_lines(path)
        ceiling = GRANDFATHERED.get(rel, MAX_LINES)
        if rel in GRANDFATHERED:
            seen_grandfathered.add(rel)
        if lines > ceiling:
            violations.append((rel, lines, ceiling))
        elif rel in GRANDFATHERED and lines <= MAX_LINES:
            shrunk.append((rel, lines))

    # 分割完了で MAX_LINES 以下になった grandfathered ファイルは登録解除を促す
    for rel, lines in shrunk:
        print(
            f"::warning::src/{rel} は {lines} 行 (<= {MAX_LINES})。"
            f"GRANDFATHERED から削除して ratchet を締めてください。"
        )

    # 登録済みなのに存在しないパス（移動/分割済み）も掃除を促す
    for rel in sorted(set(GRANDFATHERED) - seen_grandfathered):
        print(f"::warning::GRANDFATHERED の src/{rel} が見つかりません。登録を削除してください。")

    if violations:
        print(f"\n[file-size-check] 行数ゲート違反 (上限 {MAX_LINES} 行):")
        for rel, lines, ceiling in violations:
            limit_note = (
                f"grandfather 上限 {ceiling}" if rel in GRANDFATHERED else f"上限 {ceiling}"
            )
            print(f"  src/{rel}: {lines} 行 ({limit_note})")
        print(
            "\n責務単位でモジュール分割してください"
            "（re-export ファサード方式が後方互換を保てて安全）。"
        )
        return 1

    print(
        f"[file-size-check] OK: src/ 配下の全 .py が上限 {MAX_LINES} 行以内"
        f"（grandfathered {len(GRANDFATHERED)} 件は現状維持）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
