"""
パフォーマンスリグレッション検出スクリプト

CI で前回の benchmark_results.json と比較し、
20% 以上遅くなった場合は警告メッセージを regression_warning.txt に書き出して
終了コード 1 で終了する。

初回実行時（ベースラインなし）は正常終了する。
"""

import json
import os
import sys

THRESHOLD = 0.20  # 20% 超過でリグレッション判定

PERF_DIR = os.path.dirname(__file__)
PREV_FILE = os.path.join(PERF_DIR, "benchmark_results_prev.json")
CURR_FILE = os.path.join(PERF_DIR, "benchmark_results.json")
WARNING_FILE = os.path.join(os.path.dirname(PERF_DIR), "..", "regression_warning.txt")


def main() -> int:
    if not os.path.exists(PREV_FILE):
        print("ベースラインが存在しないため、初回ベースラインとして記録します。")
        return 0

    if not os.path.exists(CURR_FILE):
        print("現在のベンチマーク結果が見つかりません。")
        return 0

    with open(PREV_FILE) as f:
        prev = json.load(f)
    with open(CURR_FILE) as f:
        curr = json.load(f)

    regressions = []
    for name, curr_time in curr.items():
        if name not in prev:
            continue
        prev_time = prev[name]
        if prev_time > 0 and curr_time > prev_time * (1 + THRESHOLD):
            pct = (curr_time - prev_time) / prev_time * 100
            regressions.append(f"- `{name}`: {prev_time:.3f}s → {curr_time:.3f}s (+{pct:.1f}%)")

    if not regressions:
        print("パフォーマンスリグレッションは検出されませんでした。")
        return 0

    lines = [
        "## ⚠️ パフォーマンスリグレッション検出",
        "",
        "以下のベンチマークが前回比 20% 以上遅くなりました:",
        "",
        *regressions,
        "",
        "マージ前に原因を調査してください。",
    ]
    msg = "\n".join(lines)
    print(msg)

    warning_path = os.path.normpath(WARNING_FILE)
    with open(warning_path, "w", encoding="utf-8") as f:
        f.write(msg)

    return 1


if __name__ == "__main__":
    sys.exit(main())
