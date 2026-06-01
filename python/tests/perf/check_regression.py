"""
パフォーマンスリグレッション検出スクリプト

CI で前回の benchmark_results.json と比較し、以下の両条件を満たす場合に
警告メッセージを regression_warning.txt に書き出して終了コード 1 で終了する。

  - 相対変化が THRESHOLD (20%) 超
  - かつ絶対差が MIN_ABS_DELTA (50ms) 超

100ms 未満の計測は CI ランナーのスケジューリングノイズで容易に 20〜80% 振れるため、
絶対差による下限を設けて誤検知を防ぐ。

初回実行時（ベースラインなし）は正常終了する。
"""

import json
import os
import sys

THRESHOLD = 0.20  # 相対変化 20% 超でリグレッション候補
MIN_ABS_DELTA = 0.050  # 絶対差 50ms 未満はノイズとみなして無視

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
        delta = curr_time - prev_time
        if prev_time > 0 and delta > MIN_ABS_DELTA and curr_time > prev_time * (1 + THRESHOLD):
            pct = delta / prev_time * 100
            regressions.append(f"- `{name}`: {prev_time:.3f}s → {curr_time:.3f}s (+{pct:.1f}%)")

    if not regressions:
        print("パフォーマンスリグレッションは検出されませんでした。")
        return 0

    lines = [
        "## ⚠️ パフォーマンスリグレッション検出",
        "",
        "以下のベンチマークが前回比 20% 以上かつ 50ms 超遅くなりました:",
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
