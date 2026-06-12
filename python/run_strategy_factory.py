"""
戦略ファクトリー手動実行 CLI（#369 Phase 1）

夜間バッチ（毎日 05:00、FACTORY_ENABLED=true で有効）と同じ処理を手動実行する。
FACTORY_ENABLED の値に関わらず実行される（force=True）。

使用例:
    # 日替わりロジックでマーケットを自動選択して実行
    py run_strategy_factory.py

    # マーケット・仮説数・シードを指定
    py run_strategy_factory.py --market jp --budget 5 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Windowsコンソール(cp932)でUTF-8出力を可能にする
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp932", "shift-jis", "shift_jis"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.port_wiring import wire_ports  # noqa: E402
from src.orchestration.scheduler import run_nightly_strategy_factory  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="戦略ファクトリー夜間バッチの手動実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--market",
        default=None,
        choices=["jp", "us"],
        help="対象マーケット（省略時は jp/us 日替わり）",
    )
    p.add_argument(
        "--budget", type=int, default=None, help="仮説数（省略時は FACTORY_NIGHTLY_BUDGET）"
    )
    p.add_argument("--seed", type=int, default=None, help="サンプラーシード（省略時は日付ベース）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wire_ports()
    run_nightly_strategy_factory(force=True, market=args.market, budget=args.budget, seed=args.seed)


if __name__ == "__main__":
    main()
