"""
バックテストによる仮の入れ替え CLI エントリポイント

夜間ジョブ（スケジューラの nightly_backtest_promotion）と同じ処理を手で1回動かし、
結果の JSON のパスを出す。

使い方:
  py run_backtest_promotion.py              # 夜間ジョブと同じ（同じ版はもう比べない）
  py run_backtest_promotion.py --force      # 同じ版・戻した版でも比べ直す
  py run_backtest_promotion.py --dry-run    # 比べるだけで、モデルも状態も変えない
"""

import argparse


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="バックテストで候補を評価し、勝てば暫定昇格する")
    parser.add_argument(
        "--force",
        action="store_true",
        help="同じ版・戻した版でも評価し直す（見張り中は評価しない）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="評価だけして、モデルのファイルも状態も変えない"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from src.orchestration.port_wiring import wire_ports

    wire_ports()

    from src.orchestration.jobs import backtest_promotion_job

    result = backtest_promotion_job.run_backtest_promotion(force=args.force, dry_run=args.dry_run)
    print(f"outcome: {result['outcome']}")
    if result.get("result_path"):
        print(f"結果の JSON: {result['result_path']}")
    else:
        print(f"結果の JSON は書いていない: {result.get('reason')}")


if __name__ == "__main__":
    main()
