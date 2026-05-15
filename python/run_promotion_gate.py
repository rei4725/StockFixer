r"""
モデル昇格ゲート CLI エントリポイント（R-302）

shadow モデルの Walk-Forward 成績が current モデルを上回るか判定し、
結果を標準出力と JSON ファイルに記録する。

使い方:
  py run_promotion_gate.py --shadow UnifiedStockXGBoost_new --current UnifiedStockXGBoost
  py run_promotion_gate.py --shadow UnifiedStockXGBoost_new --current UnifiedStockXGBoost \\
      --horizon 1 --auto-approve

終了コード:
  0: 昇格基準を満たしている（eligible=True）
  1: 昇格基準を満たしていない（eligible=False）
"""

import argparse
import json
import sys

from src.prediction.promotion_gate import evaluate_promotion, save_promotion_result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="モデル昇格ゲート（R-302）: shadow モデルの昇格可否を判定する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--shadow",
        required=True,
        metavar="MODEL_NAME",
        help="shadow モデル名（例: UnifiedStockXGBoost_new）",
    )
    parser.add_argument(
        "--current",
        required=True,
        metavar="MODEL_NAME",
        help="current モデル名（例: UnifiedStockXGBoost）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        metavar="N",
        help="予測ホライズン（デフォルト: 1）",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="このフラグを指定すると手動承認なしで昇格可能とみなす",
    )
    parser.add_argument(
        "--shadow-run-id",
        default=None,
        metavar="UUID",
        help="shadow モデルの run_id（A/B テスト連携用。省略時は最新）",
    )
    parser.add_argument(
        "--current-run-id",
        default=None,
        metavar="UUID",
        help="current モデルの run_id（A/B テスト連携用。省略時は最新）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    result = evaluate_promotion(
        shadow_model_name=args.shadow,
        current_model_name=args.current,
        horizon=args.horizon,
        require_manual_approval=not args.auto_approve,
        shadow_run_id=args.shadow_run_id,
        current_run_id=args.current_run_id,
    )

    saved_path = save_promotion_result(result)

    print(f"shadow:  {result.shadow_model}")
    print(f"current: {result.current_model}")
    print(f"evaluated_at: {result.evaluated_at}")
    print(f"eligible: {result.eligible}")
    print(f"require_manual_approval: {result.require_manual_approval}")
    print(f"reason: {result.reason}")
    print("")
    print("--- criteria ---")
    print(json.dumps(result.criteria, ensure_ascii=False, indent=2))
    if result.run_ids:
        print(f"\nrun_ids: {result.run_ids}")
    if result.wf_source_file:
        print(f"wf_source_file: {result.wf_source_file}")
    print(f"\n結果を保存しました: {saved_path}")

    return 0 if result.eligible else 1


if __name__ == "__main__":
    sys.exit(main())
