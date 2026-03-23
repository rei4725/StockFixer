r"""
全銘柄バックテスト最適化バッチスクリプト（ラッパー）

ウォッチリストの全銘柄に対してグリッドサーチ＋Walk-Forward検証を
並列実行し、最適パラメータを config/optimal_params.json に保存する。

使用例:
    # デフォルト（閾値のみ最適化・並列数3）
    py run_backtest_optimize_batch.py

    # ストップロス・テイクプロフィットも最適化
    py run_backtest_optimize_batch.py --optimize-risk

    # 並列数・閾値レンジをカスタマイズ
    py run_backtest_optimize_batch.py --max-workers 5 \
        --threshold-min 0.0 --threshold-max 0.02 --threshold-step 0.002

    # アンサンブルモード
    py run_backtest_optimize_batch.py --ensemble
"""

import argparse
import sys

from src.services.backtest_optimize_pipeline import run_optimize_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="全銘柄バックテスト最適化バッチ（並列グリッドサーチ）")
    parser.add_argument(
        "--model-type",
        type=str,
        default="XGBoostModel",
        choices=["XGBoostModel", "LightGBMModel"],
        help="モデルタイプ (default: XGBoostModel)",
    )
    parser.add_argument("--ensemble", action="store_true", help="XGBoost+LightGBMアンサンブル予測")
    parser.add_argument(
        "--source",
        type=str,
        default="file",
        choices=["file", "api", "raw"],
        help="データソース (default: file)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-Forward分割数 (default: 5)")
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="初期資金")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="取引手数料率 (default: 0.001)")
    parser.add_argument("--slippage", type=float, default=0.0, help="スリッページ (default: 0.0)")

    # 閾値グリッド
    parser.add_argument("--threshold-min", type=float, default=0.0, help="閾値の最小値 (default: 0.0)")
    parser.add_argument(
        "--threshold-max", type=float, default=0.015, help="閾値の最大値 (default: 0.015)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.001, help="閾値のステップ (default: 0.001)"
    )

    # リスク管理グリッド
    parser.add_argument(
        "--optimize-risk",
        action="store_true",
        help="ストップロス・テイクプロフィットもグリッドサーチ",
    )

    # バッチ並列数
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="銘柄間の並列数 (default: 3 ※DuckDB読み込み負荷を考慮)",
    )

    # ソート基準
    parser.add_argument(
        "--sort-by",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return", "profit_factor", "win_rate", "max_drawdown"],
        help="最適パラメータ選定基準 (default: sharpe_ratio)",
    )

    # スキップ設定
    parser.add_argument(
        "--skip-days",
        type=int,
        default=None,
        metavar="N",
        help="最終最適化からN日以内の銘柄をスキップ（例: 7 → 1週間以内は再実行しない）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = run_optimize_batch(
        model_type=args.model_type,
        ensemble=args.ensemble,
        source=args.source,
        n_splits=args.n_splits,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        optimize_risk=args.optimize_risk,
        max_workers=args.max_workers,
        sort_by=args.sort_by,
        skip_days=args.skip_days,
    )

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*60}")
    print("全銘柄最適化バッチ 完了")
    print(f"  成功:       {len(success)} 銘柄")
    print(f"  スキップ:   {len(skipped)} 銘柄")
    print(f"  失敗:       {len(errors)} 銘柄")
    if errors:
        print("\n  [失敗銘柄]")
        for e in errors:
            print(f"    {e['market']}/{e['symbol']}: {e.get('error', '不明')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"全銘柄最適化バッチ 異常終了: {e}", exc_info=True)
        sys.exit(1)
