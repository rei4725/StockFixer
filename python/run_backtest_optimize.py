r"""
バックテスト最適化スクリプト（ラッパー）

閾値・ストップロス・テイクプロフィットのグリッドサーチを
Walk-Forward検証で実行し、最適パラメータを特定する。

使用例:
    # デフォルト（閾値のみ最適化）
    py run_backtest_optimize.py --market jp --symbol 7203

    # ストップロス・テイクプロフィットも最適化
    py run_backtest_optimize.py --market jp --symbol 7203 --optimize-risk

    # アンサンブルモード
    py run_backtest_optimize.py --market jp --symbol 7203 --ensemble

    # カスタム閾値レンジ
    py run_backtest_optimize.py --market jp --symbol 7203 \
        --threshold-min 0.0 --threshold-max 0.02 --threshold-step 0.002
"""

import argparse
import sys

from src.backtest.optimizer import (
    print_optimization_results,
    run_optimization,
    save_optimal_params_json,
    save_optimization_results,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="バックテスト最適化（グリッドサーチ）")
    parser.add_argument("--market", type=str, default="jp", help="マーケット (例: jp, us)")
    parser.add_argument("--symbol", type=str, required=True, help="銘柄コード (例: 7203, AAPL)")
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

    # 閾値グリッド
    parser.add_argument(
        "--threshold-min", type=float, default=0.0, help="閾値の最小値 (default: 0.0)"
    )
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
    parser.add_argument(
        "--fee-rate", type=float, default=0.001, help="取引手数料率 (default: 0.001)"
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=None,
        help="片道スリッページ率 (未指定時は市場別デフォルト US 5bps / JP 10bps)",
    )
    parser.add_argument(
        "--no-dynamic-slippage",
        action="store_true",
        help="出来高連動の動的スリッページ（平方根インパクト）を無効化する",
    )
    parser.add_argument(
        "--position-sizing",
        type=str,
        default="full",
        choices=["full", "fixed", "confidence", "atr"],
        help="ポジションサイジング (default: full)",
    )
    parser.add_argument(
        "--position-fraction",
        type=float,
        default=0.5,
        help="fixed モード用の資金比率 (default: 0.5)",
    )
    parser.add_argument(
        "--atr-risk-pcts",
        type=str,
        default="0.02",
        help="ATR モード時のリスク割合候補をカンマ区切りで指定 (default: 0.02)",
    )
    parser.add_argument(
        "--atr-multipliers",
        type=str,
        default="1.0",
        help="ATR モード時の倍率候補をカンマ区切りで指定 (default: 1.0)",
    )
    parser.add_argument(
        "--atr-min-fraction",
        type=float,
        default=0.1,
        help="ATR モード時の建玉下限比率 (default: 0.1)",
    )
    parser.add_argument(
        "--atr-max-fraction",
        type=float,
        default=1.0,
        help="ATR モード時の建玉上限比率 (default: 1.0)",
    )

    # ソート基準
    parser.add_argument(
        "--sort-by",
        type=str,
        default="sharpe_ratio",
        choices=[
            "sharpe_ratio",
            "total_return",
            "gross_total_return",
            "cost_impact_return",
            "profit_factor",
            "win_rate",
            "max_drawdown",
        ],
        help="結果のソート基準 (default: sharpe_ratio)",
    )
    return parser.parse_args()


def main():
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()

    result_df = run_optimization(
        market=args.market,
        symbol=args.symbol,
        model_type=args.model_type,
        ensemble=args.ensemble,
        source=args.source,
        n_splits=args.n_splits,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        dynamic_slippage=not args.no_dynamic_slippage,
        position_sizing=args.position_sizing,
        position_fraction=args.position_fraction,
        atr_risk_pcts=_parse_csv_floats(args.atr_risk_pcts),
        atr_multipliers=_parse_csv_floats(args.atr_multipliers),
        atr_min_fraction=args.atr_min_fraction,
        atr_max_fraction=args.atr_max_fraction,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        optimize_risk=args.optimize_risk,
    )

    print_optimization_results(result_df, args.sort_by)

    csv_path = save_optimization_results(result_df, args.market, args.symbol)
    logger.info(f"\n結果保存（CSV）: {csv_path}")

    json_path = save_optimal_params_json(result_df, args.market, args.symbol, sort_by=args.sort_by)
    logger.info(f"最適パラメータ保存（JSON）: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"バックテスト最適化 異常終了: {e}", exc_info=True)
        sys.exit(1)
