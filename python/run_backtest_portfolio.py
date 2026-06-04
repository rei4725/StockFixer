r"""
ポートフォリオバックテスト実行スクリプト

全銘柄の予測シグナルを統合し、Top-N 銘柄に予測スコア比例配分（softmax）で
資金を投じるポートフォリオシミュレーションを実行する。

使用例:
    # JPマーケット全銘柄・上位5銘柄・週次リバランス
    py run_backtest_portfolio.py --market jp --top-n 5 --rebalance-freq weekly

    # 全マーケット・上位10銘柄・月次リバランス・グラフ保存
    py run_backtest_portfolio.py --top-n 10 --rebalance-freq monthly --save-chart

    # LightGBM + アンサンブル
    py run_backtest_portfolio.py --market jp --model-type LightGBMModel --ensemble

    # Top-N を変えて比較（3 / 5 / 10 / 20）
    py run_backtest_portfolio.py --market jp --top-n 3 5 10 20 --rebalance-freq weekly
"""

import argparse
import sys

from config.settings import MAX_SECTOR_POSITIONS
from src.backtest.portfolio import (
    plot_portfolio,
    print_portfolio_metrics,
    run_portfolio_backtest,
    save_portfolio_results,
)
from src.backtest.ports import set_model_manager_factory
from src.prediction.manager import ModelManager
from src.utils.logger import get_logger

set_model_manager_factory(ModelManager)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ポートフォリオバックテストを実行する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基本設定
    parser.add_argument(
        "--market",
        type=str,
        default=None,
        choices=["jp", "us"],
        help="マーケット絞り込み（省略=全マーケット）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        nargs="+",
        default=[5],
        help="同時保有銘柄数（複数指定でスイープ実行、例: --top-n 3 5 10）",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=str,
        nargs="+",
        default=["weekly"],
        choices=["daily", "weekly", "monthly"],
        help="リバランス頻度（複数指定でスイープ実行）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="学習データ比率",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="file",
        choices=["file", "raw"],
        help="データソース",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000,
        help="初期資金",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="片道手数料率",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="シグナル発生の最低予測上昇率（0.0=制限なし）",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="XGBoostModel",
        choices=["XGBoostModel", "LightGBMModel"],
        help="モデルタイプ",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="XGBoost+LightGBM アンサンブル予測を使用する",
    )
    parser.add_argument(
        "--save-chart",
        action="store_true",
        help="バックテスト結果グラフ (PNG) を results/ に保存する",
    )
    parser.add_argument(
        "--max-sector-positions",
        type=int,
        default=MAX_SECTOR_POSITIONS,
        help="同一セクターで許容する最大銘柄数。0 で制約無効",
    )

    return parser.parse_args()


def _run_one(args, top_n: int, rebalance_freq: str) -> None:
    """top_n × rebalance_freq の 1 組合せを実行する。"""
    market_label = args.market or "ALL"
    logger.info(
        f"[PF] 開始: market={market_label} top_n={top_n} freq={rebalance_freq} "
        f"model={args.model_type} ensemble={args.ensemble}"
    )

    equity_df, metrics, holdings_df = run_portfolio_backtest(
        market=args.market,
        model_type=args.model_type,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        train_ratio=args.train_ratio,
        source=args.source,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        threshold=args.threshold,
        ensemble=args.ensemble,
        max_sector_positions=args.max_sector_positions,
    )

    if equity_df.empty:
        logger.warning(f"[PF] 結果が空です: top_n={top_n} freq={rebalance_freq}")
        return

    print_portfolio_metrics(
        metrics, market=market_label, top_n=top_n, rebalance_freq=rebalance_freq
    )

    save_portfolio_results(
        equity_df,
        metrics,
        holdings_df,
        market=market_label,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
    )

    if args.save_chart:
        chart_path = plot_portfolio(
            equity_df=equity_df,
            holdings_df=holdings_df,
            metrics=metrics,
            market=market_label,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
        )
        if chart_path:
            logger.info(f"[PF] グラフ保存: {chart_path}")


def main():
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()

    combos = [(n, f) for n in args.top_n for f in args.rebalance_freq]

    if len(combos) > 1:
        logger.info(f"[PF] スイープ実行: {len(combos)} 組合せ")

    for top_n, freq in combos:
        _run_one(args, top_n, freq)

    logger.info("[PF] 完了")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"ポートフォリオバックテスト 異常終了: {e}", exc_info=True)
        sys.exit(1)
