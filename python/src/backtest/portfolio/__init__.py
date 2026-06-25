"""
ポートフォリオバックテストサービス

全銘柄を対象に日次シグナルマトリクスを構築し、Top-N 銘柄へ
予測スコア比例配分（softmax）で資金を投じるポートフォリオシミュレーションを行う。

Usage (from run_backtest_portfolio.py):
    results = run_portfolio_backtest(
        market="jp",
        top_n=5,
        rebalance_freq="weekly",
        train_ratio=0.8,
        source="file",
        initial_cash=1_000_000,
        fee_rate=0.001,
    )

Issue #511: 肥大化した portfolio.py を責務別モジュールへ分割し、
本 __init__ で re-export ファサードとして public/internal API を再公開する。
`from src.backtest.portfolio import X` の後方互換を維持する。
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from config.settings import MAX_SECTOR_POSITIONS
from src.utils.logger import get_logger

from .metrics import (  # noqa: F401
    _attach_regime_metrics,
    _compute_portfolio_metrics,
    _compute_regime_leg_metrics,
    _compute_regime_metrics,
)
from .reporting import plot_portfolio, print_portfolio_metrics, save_portfolio_results  # noqa: F401
from .signal_matrix import _build_signal_matrix  # noqa: F401
from .simulation import (  # noqa: F401
    _apply_sector_rotation,
    _build_market_proxy_frame,
    _get_portfolio_symbol_sector,
    _get_rebalance_dates,
    _limit_portfolio_candidates_by_sector,
    _simulate_portfolio,
    _softmax_weights,
    _split_symbol_key,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────
# Public API（オーケストレーター）
# ─────────────────────────────────────────


def run_portfolio_backtest(
    market: Optional[str] = None,
    model_type: str = "XGBoostModel",
    top_n: int = 5,
    rebalance_freq: str = "weekly",
    train_ratio: float = 0.8,
    source: str = "file",
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    threshold: float = 0.0,
    ensemble: bool = False,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
    use_sector_rotation: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    ポートフォリオバックテストを実行する。

    Args:
        market: "jp" / "us" / None（全マーケット）
        model_type: "XGBoostModel" or "LightGBMModel"
        top_n: 同時保有する最大銘柄数
        rebalance_freq: "daily" / "weekly" / "monthly"
        train_ratio: 学習データ比率
        source: データソース ("file" / "raw")
        initial_cash: 初期資金
        fee_rate: 片道手数料率
        threshold: 買いシグナル発生の最低予測上昇率（0.0=制限なし）
        ensemble: XGBoost+LightGBM アンサンブルを使用
        max_sector_positions: 同一セクターで許容する最大銘柄数（0 以下で無効）
        use_sector_rotation: True のとき市場レジームに応じてセクターウェイトを切り替える

    Returns:
        (equity_df, metrics, holdings_df)
        - equity_df: 日次ポートフォリオ価値の DataFrame (columns: date, portfolio_value, equal_weight_value)
        - metrics: パフォーマンス指標辞書
        - holdings_df: リバランス日ごとの保有銘柄・ウェイト一覧
    """
    from src.utils.db.stock_features import get_all_symbols

    # 対象銘柄リスト取得
    all_symbols = get_all_symbols()
    if market:
        all_symbols = [(m, s) for m, s in all_symbols if m == market]

    if not all_symbols:
        logger.error(f"対象銘柄が見つかりません: market={market}")
        return pd.DataFrame(), {}, pd.DataFrame()

    logger.info(f"[ポートフォリオ] 対象銘柄数: {len(all_symbols)}")

    # 各銘柄の予測スコアと Close 価格を収集
    score_matrix, close_matrix = _build_signal_matrix(
        all_symbols, model_type, train_ratio, source, threshold, ensemble
    )

    if score_matrix.empty:
        logger.error("[ポートフォリオ] スコアマトリクスが空です")
        return pd.DataFrame(), {}, pd.DataFrame()

    # リバランス日を決定
    rebalance_dates = _get_rebalance_dates(pd.DatetimeIndex(score_matrix.index), rebalance_freq)

    # ポートフォリオシミュレーション
    equity_df, holdings_records = _simulate_portfolio(
        score_matrix=score_matrix,
        close_matrix=close_matrix,
        rebalance_dates=rebalance_dates,
        top_n=top_n,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        max_sector_positions=max_sector_positions,
        use_sector_rotation=use_sector_rotation,
    )

    metrics = _compute_portfolio_metrics(equity_df, initial_cash)
    equity_df, regime_metrics = _attach_regime_metrics(equity_df, close_matrix)
    if regime_metrics:
        metrics["regime_metrics"] = regime_metrics
    metrics["max_sector_positions"] = max_sector_positions

    holdings_df = pd.DataFrame(holdings_records)

    return equity_df, metrics, holdings_df


def compare_sector_rotation_kpi(
    market: Optional[str] = None,
    model_type: str = "XGBoostModel",
    top_n: int = 5,
    rebalance_freq: str = "weekly",
    train_ratio: float = 0.8,
    source: str = "file",
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    ensemble: bool = False,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
) -> dict[str, Any]:
    """
    セクターローテーション有効時/無効時の KPI を比較する。

    シグナルマトリクスを一度だけ構築し、両シミュレーションに再利用する。

    Returns:
        {"rotation_off": metrics, "rotation_on": metrics, "kpi_diff": diff}
        対象銘柄が存在しない場合やデータ不足の場合は空辞書
    """
    from src.utils.db.stock_features import get_all_symbols

    all_symbols = get_all_symbols()
    if market:
        all_symbols = [(m, s) for m, s in all_symbols if m == market]
    if not all_symbols:
        logger.error(f"[セクターローテーション比較] 対象銘柄が見つかりません: market={market}")
        return {}

    score_matrix, close_matrix = _build_signal_matrix(
        all_symbols, model_type, train_ratio, source, 0.0, ensemble
    )
    if score_matrix.empty:
        logger.error("[セクターローテーション比較] スコアマトリクスが空です")
        return {}

    rebalance_dates = _get_rebalance_dates(pd.DatetimeIndex(score_matrix.index), rebalance_freq)

    eq_off, _ = _simulate_portfolio(
        score_matrix,
        close_matrix,
        rebalance_dates,
        top_n,
        initial_cash,
        fee_rate,
        max_sector_positions,
        use_sector_rotation=False,
    )
    eq_on, _ = _simulate_portfolio(
        score_matrix,
        close_matrix,
        rebalance_dates,
        top_n,
        initial_cash,
        fee_rate,
        max_sector_positions,
        use_sector_rotation=True,
    )

    metrics_off = _compute_portfolio_metrics(eq_off, initial_cash)
    metrics_on = _compute_portfolio_metrics(eq_on, initial_cash)

    kpi_diff = {
        "total_return_diff": round(
            metrics_on.get("total_return", 0.0) - metrics_off.get("total_return", 0.0), 6
        ),
        "sharpe_diff": round(
            metrics_on.get("sharpe_ratio", 0.0) - metrics_off.get("sharpe_ratio", 0.0), 4
        ),
        "max_drawdown_diff": round(
            metrics_on.get("max_drawdown", 0.0) - metrics_off.get("max_drawdown", 0.0), 6
        ),
    }

    return {
        "rotation_off": metrics_off,
        "rotation_on": metrics_on,
        "kpi_diff": kpi_diff,
    }


__all__ = [
    "run_portfolio_backtest",
    "compare_sector_rotation_kpi",
    "save_portfolio_results",
    "plot_portfolio",
    "print_portfolio_metrics",
    "_build_signal_matrix",
    "_get_rebalance_dates",
    "_softmax_weights",
    "_apply_sector_rotation",
    "_simulate_portfolio",
    "_get_portfolio_symbol_sector",
    "_split_symbol_key",
    "_limit_portfolio_candidates_by_sector",
    "_build_market_proxy_frame",
    "_compute_portfolio_metrics",
    "_attach_regime_metrics",
    "_compute_regime_metrics",
    "_compute_regime_leg_metrics",
]
