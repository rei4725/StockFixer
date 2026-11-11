"""買い候補の選別と保有銘柄・エグジットシグナルの判定。"""

import os
from collections import Counter
from typing import Iterable

import pandas as pd

from config.settings import MAX_ORDERS_PER_RUN, MAX_SECTOR_POSITIONS
from src.domain.sector_cap import filter_by_sector_cap
from src.domain.trading_rules import ML_EXIT_PROB_THRESHOLD as _ML_EXIT_PROB_THRESHOLD
from src.trading.brokers.base import BrokerBase
from src.trading.models.exit_model import ExitModel
from src.utils.logger import get_logger
from src.utils.sector_constraints import get_symbol_sector

from .stats import OrderExecutionStats, count_skip

logger = get_logger(__name__)


def _count_dropped(stats: OrderExecutionStats, n_dropped: int, reason: str, label: str) -> None:
    """落とした候補を reason の鍵で skipped と skip_reasons に数え、件数をログに出す（#1770）。"""
    if n_dropped <= 0:
        return
    logger.info("[exec] %s で候補 %d 件を落とした（%s）", label, n_dropped, reason)
    for _ in range(n_dropped):
        count_skip(stats, reason)


def _cap_per_run(
    candidates: pd.DataFrame, stats: OrderExecutionStats, allowed: bool = True
) -> pd.DataFrame:
    """1回の発注の上限（MAX_ORDERS_PER_RUN）で先頭から切り、落とした候補を数える（#1770）。

    鍵は 16:30 の RuleGateRecord とそろえて max_orders_per_run。allowed が偽（リスクのゲートで
    新規を止めた回）は 0 件を返し、上限で落としたとは数えない。
    """
    if not allowed:
        return candidates.head(0)
    kept = candidates.head(MAX_ORDERS_PER_RUN)
    _count_dropped(stats, len(candidates) - len(kept), "max_orders_per_run", "1回の発注の上限")
    return kept


def _select_buy_signals(
    candidates: pd.DataFrame,
    held_symbols: Iterable[str],
    market: str,
    stats: OrderExecutionStats,
) -> pd.DataFrame:
    """新規買いの候補に業種の上限と1回の発注の上限を当てる。選び方と順は変えない（#1770）。

    業種の上限で落とした候補は sector_cap、head で落とした候補は max_orders_per_run に数える
    （鍵は 16:30 の RuleGateRecord とそろえる）。
    """
    limited = _apply_buy_sector_limit(candidates, held_symbols=held_symbols, market=market)
    _count_dropped(stats, len(candidates) - len(limited), "sector_cap", "業種の上限")
    return _cap_per_run(limited, stats)


def _apply_buy_sector_limit(
    predictions: pd.DataFrame,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
    held_symbols: Iterable[str] = (),
    market: str = "jp",
) -> pd.DataFrame:
    """買い候補に同一セクター上限を適用する。

    held_symbols を渡すと、保有中の銘柄の業種の数と候補の数の和で上限を判定する（#908）。
    """
    if predictions.empty or max_sector_positions <= 0:
        return predictions.copy()

    held_sector_counts = Counter(get_symbol_sector(market, s) for s in sorted(held_symbols))

    score_col = (
        "multi_horizon_score" if "multi_horizon_score" in predictions.columns else "diff_ratio"
    )
    ordered_rows = list(predictions.sort_values(score_col, ascending=False).iterrows())
    sector_cache: dict[int, str] = {}

    def _sector_getter(item: tuple[int, pd.Series]) -> str:
        idx, row = item
        if idx not in sector_cache:
            sector_cache[idx] = get_symbol_sector(str(row["market"]), str(row["symbol"]))
        return sector_cache[idx]

    selected_rows = filter_by_sector_cap(
        ordered_rows,
        max_sector_positions=max_sector_positions,
        sector_getter=_sector_getter,
        initial_counts=held_sector_counts,
    )
    if not selected_rows:
        return predictions.iloc[0:0].copy()

    selected_index = [idx for idx, _ in selected_rows]
    limited = predictions.loc[selected_index].copy()
    limited["sector"] = [sector_cache[idx] for idx, _ in selected_rows]
    return limited


def _get_held_symbols(broker: BrokerBase) -> set[str]:
    """保有中の銘柄コードセットを返す"""
    positions = broker.get_positions()
    return {p["symbol"].replace(".T", "") for p in positions if p.get("qty", 0) > 0}


def _get_pending_buy_symbols(broker: BrokerBase) -> set[str]:
    """約定待ちの BUY の銘柄コードを返す。読む関数が無い Broker は空（#1137）。

    読めない日を0件と取り違えると上限が甘くなるため、例外は上へ出す（#1465）。
    """
    reader = getattr(broker, "get_pending_buy_symbols", None)
    if reader is None:
        return set()
    try:
        return {str(s).replace(".T", "") for s in reader()}
    except Exception:
        logger.error("[exec] 約定待ちの BUY を読めない（新規は止める）", exc_info=True)
        raise


def _get_rule_buy_symbols_placed_today(broker: BrokerBase) -> set[str]:
    """今日のルールの回が出した約定待ちの BUY の銘柄コードを返す。読む関数が無い Broker は空（#1660）

    読めない日の扱いは _get_pending_buy_symbols と同じで、例外は上へ出す。
    """
    reader = getattr(broker, "get_rule_buy_symbols_placed_today", None)
    if reader is None:
        return set()
    try:
        return {str(s).replace(".T", "") for s in reader()}
    except Exception:
        logger.error("[exec] 今日のルールの回の BUY を読めない（新規は止める）", exc_info=True)
        raise


def _load_exit_model(market: str) -> ExitModel | None:
    """
    保存済みの ExitModel を models/ から読み込む。

    モデルファイルが存在しない場合は None を返し、固定閾値ロジックにフォールバックする。
    モデルパス: {models_dir}/{market}_ExitModel.joblib
    """
    from src.utils.data_path_utils import get_models_dir

    model_path = os.path.join(get_models_dir(), f"{market}_ExitModel.joblib")
    if not os.path.exists(model_path):
        logger.debug("[exit_model] 学習済み ExitModel が見つかりません: %s", model_path)
        return None
    try:
        exit_model = ExitModel(model_name="ExitModel")
        exit_model.load_model(model_path)
        logger.info("[exit_model] ExitModel をロードしました: %s", model_path)
        return exit_model
    except Exception:
        logger.error("[exit_model] ExitModel のロードに失敗しました: %s", model_path, exc_info=True)
        return None


def _compute_ml_exit_signals(
    held_predictions: pd.DataFrame,
    exit_model: ExitModel,
    threshold: float = _ML_EXIT_PROB_THRESHOLD,
) -> set[str]:
    """
    ML モデルで保有銘柄のエグジット確率を計算し、閾値超の銘柄セットを返す。

    Args:
        held_predictions: 保有中銘柄の予測 DataFrame（symbol 列を含む）
        exit_model: 学習済み ExitModel
        threshold: エグジット判定の確率閾値

    Returns:
        ML エグジットシグナルが発火した銘柄コードのセット
    """
    if held_predictions.empty:
        return set()
    try:
        X = ExitModel.prepare_features(held_predictions)
        proba = exit_model.predict(X)
        exit_mask = proba >= threshold
        triggered = set(held_predictions.loc[exit_mask.values, "symbol"].astype(str))
        if triggered:
            logger.info(
                "[exit_model] ML エグジットシグナル発火: %s (threshold=%.2f)",
                sorted(triggered),
                threshold,
            )
        return triggered
    except Exception:
        logger.error("[exit_model] ML エグジット確率計算でエラー", exc_info=True)
        return set()


def _determine_entry_horizon(row: pd.Series) -> int:
    """予測行の中で最大絶対変化率を持つホライズン（日数）を返す。"""
    candidates: dict[int, float] = {1: abs(float(row.get("diff_ratio") or 0.0))}
    for h, col in [(3, "diff_ratio_3d"), (5, "diff_ratio_5d"), (10, "diff_ratio_10d")]:
        if col in row.index:
            v = row[col]
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                try:
                    candidates[h] = abs(float(v))
                except (TypeError, ValueError):
                    pass
    return max(candidates, key=lambda k: candidates[k])
