"""
銘柄別モデルの昇格フロー（手動承認）

shadow_evaluation.py から移した（#1866）。評価（evaluate_shadow_models）と
統合モデルの影の予想・昇格は shadow_evaluation.py に残る。
"""

from __future__ import annotations

import os
import shutil

from src.utils.data_path_utils import get_models_subdir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# モデル昇格フロー（銘柄別）
# ---------------------------------------------------------------------------

_PRODUCTION_MODEL_NAMES = ["StockXGBoostModel", "StockLightGBMModel"]
_CHALLENGER_MODEL_NAMES = ["ChallengerXGBoostModel", "ChallengerLightGBMModel"]


def promote_challenger_to_production(
    market: str,
    symbol: str,
    dry_run: bool = False,
) -> dict:
    """
    チャレンジャーモデルを本番モデルとして昇格させる（手動承認フロー）。

    ChallengerXGBoostModel / ChallengerLightGBMModel の joblib ファイルを
    StockXGBoostModel / StockLightGBMModel として上書きコピーする。
    evaluate_shadow_models() で challenger_wins=True を確認した後に呼び出すこと。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        dry_run: True の場合はファイル操作を行わずログのみ出力する

    Returns:
        dict:
            promoted: list[str] — 昇格に成功したモデル名のリスト
            skipped: list[str] — チャレンジャーファイルが存在しなかったモデル名
            dry_run: bool
    """
    model_dir = get_models_subdir(market, symbol)
    promoted = []
    skipped = []

    for challenger_name, production_name in zip(_CHALLENGER_MODEL_NAMES, _PRODUCTION_MODEL_NAMES):
        src = os.path.join(model_dir, f"{challenger_name}.joblib")
        dst = os.path.join(model_dir, f"{production_name}.joblib")

        if not os.path.exists(src):
            logger.warning(f"チャレンジャーモデルが見つかりません: {src}")
            skipped.append(challenger_name)
            continue

        if dry_run:
            logger.info(f"[dry_run] 昇格スキップ: {challenger_name} -> {production_name}")
        else:
            shutil.copy2(src, dst)
            logger.info(
                f"モデル昇格完了: {challenger_name} -> {production_name} ({market}/{symbol})"
            )

        promoted.append(production_name)

    return {"promoted": promoted, "skipped": skipped, "dry_run": dry_run}
