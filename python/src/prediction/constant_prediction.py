"""全銘柄がほぼ同じ値を出すモデルの見分けと、統合モデルの学習の終わりの確かめ（#1602）。

学習のログに best_iteration・木の数・検証の行の予想の標準偏差を書き、標準偏差が線より
小さいモデルが保存された回は1回知らせる。確かめの失敗で学習を止めない。

線の根拠: 本番の予想の標準偏差は 0.016〜0.028、定数の challenger は 6e-17 である。
1e-4 は本番の最小の 160 分の 1 で、定数の 1e12 倍あり、どちらの桁からも十分に離れている。
夜間の入れ替えの「道の一致」も、diff_ratio の差が 1e-4 以下なら同じ予想と見なす。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.domain.ports import AlertLevel
from src.prediction import ports
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONSTANT_PREDICTION_STD = 1e-4


def is_constant_prediction(std: Optional[float]) -> bool:
    """銘柄の間の予想の標準偏差が線より小さければ True。分からない（None）なら False。"""
    return std is not None and std < CONSTANT_PREDICTION_STD


def _best_iteration(estimator: Any) -> Optional[int]:
    """XGBoost は best_iteration、LightGBM は best_iteration_。取れない型は None を返す。"""
    for name in ("best_iteration", "best_iteration_"):
        value = getattr(estimator, name, None)
        if value is not None:
            return int(value)
    return None


def _tree_count(estimator: Any) -> Optional[int]:
    """使える木の数を返す。木のモデルでなければ None。"""
    get_booster = getattr(estimator, "get_booster", None)
    if get_booster is not None:
        return int(get_booster().num_boosted_rounds())
    booster = getattr(estimator, "booster_", None)
    return None if booster is None else int(booster.num_trees())


def _notify_constant_model(model_name: str, detail: str) -> None:
    """定数のモデルが保存されたことを1回知らせる。知らせの失敗で学習を止めない。"""
    logger.warning("統合モデルの予想が全行でほぼ同じ値: model=%s %s", model_name, detail)
    notifier = ports.get_notification_port()
    if notifier is None:
        return
    try:
        notifier.send_alert(
            f"統合モデルの予想がほぼ定数: {model_name}",
            f"{detail}\n学習は止めていない。比べと昇格ではこのモデルを勝ちにしない。",
            level=AlertLevel.WARNING,
        )
    except Exception:
        logger.error("定数のモデルの知らせに失敗: model=%s", model_name, exc_info=True)


def check_unified_training(model: Any, eval_set: Optional[list]) -> None:
    """学習を終えたモデルの数字をログに書き、定数のモデルなら知らせる。

    検証の行が無い回（学習が 1000 行に満たない回）は val_pred_std を None と書き、知らせない。

    Args:
        model: 学習済みの BaseModel
        eval_set: 学習に渡した [(X_val, y_val)]。無ければ None
    """
    try:
        estimator = model.model
        best_iteration = _best_iteration(estimator)
        trees = _tree_count(estimator)
        val_rows = 0
        val_std: Optional[float] = None
        if eval_set:
            X_val = eval_set[0][0]
            val_rows = len(X_val)
            val_std = float(np.std(np.asarray(model.predict(X_val), dtype=float)))
    except Exception:
        logger.error("統合モデルの学習の確かめに失敗: model=%s", model.model_name, exc_info=True)
        return

    detail = (
        f"best_iteration={best_iteration} trees={trees} val_rows={val_rows} val_pred_std={val_std}"
    )
    logger.info("統合モデルの学習の確かめ: model=%s %s", model.model_name, detail)
    if is_constant_prediction(val_std):
        _notify_constant_model(model.model_name, detail)
