"""統合モデルの学習の確かめの数字を results/unified_training に残す（#2131）。

今までは数字がログの1行にだけあり、ログが回ると読み返せなかった。
ここでは1回の学習ごとに1件の JSON を書くだけで、学習・保存・知らせには効かない。
同じ日（JST）の同じモデルの学び直しは上書きする。
書けなくても例外は外に出さず、logger.error に残す。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from src.utils.data_path_utils import get_results_dir
from src.utils.japan_time import isoformat_utc, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAINING_RECORD_SUBDIR = "unified_training"


@dataclass
class UnifiedTrainingRecord:
    """1回の学習の確かめの数字。検証の行から出す値は、検証の行が無ければ None。"""

    model_name: str
    best_iteration: Optional[int]
    trees: Optional[int]
    val_rows: int
    val_pred_std: Optional[float]
    is_constant: Optional[bool]


def get_training_record_dir() -> str:
    """記録の置き場（results/unified_training）を返す。"""
    return os.path.join(get_results_dir(), TRAINING_RECORD_SUBDIR)


def _up_rate(values: Any) -> Optional[float]:
    """0 より大きい値の割合を返す。値が無ければ None。"""
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.mean(array > 0)) if array.size else None


def write_training_record(record: UnifiedTrainingRecord, val_pred: Any, val_actual: Any) -> None:
    """記録を <JST の日付>_<model_name>.json に書く。

    Args:
        record: 学習の確かめの数字
        val_pred: 検証の行の予想。検証の行が無ければ None
        val_actual: 検証の行の実際の値。検証の行が無ければ None
    """
    try:
        stamp = now_jst()
        payload = {
            **asdict(record),
            "recorded_at": isoformat_utc(stamp),
            "val_predicted_up_rate": _up_rate(val_pred),
            "val_actual_up_rate": _up_rate(val_actual),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        target_dir = get_training_record_dir()
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"{stamp.strftime('%Y-%m-%d')}_{record.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        logger.error("統合モデルの学習の記録を書けず: model=%s", record.model_name, exc_info=True)
