"""日次ドリフトチェックのその日の判定と再学習の数を results/drift_check に残す（#2050）。

今までは判定の数・割合・線・再学習の数がログの行にだけあり、ログが回ると読み返せなかった。
ここでは1日1件の JSON（ファイル名は JST の日付）を書くだけで、判定・再学習・知らせには効かない。
同じ日のやり直しは上書きする。書けなくても例外は外に出さず、logger.error に残す。
読み返す側は src/reporting/drift_check_trend.py。
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

from src.utils.data_path_utils import get_results_dir
from src.utils.japan_time import isoformat_utc, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

DRIFT_RECORD_SUBDIR = "drift_check"


@dataclass
class DriftCheckRecord:
    """日次ドリフトチェックの1日分の判定と再学習の数。"""

    n_evaluated: int
    n_triggered: int
    triggered_share: float
    warn_share: float
    warn: bool
    mae_threshold: float
    hit_rate_threshold: float
    fresh_skipped: int = 0
    retrain_targets: int = 0
    retrain_success: int = 0


def get_drift_record_dir() -> str:
    """記録の置き場（results/drift_check）を返す。"""
    return os.path.join(get_results_dir(), DRIFT_RECORD_SUBDIR)


def write_drift_check_record(
    record: DriftCheckRecord, out_dir: str | None = None, now: datetime | None = None
) -> str | None:
    """記録を <out_dir>/<JST の日付>.json に書き、書いたパスを返す（失敗したら None）。"""
    try:
        stamp = now if now is not None else now_jst()
        target_dir = out_dir if out_dir is not None else get_drift_record_dir()
        os.makedirs(target_dir, exist_ok=True)
        date_key = stamp.strftime("%Y-%m-%d")
        payload = {"date": date_key, "recorded_at": isoformat_utc(), **asdict(record)}
        path = os.path.join(target_dir, f"{date_key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        logger.error("ドリフトチェックの日次の記録を書けず: %s", e, exc_info=True)
        return None
