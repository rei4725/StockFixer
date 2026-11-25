"""日次ドリフトチェックのその日の判定と再学習の数を results/drift_check に残す（#2050）。

今までは判定の数・割合・線・再学習の数がログの行にだけあり、ログが回ると読み返せなかった。
ここでは1日1件の JSON（ファイル名は JST の日付）を書くだけで、判定・再学習・知らせには効かない。
同じ日のやり直しは上書きする。書けなくても例外は外に出さず、logger.error に残す。
読み返す側は src/reporting/drift_check_trend.py。
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from src.utils.data_path_utils import get_results_dir
from src.utils.japan_time import isoformat_utc, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

DRIFT_RECORD_SUBDIR = "drift_check"

# 閾値を超えた銘柄のその日の扱い（#2167）
FRESH_SKIPPED = "fresh_skipped"  # 直近に学び直したため外した
RETRAINED = "retrained"
RETRAIN_FAILED = "retrain_failed"  # 例外・success でない・タスクなし
OVER_CAP = "over_cap"  # 再学習の上限の外


@dataclass
class DriftTriggeredSymbol:
    """閾値を超えた1銘柄の値・超えた線・扱い。"""

    market: str
    symbol: str
    mean_abs_error: float | None
    direction_accuracy: float | None
    mae_over: bool
    hit_rate_over: bool
    disposition: str


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
    # 超えた銘柄の一覧（超過0の日は空、組めなかった日は None）と再学習の所要秒
    # （再学習の無い日は None）（#2167）
    triggered_symbols: list[DriftTriggeredSymbol] | None = None
    retrain_seconds: float | None = None


def _number_or_none(value: Any) -> float | None:
    """欠け（None・NaN）は None、それ以外は float にする。"""
    return None if value is None or math.isnan(value) else float(value)


def build_triggered_symbols(
    ranked: list[dict],
    fresh_skipped: list[dict],
    targets: list[dict],
    retrained: list[dict],
    mae_threshold: float,
    hit_rate_threshold: float,
) -> list[DriftTriggeredSymbol] | None:
    """閾値を超えた銘柄の一覧を、ジョブが持つ値から組む（組めなければ None。例外は出さない）。

    超えた線の式は reporting/drift_thresholds.py の select_drifted_symbols と同じ（等号を含む）。
    """
    try:
        dispositions: dict[tuple[str, str], str] = {}
        # 後から書くものが勝つ（対象のうち成功したものだけ retrained になる）
        for symbols, disposition in (
            (fresh_skipped, FRESH_SKIPPED),
            (targets, RETRAIN_FAILED),
            (retrained, RETRAINED),
        ):
            for s in symbols:
                dispositions[(s["market"], s["symbol"])] = disposition
        return [
            DriftTriggeredSymbol(
                market=str(s["market"]),
                symbol=str(s["symbol"]),
                mean_abs_error=_number_or_none(s["mean_abs_error"]),
                direction_accuracy=_number_or_none(s["direction_accuracy"]),
                mae_over=bool(s["mean_abs_error"] >= mae_threshold),
                hit_rate_over=bool(s["direction_accuracy"] <= hit_rate_threshold),
                disposition=dispositions.get((s["market"], s["symbol"]), OVER_CAP),
            )
            for s in ranked
        ]
    except Exception as e:
        logger.error("ドリフトチェックの超えた銘柄の一覧を組めず: %s", e, exc_info=True)
        return None


def read_timer() -> float | None:
    """所要秒を測る時計を読む（読めなければ None。再学習を止めない）。"""
    try:
        return time.monotonic()
    except Exception as e:
        logger.error("ドリフト再学習の所要秒を測れず: %s", e, exc_info=True)
        return None


def elapsed_seconds(start: float | None, end: float | None) -> float | None:
    """2つの時計の読みの差の秒（どちらかが無ければ None）。"""
    return None if start is None or end is None else end - start


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
