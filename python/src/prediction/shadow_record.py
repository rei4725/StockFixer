"""週の学習の影の比べと昇格ゲートの結果を、ISO 週の名前の JSON に書く（#1503）。

置き場: results/accuracy/shadow/<ISO 年>-W<週>.json（同じ週のやり直しは上書きする）。
challenger の勝ち負けと昇格を飛ばした理由に関わらず、毎週1本書く。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SHADOW_KEYS = (
    "challenger_trained_at",
    "production_changed_at",
    "window_days",
    "production_hit_rate",
    "production_sharpe",
    "challenger_hit_rate",
    "challenger_sharpe",
    "n_production",
    "n_challenger",
    "challenger_wins",
)


def new_weekly_shadow_record(days: int) -> dict[str, Any]:
    """欄をすべて持つ空の記録を返す。days は影の比べの読む日数。"""
    record: dict[str, Any] = {
        "shadow_evaluated": False,
        "shadow_failed": False,
        "shadow_error": None,
        "days": days,
        "fresh_challengers": [],
        "gate_evaluated": False,
        "gate_skipped_reason": None,
        "gate_failed": False,
        "eligible": None,
        "reason": None,
        "promoted": False,
    }
    for key in _SHADOW_KEYS:
        record[key] = None
    return record


def apply_shadow_result(record: dict[str, Any], result: dict) -> None:
    """evaluate_shadow_models() の結果を記録に写す。"""
    record["shadow_evaluated"] = True
    for key in _SHADOW_KEYS:
        record[key] = result.get(key)


def weekly_shadow_record_path(now: datetime, results_dir: Optional[str] = None) -> Path:
    """now の ISO 週の記録のパスを返す。"""
    year, week, _ = now.isocalendar()
    base = Path(results_dir if results_dir is not None else get_results_dir())
    return base / "accuracy" / "shadow" / f"{year}-W{week:02d}.json"


def save_weekly_shadow_record(
    record: dict[str, Any],
    now: Optional[datetime] = None,
    results_dir: Optional[str] = None,
) -> str:
    """記録を ISO 週の名前の JSON に書き、ファイルのパスを返す。"""
    now = now or datetime.now(timezone.utc)
    dest = weekly_shadow_record_path(now, results_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)

    year, week, _ = now.isocalendar()
    payload = {"week": f"{year}-W{week:02d}", "written_at": now.isoformat(), **record}
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("影の比べの記録を保存: %s", dest)
    return str(dest)
