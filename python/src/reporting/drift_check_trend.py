"""results/drift_check の日次ドリフトチェックの記録を読み返し、日ごとに1行の表を作る（#2050）。

書く側（orchestration/jobs/drift_record.py）は1日1件の JSON で、閾値超過の割合や
再学習の数の流れを見るにはファイルを1つずつ開くしかなかった。ここでは読むだけで、
日ごとに1行の表（DataFrame）にする。本番のジョブには組み込まない。

- 名前が YYYY-MM-DD.json でないファイル、読めないファイル、壊れた JSON、
  オブジェクトでない JSON は警告を出して飛ばす
- 無い項目は空（None）にする
"""

import glob
import json
import os
import re

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TREND_COLUMNS: list[str] = [
    "date",
    "n_evaluated",
    "n_triggered",
    "triggered_share",
    "warn_share",
    "warn",
    "mae_threshold",
    "hit_rate_threshold",
    "fresh_skipped",
    "retrain_targets",
    "retrain_success",
]

_DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def load_drift_check_records(record_dir: str) -> list[dict]:
    """record_dir の YYYY-MM-DD.json を日の順に読む（読めないものは飛ばす）。"""
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(record_dir, "*.json"))):
        match = _DATE_PATTERN.match(os.path.basename(path))
        if match is None:
            logger.warning("ドリフトチェックの記録の名前の形が違うため飛ばす: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning("ドリフトチェックの記録を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("ドリフトチェックの記録の形が違うため飛ばす: %s", path)
            continue
        row = {column: data.get(column) for column in TREND_COLUMNS}
        row["date"] = match.group(1)
        rows.append(row)
    return rows


def build_drift_check_trend(rows: list[dict]) -> pd.DataFrame:
    """日ごとの行を日の順の表にする。"""
    df = pd.DataFrame(rows, columns=TREND_COLUMNS)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_drift_check_trend(record_dir: str) -> pd.DataFrame:
    """record_dir の記録を読み、日ごとの表を返す。"""
    return build_drift_check_trend(load_drift_check_records(record_dir))
