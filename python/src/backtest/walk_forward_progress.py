"""
週の Walk-Forward レポートの途中の結果（#1931）

全銘柄に約6時間かかり、デプロイの起き直しで回が消える。銘柄ごとの結果の行を、週の鍵が
名前で分かる途中のファイルに1行ずつ書き足し、同じ週の補完の回では済んだ銘柄を飛ばす。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger

logger = get_logger(__name__)

# スケジューラの週の鍵（scheduler_queue の _build_period_key）と同じ暦で数える
_JST = ZoneInfo("Asia/Tokyo")
_PARAMS_KEY = "_params"


def week_key_at(moment: Optional[datetime] = None) -> str:
    """その時点の ISO 週の鍵（例: 2026-W46）を JST で返す。"""
    now = moment.astimezone(_JST) if moment is not None else datetime.now(_JST)
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def partial_path(report_dir: Path, week_key: str) -> Path:
    return report_dir / f"wf_partial_{week_key}.jsonl"


def remove_other_weeks(report_dir: Path, week_key: str) -> None:
    """ほかの週の途中のファイルは続きに使わないので消す。"""
    keep = partial_path(report_dir, week_key).name
    for path in report_dir.glob("wf_partial_*.jsonl"):
        if path.name != keep:
            logger.info(f"[WF] 別の週の途中のファイルを消す: {path.name}")
            path.unlink(missing_ok=True)


def load_partial_rows(path: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
    """途中のファイルから済んだ行を読む。条件が違う・無いときは空を返し、頭から回す。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    header: Optional[dict[str, Any]] = None
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                # 消えた回が書きかけた最後の行は読まずに飛ばす
                logger.warning(f"[WF] 途中のファイルの壊れた行を飛ばす: {path.name}:{line_no}")
                continue
            if not isinstance(record, dict):
                continue
            if _PARAMS_KEY in record:
                header = record[_PARAMS_KEY]
                continue
            rows.append(record)
    if header != params:
        logger.warning(f"[WF] 途中のファイルの条件が違うので使わず頭から回す: {path.name}")
        return []
    return rows


def start_partial(path: Path, params: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """途中のファイルを、条件の行と済んだ行で書き直す。"""
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({_PARAMS_KEY: params}, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def append_partial_row(path: Path, row: dict[str, Any]) -> None:
    """1銘柄の結果の行を書き足す。"""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        fh.flush()


def done_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(r.get("market")), str(r.get("symbol"))) for r in rows}
