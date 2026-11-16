"""results/scheduler_queue_state.json のジョブの実行の記録を読み返し、回ごとの表を作る（#1892）。

書く側（orchestration/scheduler_queue.py）は events に、始める前の started、終わりの
success か error、やり直しの retry、回数の上限で飛ばした skipped_limit を足していく。
ここでは読むだけで、ジョブ・期間（period_key）・run_id ごとに1行の表を作る。
スケジューラーには組み込まない。

- started と、同じ run_id の終わりの記録（success・error など）を1行にまとめる
- retry には run_id が無いので、同じ job_id・period_key でいま開いている回（started の後に
  終わりの記録がまだ無い回）に数える。開いた回が無い retry（run_id の無い古い記録の頃）は、
  同じ job_id・period_key の次の run_id の無い終わりの記録に数える
- started の後に同じ run_id の終わりの記録が無い回は、status を欠け（pd.NA）にし、
  missing_finish を True にする（プロセスごと消えた回。いま走っている回もこの印になる）
- run_id の無い古い記録（started を書かず、終わりの記録だけの頃）と skipped_limit は、
  その記録1件を1行にする
- 終わりの記録はあるが started が無い回（events が最新の5000件に切られた後）も1行にする
- 読めないファイル・形の違う JSON は警告を出して空の表を返す
"""

import json
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

STARTED = "started"
RETRY = "retry"
UNFINISHED = "unfinished"
ERROR_HEAD_CHARS = 120

COLUMNS: list[str] = [
    "job_id",
    "period_key",
    "run_id",
    "reason",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "retries",
    "error",
    "missing_finish",
]


def load_scheduler_events(path: str) -> list[dict]:
    """path の JSON の events を読む。読めない・形が違うときは警告を出して空を返す。"""
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("スケジューラーの記録の JSON を読めない: %s (%s)", path, e)
        return []
    events = state.get("events") if isinstance(state, dict) else None
    if not isinstance(events, list):
        logger.warning("スケジューラーの記録の JSON の形が違う（events が無い）: %s", path)
        return []
    return [event for event in events if isinstance(event, dict)]


def _text(value: Any) -> Any:
    return pd.NA if value is None else str(value)


def _error_head(value: Any) -> Any:
    if value is None:
        return pd.NA
    return str(value).splitlines()[0][:ERROR_HEAD_CHARS] if str(value) else ""


def _duration(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return pd.NA
    return round(float(value), 3)


def _new_row(event: dict) -> dict[str, Any]:
    return {
        "job_id": _text(event.get("job_id")),
        "period_key": _text(event.get("period_key")),
        "run_id": _text(event.get("run_id")),
        "reason": _text(event.get("reason")),
        "status": pd.NA,
        "started_at": _text(event.get("started_at")),
        "finished_at": pd.NA,
        "duration_seconds": pd.NA,
        "retries": 0,
        "error": pd.NA,
        "missing_finish": False,
    }


def _finish(row: dict[str, Any], event: dict) -> None:
    row["status"] = _text(event.get("status"))
    if row["started_at"] is pd.NA:
        row["started_at"] = _text(event.get("started_at"))
    if row["reason"] is pd.NA:
        row["reason"] = _text(event.get("reason"))
    row["finished_at"] = _text(event.get("finished_at"))
    row["duration_seconds"] = _duration(event.get("duration_seconds"))
    row["error"] = _error_head(event.get("error"))
    row["missing_finish"] = False


def build_scheduler_run_trend(events: list[dict]) -> pd.DataFrame:
    """events を古い順に読み、回ごとに1行の表を返す（行の並びは記録の順）。"""
    rows: list[dict[str, Any]] = []
    by_run_id: dict[str, dict[str, Any]] = {}
    open_runs: dict[tuple[Any, Any], dict[str, Any]] = {}
    pending_retries: dict[tuple[Any, Any], int] = {}

    for event in events:
        status = event.get("status")
        key = (event.get("job_id"), event.get("period_key"))
        run_id = event.get("run_id")

        if status == STARTED:
            row = _new_row(event)
            row["missing_finish"] = True
            rows.append(row)
            if run_id is not None:
                by_run_id[str(run_id)] = row
            open_runs[key] = row
            continue

        if status == RETRY:
            if key in open_runs:
                open_runs[key]["retries"] += 1
            else:
                pending_retries[key] = pending_retries.get(key, 0) + 1
            continue

        row_or_none = by_run_id.pop(str(run_id), None) if run_id is not None else None
        if row_or_none is None:
            row = _new_row(event)
            rows.append(row)
            if run_id is None:
                row["retries"] = pending_retries.pop(key, 0)
        else:
            row = row_or_none
        _finish(row, event)
        if open_runs.get(key) is row:
            del open_runs[key]

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["duration_seconds"] = df["duration_seconds"].astype("Float64")
    df["retries"] = df["retries"].astype("Int64")
    df["missing_finish"] = df["missing_finish"].astype("boolean")
    return df.reset_index(drop=True)


def load_scheduler_run_trend(path: str) -> pd.DataFrame:
    """path の JSON を読み、回ごとの表を返す。"""
    return build_scheduler_run_trend(load_scheduler_events(path))


def filter_scheduler_runs(
    df: pd.DataFrame,
    job: str | None = None,
    since: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """job_id・始めた日（since 以後。YYYY-MM-DD）・status で絞る。

    status に unfinished を渡すと、終わりの記録が無い回だけを返す。
    """
    mask = pd.Series(True, index=df.index)
    if job:
        mask &= df["job_id"] == job
    if since:
        started = df["started_at"].fillna("").astype(str).str.slice(0, 10)
        mask &= started >= since
    if status == UNFINISHED:
        mask &= df["missing_finish"].fillna(False).astype(bool)
    elif status:
        mask &= df["status"] == status
    return df[mask.fillna(False).astype(bool)].reset_index(drop=True)


def _cell(value: Any) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return "null"
    return str(value).replace("|", "\\|")


def format_scheduler_run_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（欠けは null。行が無ければ「データなし」）。"""
    if df.empty:
        return "データなし"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in values) + " |")
    return "\n".join(lines)


def format_scheduler_run_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書く）。"""
    return df.to_csv(index=False, na_rep="null")
