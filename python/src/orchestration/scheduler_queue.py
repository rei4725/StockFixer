"""
スケジューラのキュー方式実行管理

所定時刻以降に実行ログが無いジョブを補完実行し、
実行結果ログと実行回数制御を提供する。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from config.settings import settings
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.heartbeat import ping_heartbeat
from src.utils.japan_time import isoformat_utc

logger = logging.getLogger("scheduler")

_WEEKDAY_MAP = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


class SchedulerQueueManager:
    """定期ジョブの実行状態を管理する。"""

    def __init__(
        self,
        schedule_config: dict[str, dict[str, Any]],
        state_file_path: str | None = None,
        timezone_name: str = "Asia/Tokyo",
        error_notifier: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.schedule_config = schedule_config
        self.tz = ZoneInfo(timezone_name)
        if state_file_path is None:
            state_file_path = os.path.join(get_results_dir(), "scheduler_queue_state.json")
        self.state_file_path = state_file_path
        self._in_progress_jobs: set[str] = set()
        self._error_notifier = error_notifier
        self._state = self._load_state()

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def _emit_job_log(self, level: str, phase: str, **fields: Any) -> None:
        """集計しやすいJSON 1行形式でジョブログを出力する。"""
        payload = {
            "event": "scheduler_job",
            "phase": phase,
            **fields,
        }
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def run_job(
        self, job_id: str, reason: str, force: bool = False, period_key: str | None = None
    ) -> bool:
        """
        ジョブを実行し、ログと実行回数制御を適用する。

        Args:
            period_key: ログに記録する期間キー。Noneの場合は実行日時から自動生成。
                        前日分の補完実行など、実際の対象期間を明示したい場合に使用する。

        Returns:
            実際に実行した場合はTrue、スキップ時はFalse
        """
        if job_id not in self.schedule_config:
            raise KeyError(f"未知のjob_id: {job_id}")

        config = self.schedule_config[job_id]
        now = self.now()
        if period_key is None:
            period_key = self._build_period_key(config, now)

        if job_id in self._in_progress_jobs:
            self._emit_job_log(
                "info",
                "skip",
                job_id=job_id,
                reason=reason,
                period_key=period_key,
                status="skipped_in_progress",
            )
            return False

        current_count = self.get_execution_count(job_id, period_key)
        max_executions = config.get("max_executions_per_period", 1)
        if not force and current_count >= max_executions:
            self._append_event(
                {
                    "job_id": job_id,
                    "reason": reason,
                    "status": "skipped_limit",
                    "period_key": period_key,
                    "started_at": isoformat_utc(now),
                    "finished_at": isoformat_utc(now),
                    "duration_seconds": 0.0,
                    "error": None,
                }
            )
            self._emit_job_log(
                "info",
                "skip",
                job_id=job_id,
                reason=reason,
                period_key=period_key,
                status="skipped_limit",
                current_count=current_count,
                max_executions=max_executions,
            )
            return False

        started_at = self.now()
        self._in_progress_jobs.add(job_id)
        status = "success"
        error_message = None

        self._emit_job_log(
            "info",
            "start",
            job_id=job_id,
            reason=reason,
            period_key=period_key,
            force=force,
        )

        try:
            self._run_with_retry(job_id, config, reason, period_key)
            return True
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            finished_at = self.now()
            duration_seconds = (finished_at - started_at).total_seconds()
            rounded_duration = round(duration_seconds, 3)

            if status == "success":
                self._emit_job_log(
                    "info",
                    "finish",
                    job_id=job_id,
                    reason=reason,
                    period_key=period_key,
                    status="success",
                    duration_seconds=rounded_duration,
                )
            else:
                self._emit_job_log(
                    "error",
                    "finish",
                    job_id=job_id,
                    reason=reason,
                    period_key=period_key,
                    status="error",
                    duration_seconds=rounded_duration,
                    error=error_message,
                )

            self._append_event(
                {
                    "job_id": job_id,
                    "reason": reason,
                    "status": status,
                    "period_key": period_key,
                    "started_at": isoformat_utc(started_at),
                    "finished_at": isoformat_utc(finished_at),
                    "duration_seconds": duration_seconds,
                    "error": error_message,
                }
            )
            self._in_progress_jobs.discard(job_id)
            # 外部死活監視（#496）: 実行したジョブのみ ping（skip 経路は送らない）
            ping_heartbeat(job_id, success=(status == "success"))

    def _run_with_retry(
        self, job_id: str, config: dict[str, Any], reason: str, period_key: str
    ) -> None:
        """Exponential backoff + jitter でジョブ関数を実行する。"""
        max_retries = config.get("max_retries", settings.SCHEDULER_MAX_RETRIES)
        base_wait = config.get(
            "retry_base_wait_seconds", settings.SCHEDULER_RETRY_BASE_WAIT_SECONDS
        )

        for attempt in range(max_retries + 1):
            try:
                config["func"]()
                return
            except Exception as exc:
                if attempt < max_retries:
                    wait = base_wait * (2**attempt) + random.uniform(0, base_wait * 0.1)
                    self._append_event(
                        {
                            "job_id": job_id,
                            "reason": reason,
                            "status": "retry",
                            "period_key": period_key,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error": str(exc),
                            "wait_seconds": round(wait, 3),
                            "timestamp": isoformat_utc(self.now()),
                        }
                    )
                    self._emit_job_log(
                        "error",
                        "retry",
                        job_id=job_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_seconds=round(wait, 3),
                        error=str(exc),
                    )
                    time.sleep(wait)
                else:
                    self._emit_job_log(
                        "error",
                        "max_retries_exceeded",
                        job_id=job_id,
                        max_retries=max_retries,
                        error=str(exc),
                    )
                    self._send_error_notification(job_id, max_retries, str(exc))
                    raise

    def _send_error_notification(self, job_id: str, max_retries: int, error_message: str) -> None:
        """最大リトライ超過後の Discord エラー通知を送信する。"""
        if self._error_notifier is not None:
            try:
                self._error_notifier(job_id, error_message)
            except Exception:
                logger.error(
                    "Discord エラー通知の送信に失敗しました",
                    exc_info=True,
                )

    def get_missed_past_periods(
        self, job_id: str, lookback_days: int = 2, now: datetime | None = None
    ) -> list[str]:
        """
        過去N日分で未実行（成功なし・上限未到達）の period_key を返す（daily ジョブ専用）。

        コンテナ再起動などで当日スケジュール時刻前に起動した場合に、
        前日分の欠落を検出して補完実行するために使用する。
        """
        if job_id not in self.schedule_config:
            return []

        config = self.schedule_config[job_id]
        if config.get("period", "daily") != "daily":
            return []

        now = now or self.now()
        missed: list[str] = []
        max_executions = config.get("max_executions_per_period", 1)

        for days_back in range(1, lookback_days + 1):
            past_dt = now - timedelta(days=days_back)
            if not self._is_scheduled_day(config, past_dt):
                continue
            past_period_key = self._build_period_key(config, past_dt)
            if self.has_success(job_id, past_period_key):
                continue
            if self.get_execution_count(job_id, past_period_key) >= max_executions:
                continue
            missed.append(past_period_key)

        return missed

    def should_recover(self, job_id: str, now: datetime | None = None) -> bool:
        """補完実行すべきか判定する。"""
        if job_id not in self.schedule_config:
            return False

        config = self.schedule_config[job_id]
        now = now or self.now()

        if not self._is_scheduled_day(config, now):
            return False

        scheduled_dt = now.replace(
            hour=config["hour"],
            minute=config["minute"],
            second=0,
            microsecond=0,
        )
        recovery_delay = timedelta(minutes=config.get("recovery_delay_minutes", 10))

        if now < (scheduled_dt + recovery_delay):
            return False

        period_key = self._build_period_key(config, now)
        if self.has_success(job_id, period_key):
            return False

        current_count = self.get_execution_count(job_id, period_key)
        max_executions = config.get("max_executions_per_period", 1)
        if current_count >= max_executions:
            return False

        return True

    def has_success(self, job_id: str, period_key: str) -> bool:
        for event in self._state.get("events", []):
            if (
                event.get("job_id") == job_id
                and event.get("period_key") == period_key
                and event.get("status") == "success"
            ):
                return True
        return False

    def get_execution_count(self, job_id: str, period_key: str) -> int:
        count = 0
        for event in self._state.get("events", []):
            if event.get("job_id") != job_id or event.get("period_key") != period_key:
                continue
            if event.get("status") in {"success", "error"}:
                count += 1
        return count

    def _build_period_key(self, config: dict[str, Any], now: datetime) -> str:
        period_type = config.get("period", "daily")
        if period_type == "weekly":
            iso = now.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
        return now.strftime("%Y-%m-%d")

    def _is_scheduled_day(self, config: dict[str, Any], now: datetime) -> bool:
        day_expr = config.get("day_of_week", "*")
        valid_weekdays = self._parse_day_of_week(day_expr)
        if not valid_weekdays:
            return True
        return now.weekday() in valid_weekdays

    def _parse_day_of_week(self, expr: str) -> set[int]:
        expr = (expr or "").strip().lower()
        if not expr or expr == "*":
            return set()

        weekdays: set[int] = set()
        for token in [part.strip() for part in expr.split(",") if part.strip()]:
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = _WEEKDAY_MAP.get(start_text)
                end = _WEEKDAY_MAP.get(end_text)
                if start is None or end is None:
                    continue
                if start <= end:
                    weekdays.update(range(start, end + 1))
                else:
                    weekdays.update(range(start, 7))
                    weekdays.update(range(0, end + 1))
            else:
                value = _WEEKDAY_MAP.get(token)
                if value is not None:
                    weekdays.add(value)
        return weekdays

    def _load_state(self) -> dict[str, Any]:
        if not os.path.exists(self.state_file_path):
            return {"events": []}

        try:
            with open(self.state_file_path, "r", encoding="utf-8") as file:
                state = json.load(file)
            if not isinstance(state, dict):
                return {"events": []}
            events = state.get("events")
            if not isinstance(events, list):
                state["events"] = []
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"スケジューラ状態ファイル読込失敗: {exc}")
            return {"events": []}

    def _save_state(self) -> None:
        ensure_dir(os.path.dirname(self.state_file_path))
        with open(self.state_file_path, "w", encoding="utf-8") as file:
            json.dump(self._state, file, ensure_ascii=False, indent=2)

    def _append_event(self, event: dict[str, Any]) -> None:
        self._state.setdefault("events", []).append(event)
        max_events = 5000
        if len(self._state["events"]) > max_events:
            self._state["events"] = self._state["events"][-max_events:]
        self._save_state()
