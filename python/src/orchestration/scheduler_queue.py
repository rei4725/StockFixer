"""
スケジューラのキュー方式実行管理

所定時刻以降に実行ログが無いジョブを補完実行し、
実行結果ログと実行回数制御を提供する。

状態ファイルは常駐のスケジューラーと手での即時実行（別プロセス）が共に使うため、
判定の前に読み直し、書くときはファイルのロックの内で読み直して足し、
一時ファイルから置き換える（#826）。
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from filelock import FileLock

from config.settings import settings
from src.orchestration.scheduler_interrupted import count_executions
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
        # 起動ごとの印。始めの記録に入れ、前の起動で消えた回を見分ける（#946）
        self._boot_id = uuid.uuid4().hex
        # started_at は秒までで書くため、比べる起動の時刻も秒で切る
        self._booted_at = datetime.now(timezone.utc).replace(microsecond=0)
        self._warned_interrupted_run_ids: set[str] = set()
        self._error_notifier = error_notifier
        # 同じプロセスのスレッドの間（定時のジョブと補完のポーリング）を _lock で、
        # プロセスの間（手での即時実行）を _file_lock で守る。取る順は必ず _lock が先
        self._lock = threading.Lock()
        ensure_dir(os.path.dirname(os.path.abspath(state_file_path)))
        self._file_lock = FileLock(f"{state_file_path}.lock")
        # 壊れた状態ファイルを退避したら、このプロセスの間は補完を止める
        self._state_corrupted = False
        self._state: dict[str, Any] = {"events": []}
        with self._lock, self._file_lock:
            self._reload_state()

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

        # 実行中の確かめ・回数の確かめ・始めの記録を、ロックの内でひと続きに行う。
        # 回数は別のプロセスの記録も数えるため、ファイルを読み直してから数える
        with self._lock, self._file_lock:
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

            self._reload_state()
            current_count = self.get_execution_count(job_id, period_key)
            max_executions = config.get("max_executions_per_period", 1)
            if not force and current_count >= max_executions:
                self._append_event_locked(
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
            # 終わりまで走ったときだけ success にする。Exception でない BaseException
            # （SystemExit・KeyboardInterrupt）で止まった回を success と記録しない（#955）
            status = "error"
            error_message = None
            run_id = uuid.uuid4().hex

            # プロセスごと落ちて finally が走らなくても回数に数えるため、始める前に記録する
            self._append_event_locked(
                {
                    "job_id": job_id,
                    "reason": reason,
                    "status": "started",
                    "period_key": period_key,
                    "run_id": run_id,
                    "boot_id": self._boot_id,
                    "started_at": isoformat_utc(started_at),
                }
            )
            self._in_progress_jobs.add(job_id)

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
            status = "success"
            return True
        except Exception as exc:
            error_message = str(exc)
            raise
        except BaseException as exc:
            # KeyboardInterrupt は str が空になるため、型の名前を添える
            error_message = f"{type(exc).__name__}: {exc}"
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
                    "run_id": run_id,
                    "started_at": isoformat_utc(started_at),
                    "finished_at": isoformat_utc(finished_at),
                    "duration_seconds": duration_seconds,
                    "error": error_message,
                }
            )
            with self._lock:
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

        対象は設定に ``backfill: True`` を持つジョブだけ。印の無いジョブ（発注・約定など
        その日の相場にしか意味の無いジョブ）は、当日分と別の鍵で同じ日に二度動かないよう
        前日分を補完しない（#825）。
        """
        if job_id not in self.schedule_config:
            return []

        config = self.schedule_config[job_id]
        if config.get("period", "daily") != "daily":
            return []
        if not config.get("backfill", False):
            return []
        if not self._reload_state_for_recovery():
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

        # 定時の回が動いている間は補完しない（run_job でも飛ばすが、
        # ここで返さないと5分おきに補完の始まりの行が出る。#806）
        with self._lock:
            if job_id in self._in_progress_jobs:
                return False

        config = self.schedule_config[job_id]
        now = now or self.now()

        # hour が "*/2" のようなcron式（1日複数回実行のジョブ）の場合、
        # 「所定の1時刻」という前提のdatetime.replace()が使えない。
        # このようなジョブは自身の頻度で自然にリトライされるため、
        # 補完ポーリングの対象外として扱う（クラッシュではなくスキップ）。
        if not isinstance(config["hour"], int) or not isinstance(config["minute"], int):
            return False

        if not self._is_scheduled_day(config, now):
            return False

        # monthly は設定の day の日（とそれ以後の同じ月）だけ補完を考え、
        # 所定時刻はその月の day の日の時刻とする
        scheduled_day = now.day
        if config.get("period", "daily") == "monthly":
            scheduled_day = config.get("day", 1)
            if not isinstance(scheduled_day, int):
                return False
            if now.day < scheduled_day:
                return False

        scheduled_dt = now.replace(
            day=scheduled_day,
            hour=config["hour"],
            minute=config["minute"],
            second=0,
            microsecond=0,
        )
        recovery_delay = timedelta(minutes=config.get("recovery_delay_minutes", 10))

        if now < (scheduled_dt + recovery_delay):
            return False

        # recovery_until (hour, minute) を持つジョブは、その日のその時刻から補完しない。
        # 発注のようにその日の相場の前にしか意味の無いジョブを、引けの後や夜に動かさない（#835）
        recovery_until = config.get("recovery_until")
        if recovery_until is not None:
            until_hour, until_minute = recovery_until
            if now >= now.replace(hour=until_hour, minute=until_minute, second=0, microsecond=0):
                return False

        if not self._reload_state_for_recovery():
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
        """success・error の記録に、終わりの記録が無い started（途中で落ちた実行）を足して数える。

        設定に ``rerun_if_interrupted: True`` を持つジョブは、前の起動のプロセスで消えた回を
        1期間に1回まで数えない（#946）。
        """
        config = self.schedule_config.get(job_id, {})
        rerun_if_interrupted = bool(config.get("rerun_if_interrupted", False))
        result = count_executions(
            self._state.get("events", []),
            job_id,
            period_key,
            boot_id=self._boot_id,
            booted_at=self._booted_at,
            rerun_if_interrupted=rerun_if_interrupted,
        )
        for run_id in result.interrupted_run_ids:
            if run_id in self._warned_interrupted_run_ids:
                continue
            self._warned_interrupted_run_ids.add(run_id)
            logger.warning(
                f"前の起動のプロセスで消えた回を見つけました: job_id={job_id} "
                f"period_key={period_key} run_id={run_id} "
                f"補完の対象={'あり（1期間に1回まで）' if rerun_if_interrupted else 'なし'}"
            )
        return result.count

    def _build_period_key(self, config: dict[str, Any], now: datetime) -> str:
        period_type = config.get("period", "daily")
        if period_type == "weekly":
            iso = now.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
        if period_type == "monthly":
            return now.strftime("%Y-%m")
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

    def _reload_state_for_recovery(self) -> bool:
        """補完の判定の前に読み直す。補完してよい履歴が手元にあれば True。"""
        with self._lock, self._file_lock:
            reloaded = self._reload_state()
        return reloaded and not self._state_corrupted

    def _reload_state(self) -> bool:
        """状態ファイルを読み直す（ロックの内で呼ぶ）。読めなければ手元の記録を残して False。"""
        if not os.path.exists(self.state_file_path):
            self._state = {"events": []}
            return True

        try:
            with open(self.state_file_path, "r", encoding="utf-8") as file:
                state = json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._quarantine_state_file(str(exc))
            return False
        except OSError:
            logger.error("スケジューラ状態ファイル読込失敗", exc_info=True)
            return False

        if not isinstance(state, dict):
            self._quarantine_state_file(f"最上位が dict ではない: {type(state).__name__}")
            return False
        events = state.get("events")
        if not isinstance(events, list):
            state["events"] = []
        self._state = state
        return True

    def _quarantine_state_file(self, detail: str) -> None:
        """壊れた状態ファイルを隣へ日時つきで退避し、このプロセスの補完を止める。

        空の履歴で続けると、済んだ日次のジョブ（発注を含む）を補完でやり直すため。
        """
        self._state_corrupted = True
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine_path = f"{self.state_file_path}.corrupt-{stamp}"
        try:
            os.replace(self.state_file_path, quarantine_path)
        except OSError:
            logger.error(
                f"壊れたスケジューラ状態ファイルの退避に失敗: {self.state_file_path}",
                exc_info=True,
            )
            return
        logger.error(
            "スケジューラ状態ファイルが読めないため退避し、補完実行を止めます"
            f"（定時のジョブは続けます）: {detail} 退避先={quarantine_path}"
        )

    def _save_state(self) -> None:
        """一時ファイルに書いてから置き換える（ロックの内で呼ぶ）。"""
        state_dir = os.path.dirname(os.path.abspath(self.state_file_path))
        ensure_dir(state_dir)
        fd, tmp_path = tempfile.mkstemp(
            dir=state_dir, prefix=f"{os.path.basename(self.state_file_path)}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(self._state, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, self.state_file_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _append_event(self, event: dict[str, Any]) -> None:
        with self._lock, self._file_lock:
            # 別のプロセスがその間に足した記録を消さないよう、読み直してから足す
            self._reload_state()
            self._append_event_locked(event)

    def _append_event_locked(self, event: dict[str, Any]) -> None:
        self._state.setdefault("events", []).append(event)
        max_events = 5000
        if len(self._state["events"]) > max_events:
            self._state["events"] = self._state["events"][-max_events:]
        self._save_state()
