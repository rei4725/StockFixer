"""sandbox-runner の HTTP サーバ。POST /run でジョブを受け、決まった docker run を流す。"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sandbox_runner.job import JobError, RunnerConfig, build_command

_MAX_BODY = 64 * 1024
_TAIL = 20000
_DIGITS = re.compile(r"[0-9]+")
# 同時に走らせるサンドボックスの数。埋まっていれば待たせず 503 を返す
_SLOTS = threading.BoundedSemaphore(2)


def parse_length(value: str | None) -> int | None:
    """Content-Length を読む。無いか空なら 0、数字でなければ None（不正）。"""
    if value is None or value == "":
        return 0
    if not _DIGITS.fullmatch(value):
        return None
    return int(value)


def handle_run(body: bytes, cfg: RunnerConfig, run=subprocess.run) -> tuple[int, dict]:
    if not _SLOTS.acquire(blocking=False):
        return 503, {"error": "sandbox-runner が混んでいます"}
    try:
        return _handle_run(body, cfg, run)
    finally:
        _SLOTS.release()


def _handle_run(body: bytes, cfg: RunnerConfig, run) -> tuple[int, dict]:
    try:
        job = json.loads(body)
        if not isinstance(job, dict):
            raise JobError("ジョブは JSON のオブジェクトです")
        cmd = build_command(job, cfg)
    except (ValueError, JobError) as e:
        return 400, {"error": str(e)}
    name = "factory-sandbox-" + uuid.uuid4().hex[:12]
    cmd = cmd[:2] + ["--name", name] + cmd[2:]
    try:
        proc = run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=cfg.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        try:
            run(["docker", "kill", name], capture_output=True, text=True, timeout=30, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return 200, {"timeout": True}
    except OSError as e:
        return 500, {"error": f"docker を実行できません: {e}"}
    return 200, {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-_TAIL:],
        "stderr": proc.stderr[-_TAIL:],
    }


def main() -> None:
    cfg = RunnerConfig.from_env()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/run":
                self._send(404, {"error": "not found"})
                return
            length = parse_length(self.headers.get("Content-Length"))
            if length is None:
                self._send(400, {"error": "Content-Length が不正です"})
                return
            if length > _MAX_BODY:
                self._send(413, {"error": "body too large"})
                return
            self._send(*handle_run(self.rfile.read(length), cfg))

    ThreadingHTTPServer(("0.0.0.0", 8700), Handler).serve_forever()


if __name__ == "__main__":
    main()
