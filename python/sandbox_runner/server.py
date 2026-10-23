"""sandbox-runner の HTTP サーバ。POST /run でジョブを受け、決まった docker run を流す。"""

from __future__ import annotations

import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sandbox_runner.job import JobError, RunnerConfig, build_command

_MAX_BODY = 64 * 1024
_TAIL = 20000


def handle_run(body: bytes, cfg: RunnerConfig, run=subprocess.run) -> tuple[int, dict]:
    try:
        job = json.loads(body)
        if not isinstance(job, dict):
            raise JobError("ジョブは JSON のオブジェクトです")
        cmd = build_command(job, cfg)
    except (json.JSONDecodeError, JobError) as e:
        return 400, {"error": str(e)}
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
        return 200, {"timeout": True}
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
            (
                self._send(200, {"status": "ok"})
                if self.path == "/health"
                else self._send(404, {"error": "not found"})
            )

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            if self.path != "/run" or length > _MAX_BODY:
                self._send(404 if self.path != "/run" else 413, {"error": "not accepted"})
                return
            self._send(*handle_run(self.rfile.read(length), cfg))

    ThreadingHTTPServer(("0.0.0.0", 8700), Handler).serve_forever()


if __name__ == "__main__":
    main()
