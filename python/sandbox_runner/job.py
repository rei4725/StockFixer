"""ジョブの検査と、サンドボックスの docker run の組み立て（純粋な関数）。

ジョブから変えられるのは候補・データの場所と評価の引数だけ。イメージ・資源の上限・
ボリュームは RunnerConfig（runner の環境変数）で決まり、ジョブからは変えられない。
"""

from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass

_MOUNT = "/sandbox_share"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
_REL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,299}$")
_KEYS = {
    "source_rel",
    "data_rel",
    "windows_rel",
    "class_name",
    "rule_name",
    "description",
    "market",
    "lookback_years",
    "min_trades_per_symbol",
}


class JobError(ValueError):
    """ジョブが決まりに合わない。runner は 400 を返し、docker run を流さない。"""


@dataclass(frozen=True)
class RunnerConfig:
    image: str
    volume: str
    memory: str
    cpus: str
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "RunnerConfig":
        image = os.environ.get("SANDBOX_IMAGE", "")
        volume = os.environ.get("SANDBOX_SHARE_VOLUME", "")
        if not image or not volume:
            raise RuntimeError("SANDBOX_IMAGE と SANDBOX_SHARE_VOLUME が要ります")
        return cls(
            image,
            volume,
            os.environ.get("SANDBOX_MEMORY", "1g"),
            os.environ.get("SANDBOX_CPUS", "1"),
            int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "120")),
        )


def _rel(job: dict, key: str) -> str:
    value = job[key]
    if not isinstance(value, str) or not _REL.match(value) or ".." in value.split("/"):
        raise JobError(f"{key} は共有ボリュームの中の相対パスで書いてください: {value!r}")
    return posixpath.join(_MOUNT, value)


def _int(job: dict, key: str, low: int, high: int) -> int:
    value = job[key]
    if type(value) is not int or not low <= value <= high:
        raise JobError(f"{key} は {low}〜{high} の整数です: {value!r}")
    return value


def build_command(job: dict, cfg: RunnerConfig) -> list[str]:
    if set(job) != _KEYS:
        raise JobError(
            f"ジョブの鍵が違います: 足りない {sorted(_KEYS - set(job))} 余計 {sorted(set(job) - _KEYS)}"
        )
    for key in ("class_name", "rule_name"):
        if not isinstance(job[key], str) or not _IDENT.match(job[key]):
            raise JobError(f"{key} は識別子で書いてください: {job[key]!r}")
    if job["market"] not in ("jp", "us"):
        raise JobError(f"market は jp か us です: {job['market']!r}")
    desc = job["description"]
    if not isinstance(desc, str) or len(desc) > 500:
        raise JobError("description は500字までの文字列です")
    source, data, windows = _rel(job, "source_rel"), _rel(job, "data_rel"), _rel(job, "windows_rel")
    lookback = _int(job, "lookback_years", 1, 30)
    min_trades = _int(job, "min_trades_per_symbol", 0, 1000)
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/Logs",
        "--memory",
        cfg.memory,
        "--cpus",
        cfg.cpus,
        "-e",
        "STOCKFIXER_SANDBOX=1",
        "-e",
        "PYTHONPATH=/app",
        "-e",
        f"FACTORY_GATE_MIN_TRADES_PER_SYMBOL={min_trades}",
        "-v",
        f"{cfg.volume}:{_MOUNT}:ro",
        cfg.image,
        "python",
        "scripts/sandbox_evaluate_rule.py",
        "--source-file",
        source,
        "--class-name",
        job["class_name"],
        "--rule-name",
        job["rule_name"],
        "--description",
        desc,
        "--market",
        job["market"],
        "--lookback-years",
        str(lookback),
        "--data-dir",
        data,
        "--windows-file",
        windows,
    ]
