"""
run_id 伝搬モジュール

パイプライン実行ごとに UUID ベースの run_id を発行し、
contextvars 経由でログ・DB・Discord 通知に統一付与する。
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


_current: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def get_run_id() -> str | None:
    """現在のコンテキストの run_id を返す。未設定なら None。"""
    ctx = _current.get()
    return ctx.run_id if ctx is not None else None


def set_run_context(ctx: RunContext) -> None:
    """既存の RunContext を現在のコンテキストにセットする。"""
    _current.set(ctx)


def new_run_context() -> RunContext:
    """新しい RunContext を生成してコンテキストにセットし、返す。"""
    ctx = RunContext()
    _current.set(ctx)
    return ctx
