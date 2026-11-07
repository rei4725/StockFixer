"""
前日分の補完で、その期間がもう覆われたかの判定（#1719）

backfill の印のあるジョブの関数は日付を取らず「今」の道を流す。前の日の鍵で流しても中身は
当日の道のやり直しなので、その期間より後の日の success（当日分を含む）があれば、その期間は
覆われた扱いにする。
"""

from __future__ import annotations

from typing import Any


def has_success_since(events: list[dict[str, Any]], job_id: str, period_key: str) -> bool:
    """その日の鍵か、それより後の日の鍵に success があれば True。

    daily の鍵（YYYY-MM-DD）は文字の順が日の順なので、同じ長さの鍵を文字で比べる。
    """
    for event in events:
        key = event.get("period_key")
        if (
            event.get("job_id") == job_id
            and event.get("status") == "success"
            and isinstance(key, str)
            and len(key) == len(period_key)
            and key >= period_key
        ):
            return True
    return False
