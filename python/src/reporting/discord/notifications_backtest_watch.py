"""仮の入れ替えの見張りが判定を待っていることの Discord の知らせ（#1713）。

notifications_model.py が 600 行に近いため、別のモジュールに置く。
"""

from __future__ import annotations

from src.reporting.discord.discord_notification_specs import BACKTEST_PROMOTION_WATCH_WAITING
from src.reporting.discord.discord_text import DISCORD_DATETIME_FORMAT
from src.reporting.discord.webhook_sender import send_status_fields
from src.utils.japan_time import format_jst

_FIELD_LIMIT = 1024


def _field(name: str, value: str, inline: bool = False) -> dict:
    return {"name": name, "value": (value or "-")[:_FIELD_LIMIT], "inline": inline}


def send_backtest_promotion_watch_waiting(result: dict) -> bool:
    """見張りが飛ばした日が線に届き、まだ判定していないことを送る。

    Args:
        result: run_backtest_promotion() の戻り値。result["watch"] に見張りの途中の数
    """
    watch = result.get("watch") or {}
    skipped = list(watch.get("skipped_days") or [])
    need_days = watch.get("need_days")
    fields = [
        _field("🕐 時刻", format_jst(fmt=DISCORD_DATETIME_FORMAT), True),
        _field("🔖 入れ替えた版", f"`{watch.get('version') or '-'}`", True),
        _field("✅ 数えた日", f"{watch.get('scored_days', 0)} 日（{need_days} 日で判定）", True),
        _field("⏭️ 飛ばした日", f"{len(skipped)} 日", True),
        _field("📅 飛ばした日の一覧", "\n".join(skipped)),
        _field(
            "📏 数える線",
            f"jp の採点の行のある銘柄が {watch.get('need_symbols')} 以上の日"
            f"（ウォッチリストの jp {watch.get('n_universe_symbols')} 銘柄）",
        ),
    ]
    description = (
        "入れ替えた候補は確かめられないまま日次の予想に使われ続けている。"
        "jp の取得か採点が止まっていないかを確かめる。この知らせは見張りごとに1回だけ出す。"
    )
    return send_status_fields(BACKTEST_PROMOTION_WATCH_WAITING, fields, description=description)
