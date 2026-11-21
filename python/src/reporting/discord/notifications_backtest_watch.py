"""仮の入れ替えの見張りが判定を待っていることの Discord の知らせ（#1713）。

notifications_model.py が 600 行に近いため、別のモジュールに置く。
"""

from __future__ import annotations

from src.reporting.discord.discord_notification_specs import (
    BACKTEST_PROMOTION_STALE_FEATURES,
    BACKTEST_PROMOTION_WATCH_WAITING,
)
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


def send_backtest_promotion_stale_features(result: dict) -> bool:
    """仮の入れ替えが「特徴量がまだ今のコードで作り直されていない」で続けて飛んだことを送る（#2014）。

    Args:
        result: run_backtest_promotion() の戻り値。stale_features_nights に続けて飛んだ夜の数
    """
    fields = [
        _field("🕐 時刻", format_jst(fmt=DISCORD_DATETIME_FORMAT), True),
        _field("🌙 続けて飛んだ夜", f"{result.get('stale_features_nights')} 晩", True),
        _field("🔖 特徴量の印", f"`{result.get('features_version') or '-'}`", True),
        _field("🧩 今の特徴量の版", f"`{result.get('features_code_version') or '-'}`", True),
        _field("📦 予想の版", f"`{result.get('version') or '-'}`", True),
        _field("📂 特徴量の版の範囲", ", ".join(result.get("features_scope") or [])),
        _field("⏰ 次の日次の道", result.get("next_daily_run") or "-", True),
    ]
    # 印の後に日次の道の時刻を過ぎた夜だけを数えるので、この知らせは (b) の原因だけを指す（#2086）
    description = (
        "日次の道が走ったのに、特徴量の版の印を今の版で書けていない"
        "（印の後のコードの変更で次の日次の道を待つだけの夜は数えていない）。"
        "[1/6] のデータ取得が止まっていないか、作れなかった銘柄が線を超えていないかを確かめる。"
    )
    return send_status_fields(BACKTEST_PROMOTION_STALE_FEATURES, fields, description=description)
