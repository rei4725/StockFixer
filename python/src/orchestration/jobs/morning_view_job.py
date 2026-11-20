"""朝 8:30 の目線の告知と 15:45 の答え合わせのジョブ（#1875 段3）。

目線の出どころは pick_claude_view の 1 か所で選ぶ。段4（#1956）で、旗
MORNING_VIEW_CLAUDE_ENABLED が立つ日は Claude の CLI（WebSearch つき）の結果を差し込み、
切った日・失敗の日・数字と食い違う日は数字と先物の向きだけ（source=fallback）にする。
告知は保存してから送り、送れなかった日は記録を消す。告知の無い日は答え合わせしない。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from src.domain.jp_market_holidays import is_jp_business_day
from src.market_data import morning_quotes
from src.market_data.morning_quotes import CloseQuote
from src.reporting import morning_view_store
from src.reporting.discord import webhook_sender
from src.reporting.discord.discord_notification_specs import MORNING_MARKET_VIEW
from src.reporting.morning_view import (
    FUTURES_SYMBOL,
    INDEX_LABELS,
    ClaudeView,
    IndexMove,
    ViewRecord,
    build_morning_text,
    choose_view,
    futures_view,
)
from src.reporting.morning_view_claude import SYSTEM_PROMPT, ask_claude_view
from src.utils.japan_time import now_jst, to_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

N225_SYMBOL = "^N225"
MORNING_SYMBOLS: tuple[str, ...] = tuple(INDEX_LABELS)

# 値動きから Claude の目線を求める道（段4）。ニュースは道具が返した URL のものだけ入る
ClaudeViewSource = Callable[[Sequence[IndexMove]], ClaudeView | None]

CLAUDE_MODEL = "sonnet"
CLAUDE_TOOLS = "WebSearch,WebFetch"


@dataclass(frozen=True)
class MorningAnnouncement:
    record: ViewRecord
    text: str


def _enabled() -> bool:
    from config.feature_flags import flags

    return bool(flags.MORNING_MARKET_VIEW_ENABLED)


def claude_ask() -> ClaudeViewSource | None:
    """旗が立っていれば Claude の CLI の道を返す。切っていれば None（fallback だけ）。"""
    from config.feature_flags import flags

    if not flags.MORNING_VIEW_CLAUDE_ENABLED:
        return None

    def _run(prompt: str) -> list[str]:
        from src.infrastructure.llm.cli_adapter import run_cli_stream

        return run_cli_stream(
            prompt, system=SYSTEM_PROMPT, model=CLAUDE_MODEL, allowed_tools=CLAUDE_TOOLS
        )

    return lambda moves: ask_claude_view(moves, run=_run)


def _today(now: datetime | None) -> date:
    return (to_jst(now) if now is not None else now_jst()).date()


def pick_claude_view(
    moves: Sequence[IndexMove], ask: ClaudeViewSource | None = None
) -> ClaudeView | None:
    """目線の出どころを選ぶ 1 か所。道が無ければ（段3）Claude を呼ばず、先物の向きの fallback にする。"""
    return ask(moves) if ask is not None else None


def _n225_prev_close(n225: CloseQuote, day: date) -> float:
    # 寄りの後に手で走らせて今日の足が入っていても、前日の終値は 1 つ前の足を使う
    return n225.prev_close if n225.last_day >= day else n225.last


def build_announcement(
    day: date,
    quotes: Sequence[CloseQuote],
    records: Sequence[ViewRecord],
    ask: ClaudeViewSource | None = None,
) -> MorningAnnouncement | None:
    """告知の本文と保存する記録。日経平均が無ければ前日の終値が無く、答え合わせもできないので None。"""
    by_symbol = {q.symbol: q for q in quotes}
    n225 = by_symbol.get(N225_SYMBOL)
    if n225 is None:
        return None
    prev_close = _n225_prev_close(n225, day)
    moves = [IndexMove(q.symbol, q.last, q.prev_close) for q in quotes]
    claude = pick_claude_view(moves, ask)
    view, source, _ = choose_view(claude, moves, prev_close)
    futures = by_symbol.get(FUTURES_SYMBOL)
    fut_view = futures_view(futures.last, prev_close) if futures is not None else view
    past = [r for r in records if r.day < day]
    verified = {n.url for n in claude.news} if claude is not None else set()
    text = build_morning_text(moves, prev_close, claude, past, verified)
    return MorningAnnouncement(ViewRecord(day, view, source, fut_view, None), text)


def run_morning_view_announce(now: datetime | None = None) -> None:
    """東証の営業日の 08:30 に目線を告知する。送れなかったら記録を消して投げ直す。"""
    if not _enabled():
        logger.info("朝の目線: MORNING_MARKET_VIEW_ENABLED が false のため告知しない")
        return
    day = _today(now)
    if not is_jp_business_day(day):
        logger.info("朝の目線: %s は東証の休場日のため告知を飛ばした", day)
        return
    records = morning_view_store.load_view_records()
    if any(r.day == day for r in records):
        # 保存し直すと答え合わせが消えるため、1日1回だけ保存する
        logger.info("朝の目線: %s はもう告知したため飛ばした", day)
        return

    quotes, missing = morning_quotes.fetch_close_quotes(MORNING_SYMBOLS)
    logger.info(
        "朝の目線: 取得 %d 件、取得の欠け %s",
        len(quotes),
        ",".join(missing) if missing else "なし",
    )
    announcement = build_announcement(day, quotes, records, claude_ask())
    if announcement is None:
        logger.info("朝の目線: 日経平均が取れず %s の告知を飛ばした", day)
        return

    morning_view_store.save_view_record(announcement.record)
    if not webhook_sender.send_status_notification(
        MORNING_MARKET_VIEW, announcement.text.split("\n")
    ):
        morning_view_store.delete_view_record(day)
        raise RuntimeError(f"朝の目線の告知を Discord に送れなかった（{day}）")
    logger.info(
        "朝の目線: 送った %s %s (source=%s)",
        day,
        announcement.record.view,
        announcement.record.source,
    )


def judge_change_pct(day: date, quotes: Sequence[CloseQuote]) -> float | None:
    """日経平均の今日の終値の前日の終値比（%）。今日の足が無ければ None。"""
    for q in quotes:
        if q.symbol == N225_SYMBOL and q.last_day == day and q.prev_close != 0:
            return (q.last - q.prev_close) / q.prev_close * 100.0
    return None


def run_morning_view_judge(now: datetime | None = None) -> None:
    """東証の営業日の 15:45 に、その日の告知の答え合わせを記録に書き足す。"""
    if not _enabled():
        logger.info("朝の目線: MORNING_MARKET_VIEW_ENABLED が false のため答え合わせしない")
        return
    day = _today(now)
    if not is_jp_business_day(day):
        logger.info("朝の目線: %s は東証の休場日のため答え合わせを飛ばした", day)
        return
    if not any(r.day == day for r in morning_view_store.load_view_records()):
        logger.info("朝の目線: %s は告知が無いため答え合わせを飛ばした", day)
        return

    quotes, missing = morning_quotes.fetch_close_quotes((N225_SYMBOL,))
    pct = judge_change_pct(day, quotes)
    if pct is None:
        # 失敗で終え、同じ日の補完で取り直す
        logger.info(
            "朝の目線: %s の日経平均の終値が取れず答え合わせできない（取得の欠け %s）",
            day,
            ",".join(missing) if missing else "なし",
        )
        raise RuntimeError(f"朝の目線の答え合わせに日経平均の今日の終値が無い（{day}）")
    morning_view_store.record_view_outcome(day, pct)
