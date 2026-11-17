"""朝の目線の告知（08:30）と答え合わせ（15:45）のジョブのテスト（#1875 段3）。

yfinance・DB・Discord には出ない。取得・保存・送信は差し替え、期待値は手で数えた値で書く。
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.market_data import morning_quotes
from src.market_data.morning_quotes import CloseQuote, fetch_close_quotes
from src.orchestration.jobs import morning_view_job as job
from src.reporting import morning_view_store
from src.reporting.discord import webhook_sender
from src.reporting.morning_view import ClaudeView, IndexMove, ViewRecord

JST = ZoneInfo("Asia/Tokyo")
TODAY = date(2026, 11, 18)  # 水曜の営業日
YESTERDAY = date(2026, 11, 17)


def _frame(closes: list[float], end: str) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=index)


def _quotes(*, futures: float | None = 38250.0, sox: bool = True) -> list[CloseQuote]:
    quotes = [
        CloseQuote("^GSPC", 6000.0, 5940.0, YESTERDAY),
        CloseQuote("^N225", 38000.0, 37800.0, YESTERDAY),
    ]
    if sox:
        quotes.append(CloseQuote("^SOX", 5100.0, 5000.0, YESTERDAY))
    if futures is not None:
        quotes.append(CloseQuote("NIY=F", futures, 38000.0, TODAY))
    return quotes


@pytest.fixture
def flag_on(monkeypatch):
    from config.feature_flags import flags

    monkeypatch.setattr(flags, "MORNING_MARKET_VIEW_ENABLED", True)
    return flags


class _Calls:
    """保存・送信・削除・答え合わせの呼ばれた順を溜める差し替え。"""

    def __init__(self, monkeypatch, records=None, quotes=None, missing=None, send_ok=True):
        self.order: list[tuple] = []
        self.fetched = False
        monkeypatch.setattr(morning_view_store, "load_view_records", lambda: list(records or []))
        monkeypatch.setattr(
            morning_view_store, "save_view_record", lambda r: self.order.append(("save", r))
        )
        monkeypatch.setattr(
            morning_view_store, "delete_view_record", lambda d: self.order.append(("delete", d))
        )

        def _outcome(day, pct):
            self.order.append(("outcome", day, pct))
            return True

        monkeypatch.setattr(morning_view_store, "record_view_outcome", _outcome)

        def _fetch(symbols):
            self.fetched = True
            self.symbols = list(symbols)
            return list(quotes or []), list(missing or [])

        monkeypatch.setattr(morning_quotes, "fetch_close_quotes", _fetch)

        def _send(spec, lines):
            self.order.append(("send", spec.title, "\n".join(lines)))
            return send_ok

        monkeypatch.setattr(webhook_sender, "send_status_notification", _send)


# ── 取得 ──────────────────────────────────────────────


def test_fetch_keeps_going_when_some_symbols_fail():
    def fake_download(symbol, period):
        if symbol == "^SOX":
            return pd.DataFrame()
        if symbol == "^VIX":
            raise RuntimeError("ネットの失敗")
        if symbol == "^TNX":
            return _frame([4.2], "2026-11-17")  # 1 本だけでは前日が無い
        return _frame([100.0, 101.0, 103.0], "2026-11-17")

    quotes, missing = fetch_close_quotes(["^GSPC", "^SOX", "^VIX", "^TNX", "NIY=F"], fake_download)
    assert [q.symbol for q in quotes] == ["^GSPC", "NIY=F"]
    assert missing == ["^SOX", "^VIX", "^TNX"]
    assert quotes[0] == CloseQuote("^GSPC", 103.0, 101.0, date(2026, 11, 17))


def test_job_asks_for_all_nine_symbols(monkeypatch, flag_on):
    calls = _Calls(monkeypatch, quotes=_quotes())
    job.run_morning_view_announce(now=datetime(2026, 11, 18, 8, 30, tzinfo=JST))
    assert calls.symbols == [
        "^GSPC",
        "^IXIC",
        "^DJI",
        "^SOX",
        "^VIX",
        "^TNX",
        "JPY=X",
        "NIY=F",
        "^N225",
    ]


# ── 目線の出どころ ─────────────────────────────────────


def test_pick_claude_view_without_source_is_fallback():
    assert job.pick_claude_view([IndexMove("^GSPC", 6000.0, 5940.0)]) is None


def test_pick_claude_view_passes_moves_to_the_source():
    moves = [IndexMove("^GSPC", 6000.0, 5940.0)]
    seen: list = []
    view = ClaudeView("up", "理由", "連動")

    def ask(given):
        seen.append(list(given))
        return view

    assert job.pick_claude_view(moves, ask) is view
    assert seen == [moves]


# ── 告知の組み立て（先物の向きの目線） ─────────────────────


def test_futures_above_previous_close_is_up():
    # 先物 38250 > 日経平均の前日の終値 38000 → 上げ、差は +250 円
    ann = job.build_announcement(TODAY, _quotes(futures=38250.0), [])
    assert ann is not None
    assert ann.record == ViewRecord(TODAY, "up", "fallback", "up", None)
    assert "**上げ目線**" in ann.text
    assert "+250 円" in ann.text


def test_futures_below_previous_close_is_down():
    # 先物 37900 < 38000 → 下げ、差は -100 円
    ann = job.build_announcement(TODAY, _quotes(futures=37900.0), [])
    assert ann is not None
    assert ann.record == ViewRecord(TODAY, "down", "fallback", "down", None)
    assert "**下げ目線**" in ann.text
    assert "-100 円" in ann.text


def test_announcement_built_from_the_rest_when_some_are_missing():
    ann = job.build_announcement(TODAY, _quotes(sox=False), [])
    assert ann is not None
    assert "SOX" not in ann.text
    assert "S&P500" in ann.text
    assert ann.record.view == "up"


def test_without_futures_view_falls_back_to_down():
    ann = job.build_announcement(TODAY, _quotes(futures=None), [])
    assert ann is not None
    assert ann.record == ViewRecord(TODAY, "down", "fallback", "down", None)


def test_without_nikkei_no_announcement():
    quotes = [q for q in _quotes() if q.symbol != "^N225"]
    assert job.build_announcement(TODAY, quotes, []) is None


def test_today_bar_of_nikkei_uses_its_previous_close():
    # 手で走らせて寄りの後に今日の足が入っても、前日の終値は 1 つ前の足 37800 を使う
    quotes = [
        CloseQuote("^N225", 38100.0, 37800.0, TODAY),
        CloseQuote("NIY=F", 37900.0, 37700.0, TODAY),
    ]
    ann = job.build_announcement(TODAY, quotes, [])
    assert ann is not None
    assert ann.record.view == "up"  # 37900 > 37800
    assert "+100 円" in ann.text


def test_tally_counts_only_days_before_today():
    past = [
        ViewRecord(date(2026, 11, 16), "up", "fallback", "up", 0.8),
        ViewRecord(date(2026, 11, 17), "up", "fallback", "up", -0.4),
    ]
    ann = job.build_announcement(TODAY, _quotes(), past)
    assert ann is not None
    # 2 日告知して今日で 3 日目、目線は 1勝1敗
    assert "【検証中（3 日目）】" in ann.text
    assert "この目線 1勝1敗（50%）" in ann.text


# ── 08:30 の告知のジョブ ─────────────────────────────────


def test_announce_saves_before_sending(monkeypatch, flag_on):
    calls = _Calls(monkeypatch, quotes=_quotes(), missing=["^VIX"])
    job.run_morning_view_announce(now=datetime(2026, 11, 18, 8, 30, tzinfo=JST))
    assert [c[0] for c in calls.order] == ["save", "send"]
    assert calls.order[0][1] == ViewRecord(TODAY, "up", "fallback", "up", None)
    assert calls.order[1][1] == "🌅 朝の目線（日本株）"
    assert "**上げ目線**" in calls.order[1][2]


def test_announce_skips_market_holiday(monkeypatch, flag_on):
    calls = _Calls(monkeypatch, quotes=_quotes())
    # 2026-11-23 は月曜の勤労感謝の日
    job.run_morning_view_announce(now=datetime(2026, 11, 23, 8, 30, tzinfo=JST))
    assert calls.fetched is False
    assert calls.order == []


def test_announce_skips_weekend(monkeypatch, flag_on):
    calls = _Calls(monkeypatch, quotes=_quotes())
    job.run_morning_view_announce(now=datetime(2026, 11, 21, 8, 30, tzinfo=JST))
    assert calls.fetched is False
    assert calls.order == []


def test_announce_does_not_resave_announced_day(monkeypatch, flag_on):
    records = [ViewRecord(TODAY, "up", "fallback", "up", None)]
    calls = _Calls(monkeypatch, records=records, quotes=_quotes())
    job.run_morning_view_announce(now=datetime(2026, 11, 18, 8, 40, tzinfo=JST))
    assert calls.fetched is False
    assert calls.order == []


def test_announce_without_nikkei_saves_and_sends_nothing(monkeypatch, flag_on):
    quotes = [q for q in _quotes() if q.symbol != "^N225"]
    calls = _Calls(monkeypatch, quotes=quotes, missing=["^N225"])
    job.run_morning_view_announce(now=datetime(2026, 11, 18, 8, 30, tzinfo=JST))
    assert calls.order == []


def test_send_failure_removes_the_record_and_raises(monkeypatch, flag_on):
    calls = _Calls(monkeypatch, quotes=_quotes(), send_ok=False)
    with pytest.raises(RuntimeError):
        job.run_morning_view_announce(now=datetime(2026, 11, 18, 8, 30, tzinfo=JST))
    assert [c[0] for c in calls.order] == ["save", "send", "delete"]
    assert calls.order[2][1] == TODAY


def test_flag_off_does_nothing(monkeypatch, flag_on):
    monkeypatch.setattr(flag_on, "MORNING_MARKET_VIEW_ENABLED", False)
    calls = _Calls(monkeypatch, quotes=_quotes())
    job.run_morning_view_announce(now=datetime(2026, 11, 18, 8, 30, tzinfo=JST))
    job.run_morning_view_judge(now=datetime(2026, 11, 18, 15, 45, tzinfo=JST))
    assert calls.fetched is False
    assert calls.order == []


# ── 15:45 の答え合わせのジョブ ───────────────────────────


def test_judge_writes_nikkei_change_against_previous_close(monkeypatch, flag_on):
    records = [ViewRecord(TODAY, "up", "fallback", "up", None)]
    # 38380 / 38000 - 1 = +1.0%
    quotes = [CloseQuote("^N225", 38380.0, 38000.0, TODAY)]
    calls = _Calls(monkeypatch, records=records, quotes=quotes)
    job.run_morning_view_judge(now=datetime(2026, 11, 18, 15, 45, tzinfo=JST))
    assert calls.symbols == ["^N225"]
    assert len(calls.order) == 1
    kind, day, pct = calls.order[0]
    assert (kind, day) == ("outcome", TODAY)
    assert pct == pytest.approx(1.0)


def test_judge_skips_day_without_announcement(monkeypatch, flag_on):
    records = [ViewRecord(YESTERDAY, "up", "fallback", "up", None)]
    quotes = [CloseQuote("^N225", 38380.0, 38000.0, TODAY)]
    calls = _Calls(monkeypatch, records=records, quotes=quotes)
    job.run_morning_view_judge(now=datetime(2026, 11, 18, 15, 45, tzinfo=JST))
    assert calls.fetched is False
    assert calls.order == []


def test_judge_fails_for_recovery_when_today_bar_is_not_there(monkeypatch, flag_on):
    records = [ViewRecord(TODAY, "up", "fallback", "up", None)]
    quotes = [CloseQuote("^N225", 38000.0, 37800.0, YESTERDAY)]
    calls = _Calls(monkeypatch, records=records, quotes=quotes)
    with pytest.raises(RuntimeError):
        job.run_morning_view_judge(now=datetime(2026, 11, 18, 15, 45, tzinfo=JST))
    assert calls.order == []


def test_judge_skips_holiday(monkeypatch, flag_on):
    calls = _Calls(monkeypatch, quotes=[CloseQuote("^N225", 1.0, 1.0, date(2026, 11, 23))])
    job.run_morning_view_judge(now=datetime(2026, 11, 23, 15, 45, tzinfo=JST))
    assert calls.fetched is False
    assert calls.order == []


# ── SCHEDULE_CONFIG ────────────────────────────────────


def test_schedule_config_has_both_jobs():
    from run_scheduler import SCHEDULE_CONFIG

    announce = SCHEDULE_CONFIG["morning_market_view"]
    assert (announce["day_of_week"], announce["hour"], announce["minute"]) == ("mon-fri", 8, 30)
    # 締めは 8:50 の発注より前。遅れたら出さない
    assert announce["recovery_until"] == (8, 45)
    assert announce["max_executions_per_period"] == 1
    assert announce["max_retries"] == 0

    judge = SCHEDULE_CONFIG["morning_market_view_judge"]
    assert (judge["day_of_week"], judge["hour"], judge["minute"]) == ("mon-fri", 15, 45)
    assert "recovery_until" not in judge
    assert judge["max_executions_per_period"] >= 2
