"""月次 multibagger ジョブ + ポジション永続化の単体テスト（#434）"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.types import PositionEvent, TrendCandidate
from src.market_data import fundamentals
from src.orchestration import multibagger_job
from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.scheduler_queue import SchedulerQueueManager
from src.screening import positions as positions_mod
from src.screening import quality_gate
from src.screening.positions import add_position, close_position, list_positions, update_position
from src.screening.trend_screener import ScreeningDataError
from src.screening.types import MultibaggerCandidate


@pytest.fixture
def tmp_positions(tmp_path, monkeypatch):
    """ポジション台帳 JSON を一時ファイルに差し替える。"""
    path = tmp_path / "multibagger_positions.json"
    monkeypatch.setattr(positions_mod, "get_multibagger_positions_path", lambda: str(path))
    return path


def _candidate(symbol: str, score: float = 0.9) -> MultibaggerCandidate:
    return MultibaggerCandidate(
        market="us",
        symbol=symbol,
        multibagger_score=score,
        trend_score=0.8,
        growth_score=0.7,
        quality_score=0.6,
        close=120.0,
        revenue_cagr_3y=0.30,
        roe=0.25,
        op_margin=0.20,
        net_margin=0.15,
        debt_to_equity=0.4,
        market_cap=5e9,
        fundamentals_missing=False,
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "Close": [100.0, 150.0]})


def _patch_screen(monkeypatch, candidates):
    monkeypatch.setattr(multibagger_job, "screen_trend_candidates", lambda **_: [])
    monkeypatch.setattr(multibagger_job, "apply_quality_gate", lambda _trend: candidates)


def _patch_prices(monkeypatch):
    monkeypatch.setattr(multibagger_job, "load_stock_features", lambda *_: _prices())


def _patch_sim(monkeypatch, events):
    monkeypatch.setattr(multibagger_job, "simulate_position", lambda *_a, **_k: events)


def _patch_notify(monkeypatch):
    """send_webhook_text_chunked をモックし、呼び出しを記録する。"""
    calls = []
    import src.reporting.discord.discord_utils as discord_utils

    monkeypatch.setattr(
        discord_utils, "send_webhook_text_chunked", lambda msg, **_k: calls.append(msg) or True
    )
    return calls


# ── positions ラウンドトリップ ───────────────────────────────


def test_positions_roundtrip(tmp_positions):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    add_position("us", "MSFT", "2024-01-02", 200.0)

    opens = list_positions(status="open")
    assert {p.symbol for p in opens} == {"AAPL", "MSFT"}

    updated = update_position("us", "AAPL", held_fraction=0.8, last_action="scale_out")
    assert updated is not None and updated.held_fraction == 0.8
    assert updated.last_action == "scale_out"

    closed = close_position("us", "MSFT", close_price=400.0, reason="trail_stop")
    assert closed is not None and closed.status == "closed"
    assert closed.last_multiple == pytest.approx(2.0)

    assert {p.symbol for p in list_positions(status="open")} == {"AAPL"}
    assert {p.symbol for p in list_positions(status="closed")} == {"MSFT"}


def test_add_position_duplicate_is_noop(tmp_positions):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    add_position("us", "AAPL", "2024-01-05", 110.0)
    opens = list_positions(status="open")
    assert len(opens) == 1
    assert opens[0].entry_price == 100.0


# ── ジョブ本体 ──────────────────────────────────────────────


def test_screen_includes_new_candidates_and_hold_action(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [_candidate("NVDA")])
    _patch_prices(monkeypatch)
    # 末尾 end_of_data = 未撤退 → HOLD
    _patch_sim(
        monkeypatch,
        [
            PositionEvent("2024-01-01", "entry", 100.0, 1.0, "entry", 1.0),
            PositionEvent("2024-02-01", "exit", 150.0, 0.0, "end_of_data", 1.5),
        ],
    )
    calls = _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert [c.symbol for c in payload["new_candidates"]] == ["NVDA"]
    assert len(payload["position_actions"]) == 1
    action = payload["position_actions"][0]
    assert action["symbol"] == "AAPL"
    assert action["action"] == "HOLD"
    assert "NVDA" in payload["message"]
    assert "AAPL" in payload["message"]
    assert len(calls) == 1
    # HOLD なので open のまま
    assert list_positions(status="open")[0].symbol == "AAPL"


def test_exit_signal_closes_position(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    _patch_prices(monkeypatch)
    _patch_sim(
        monkeypatch,
        [
            PositionEvent("2024-01-01", "entry", 100.0, 1.0, "entry", 1.0),
            PositionEvent("2024-02-01", "exit", 80.0, 0.0, "ma_break", 0.8),
        ],
    )
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    action = payload["position_actions"][0]
    assert action["action"] == "EXIT"
    assert action["reason"] == "ma_break"
    assert list_positions(status="open") == []
    closed = list_positions(status="closed")
    assert len(closed) == 1 and closed[0].symbol == "AAPL"


def test_scale_out_signal_updates_held_fraction(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    _patch_prices(monkeypatch)
    _patch_sim(
        monkeypatch,
        [
            PositionEvent("2024-01-01", "entry", 100.0, 1.0, "entry", 1.0),
            PositionEvent("2024-02-01", "scale_out", 200.0, 0.8, "scale_2x", 2.0),
            PositionEvent("2024-03-01", "exit", 250.0, 0.0, "end_of_data", 2.5),
        ],
    )
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    action = payload["position_actions"][0]
    assert action["action"] == "SCALE_OUT"
    pos = list_positions(status="open")[0]
    assert pos.held_fraction == 0.8
    assert pos.last_action == "scale_out"


def test_discord_failure_does_not_crash(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [_candidate("NVDA")])
    _patch_prices(monkeypatch)
    _patch_sim(
        monkeypatch,
        [PositionEvent("2024-01-01", "entry", 100.0, 1.0, "entry", 1.0)],
    )

    import src.reporting.discord.discord_utils as discord_utils

    def _boom(*_a, **_k):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(discord_utils, "send_webhook_text_chunked", _boom)

    # 例外を送出せず正常終了すること
    payload = multibagger_job.run_monthly_multibagger_screen(market="us")
    assert payload["new_candidates"]


# ── 判定できなかった保有・スクリーン失敗（#872）────────────────


def test_missing_prices_reports_skipped_not_no_holdings(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    monkeypatch.setattr(multibagger_job, "load_stock_features", lambda *_: None)
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["position_actions"] == []
    assert [s["symbol"] for s in payload["skipped_positions"]] == ["AAPL"]
    message = payload["message"]
    assert "保有なし" not in message
    assert "判定できず" in message
    assert "AAPL" in message


def test_evaluation_exception_reports_skipped(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    _patch_prices(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("sim failed")

    monkeypatch.setattr(multibagger_job, "simulate_position", _boom)
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert [s["symbol"] for s in payload["skipped_positions"]] == ["AAPL"]
    message = payload["message"]
    assert "保有なし" not in message
    assert "判定できず" in message
    assert "AAPL" in message


def test_no_events_reports_skipped(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    _patch_prices(monkeypatch)
    _patch_sim(monkeypatch, [])
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    message = payload["message"]
    assert "保有なし" not in message
    assert "判定できず" in message
    assert "AAPL" in message


def test_no_open_positions_still_says_no_holdings(tmp_positions, monkeypatch):
    _patch_screen(monkeypatch, [])
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["skipped_positions"] == []
    assert "保有なし" in payload["message"]
    assert "判定できず" not in payload["message"]


# ── 読めない台帳（#885）──────────────────────────────────────

_BROKEN_LEDGER = '[{"market": "us", "symbol": "AAPL" "entry_date": "2024-01-01"}]'


def test_list_positions_raises_on_broken_json(tmp_positions):
    tmp_positions.write_text(_BROKEN_LEDGER, encoding="utf-8")

    with pytest.raises(positions_mod.PositionLedgerError):
        list_positions(status="open")


def test_list_positions_raises_on_non_list_json(tmp_positions):
    tmp_positions.write_text('{"market": "us"}', encoding="utf-8")

    with pytest.raises(positions_mod.PositionLedgerError):
        list_positions()


def test_job_fails_on_broken_ledger_without_no_holdings_notice(tmp_positions, monkeypatch):
    tmp_positions.write_text(_BROKEN_LEDGER, encoding="utf-8")
    _patch_screen(monkeypatch, [])
    calls = _patch_notify(monkeypatch)

    with pytest.raises(positions_mod.PositionLedgerError):
        multibagger_job.run_monthly_multibagger_screen(market="us")

    assert not any("保有なし" in c for c in calls)


def test_add_position_does_not_overwrite_broken_ledger(tmp_positions):
    tmp_positions.write_text(_BROKEN_LEDGER, encoding="utf-8")

    with pytest.raises(positions_mod.PositionLedgerError):
        add_position("us", "MSFT", "2024-01-02", 200.0)

    assert tmp_positions.read_text(encoding="utf-8") == _BROKEN_LEDGER


def test_missing_ledger_is_empty(tmp_positions):
    assert not tmp_positions.exists()
    assert list_positions() == []


def test_screen_exception_is_not_reported_as_no_candidates(tmp_positions, monkeypatch):
    def _boom(**_k):
        raise RuntimeError("screen down")

    monkeypatch.setattr(multibagger_job, "screen_trend_candidates", _boom)
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["new_candidates"] == []
    assert payload["screen_failed"] is True
    message = payload["message"]
    assert "該当なし" not in message
    assert "スクリーン失敗" in message


# ── 質ゲートの前の財務の更新（#1557）──────────────────────────

_PASSING_INFO = {
    "totalRevenue": 400.0,
    "returnOnEquity": 0.25,
    "debtToEquity": 50.0,
    "marketCap": 5e9,
}
_FAILING_INFO = {**_PASSING_INFO, "returnOnEquity": 0.01}


def _trend(symbol: str, score: float = 1.0) -> TrendCandidate:
    return TrendCandidate(
        market="us",
        symbol=symbol,
        score=score,
        close=120.0,
        dist_from_52w_high=-0.01,
        above_200dma=True,
        sma200_rising=True,
        return_6m=0.3,
        return_12m=0.5,
        avg_volume=1e6,
    )


def _yf_ticker(info: dict | None) -> MagicMock:
    """yf.Ticker の差し替え。info が None なら何も取れない銘柄を表す。"""
    ticker = MagicMock()
    ticker.info = info or {}
    periods = pd.to_datetime(["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"])
    revenue = [400.0, 300.0, 200.0, 100.0]
    ticker.financials = (
        pd.DataFrame([revenue], index=["Total Revenue"], columns=periods)
        if info is not None
        else pd.DataFrame()
    )
    ticker.income_stmt = pd.DataFrame()
    ticker.balance_sheet = pd.DataFrame()
    return ticker


class _FundamentalsWorld:
    """yfinance の答えと stock_fundamentals を手元の dict に置き換える。"""

    def __init__(self, monkeypatch, trend, infos: dict, stored=None):
        self.stored: dict[tuple[str, str], dict] = dict(stored or {})
        self.events: list[str] = []
        real_gate = multibagger_job.apply_quality_gate

        def _upsert(record):
            self.events.append(f"upsert:{record.symbol}")
            self.stored[(record.market, record.symbol)] = dict(record.__dict__)

        def _load(market, symbol):
            return self.stored.get((market, symbol))

        def _gate(candidates):
            self.events.append("gate")
            return real_gate(candidates)

        monkeypatch.setattr(multibagger_job, "screen_trend_candidates", lambda **_: trend)
        monkeypatch.setattr(multibagger_job, "apply_quality_gate", _gate)
        monkeypatch.setattr(multibagger_job, "upsert_fundamentals", _upsert, raising=False)
        monkeypatch.setattr(multibagger_job, "load_fundamentals", _load, raising=False)
        monkeypatch.setattr(quality_gate, "load_fundamentals", _load)
        monkeypatch.setattr(fundamentals.yf, "Ticker", lambda ticker: _yf_ticker(infos.get(ticker)))


def test_fundamentals_are_saved_before_quality_gate(tmp_positions, monkeypatch):
    world = _FundamentalsWorld(
        monkeypatch,
        [_trend("AAA"), _trend("BBB", 0.5)],
        {"AAA": _PASSING_INFO, "BBB": _PASSING_INFO},
    )
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert world.events == ["upsert:AAA", "upsert:BBB", "gate"]
    assert {c.symbol for c in payload["new_candidates"]} == {"AAA", "BBB"}
    assert payload["fundamentals_failed"] == 0
    assert "財務を取れず" not in payload["message"]


def test_one_fetch_none_keeps_other_candidates_and_positions(tmp_positions, monkeypatch):
    add_position("us", "POS", "2024-01-01", 100.0)
    _FundamentalsWorld(
        monkeypatch, [_trend("BAD"), _trend("AAA")], {"BAD": None, "AAA": _PASSING_INFO}
    )
    _patch_prices(monkeypatch)
    _patch_sim(
        monkeypatch,
        [
            PositionEvent("2024-01-01", "entry", 100.0, 1.0, "entry", 1.0),
            PositionEvent("2024-02-01", "exit", 150.0, 0.0, "end_of_data", 1.5),
        ],
    )
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert [c.symbol for c in payload["new_candidates"]] == ["AAA"]
    assert [a["symbol"] for a in payload["position_actions"]] == ["POS"]
    assert payload["fundamentals_failed"] == 1
    assert "財務を取れず: 1/2 件" in payload["message"]


def test_save_error_on_one_symbol_keeps_the_others(tmp_positions, monkeypatch):
    world = _FundamentalsWorld(
        monkeypatch, [_trend("BAD"), _trend("AAA")], {"BAD": _PASSING_INFO, "AAA": _PASSING_INFO}
    )
    save_to_world = multibagger_job.upsert_fundamentals

    def _upsert(record):
        if record.symbol == "BAD":
            raise RuntimeError("db down")
        save_to_world(record)

    monkeypatch.setattr(multibagger_job, "upsert_fundamentals", _upsert)
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert world.events == ["upsert:AAA", "gate"]
    assert [c.symbol for c in payload["new_candidates"]] == ["AAA"]
    assert payload["screen_failed"] is False
    assert "財務を取れず: 1/2 件" in payload["message"]


def test_all_candidates_without_fundamentals_reports_unjudged_count(tmp_positions, monkeypatch):
    _FundamentalsWorld(monkeypatch, [_trend("AAA"), _trend("BBB")], {})
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["new_candidates"] == []
    assert payload["screen_failed"] is False
    assert payload["fundamentals_missing"] == 2
    message = payload["message"]
    assert "該当なし" not in message
    assert "財務が無く判定できず: 2/2 件" in message


def test_candidates_failing_the_gate_still_say_no_match(tmp_positions, monkeypatch):
    _FundamentalsWorld(
        monkeypatch,
        [_trend("AAA"), _trend("BBB")],
        {"AAA": _FAILING_INFO, "BBB": _FAILING_INFO},
    )
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["new_candidates"] == []
    message = payload["message"]
    assert "該当なし" in message
    assert "財務" not in message


def test_no_trend_candidates_still_says_no_match(tmp_positions, monkeypatch):
    _FundamentalsWorld(monkeypatch, [], {})
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert "該当なし" in payload["message"]
    assert "財務" not in payload["message"]


def test_failed_fetch_with_old_row_is_judged_but_counted(tmp_positions, monkeypatch):
    """取得に失敗しても前の行があれば判定はでき、取れなかった数だけを載せる。"""
    old_row = {"revenue_cagr_3y": 0.01, "roe": 0.01, "debt_to_equity": 50.0, "market_cap": 5e9}
    _FundamentalsWorld(monkeypatch, [_trend("OLD")], {}, stored={("us", "OLD"): old_row})
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["fundamentals_failed"] == 1
    assert payload["fundamentals_missing"] == 0
    message = payload["message"]
    assert "該当なし" in message
    assert "財務を取れず: 1/1 件" in message


# ── ジョブの成否と同じ月の補完（#1596）──────────────────────────

_EXIT_EVENTS = [
    PositionEvent("2024-01-01", "entry", 100.0, 1.0, "entry", 1.0),
    PositionEvent("2024-02-01", "exit", 80.0, 0.0, "ma_break", 0.8),
]


def _patch_notify_false(monkeypatch):
    """send_webhook_text_chunked が False を返す（送れなかった）形にする。"""
    import src.reporting.discord.discord_utils as discord_utils

    monkeypatch.setattr(discord_utils, "send_webhook_text_chunked", lambda *_a, **_k: False)


def _patch_screen_failure(monkeypatch):
    def _boom(**_k):
        raise RuntimeError("screen down")

    monkeypatch.setattr(multibagger_job, "screen_trend_candidates", _boom)
    _patch_notify(monkeypatch)


def _patch_all_fundamentals_missing(monkeypatch):
    _FundamentalsWorld(monkeypatch, [_trend("AAA"), _trend("BBB")], {})
    _patch_notify(monkeypatch)


def _patch_skipped_position(monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    monkeypatch.setattr(multibagger_job, "load_stock_features", lambda *_: None)
    _patch_notify(monkeypatch)


def _patch_send_false(monkeypatch):
    _patch_screen(monkeypatch, [_candidate("NVDA")])
    _patch_notify_false(monkeypatch)


_FAILED_RUNS = [
    _patch_screen_failure,
    _patch_all_fundamentals_missing,
    _patch_skipped_position,
    _patch_send_false,
]


def _queue_manager(tmp_path) -> SchedulerQueueManager:
    """本番の SCHEDULE_CONFIG の設定のままのキューを作る。"""
    from run_scheduler import SCHEDULE_CONFIG

    return SchedulerQueueManager(
        {"monthly_multibagger": dict(SCHEDULE_CONFIG["monthly_multibagger"])},
        state_file_path=str(tmp_path / "queue_state.json"),
        error_notifier=MagicMock(),
    )


def _run_job(manager: SchedulerQueueManager, now: datetime) -> None:
    """monthly_multibagger を now の回として走らせ、成功の heartbeat を確かめる。"""
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat") as ping,
        patch.object(manager, "now", return_value=now),
    ):
        manager.run_job("monthly_multibagger", reason="recovery")
    ping.assert_called_once_with("monthly_multibagger", success=True)


def _failed_run(manager: SchedulerQueueManager, now: datetime) -> None:
    """失敗の回として走らせ、やり直しの待ちが無く heartbeat が失敗を送ることを確かめる。"""
    func = MagicMock(side_effect=manager.schedule_config["monthly_multibagger"]["func"])
    manager.schedule_config["monthly_multibagger"]["func"] = func
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat") as ping,
        patch("src.orchestration.scheduler_queue.time.sleep") as sleep,
        patch.object(manager, "now", return_value=now),
    ):
        with pytest.raises(NonRetryableJobError):
            manager.run_job("monthly_multibagger", reason="recovery")
    func.assert_called_once_with()
    sleep.assert_not_called()
    ping.assert_called_once_with("monthly_multibagger", success=False)


def _statuses(manager: SchedulerQueueManager) -> list[str]:
    return [e["status"] for e in manager._state["events"] if e["status"] != "started"]


@pytest.mark.parametrize("arrange", _FAILED_RUNS)
def test_failed_run_raises_without_immediate_retry(tmp_positions, tmp_path, monkeypatch, arrange):
    arrange(monkeypatch)
    manager = _queue_manager(tmp_path)

    _failed_run(manager, datetime(2026, 11, 1, 4, 0, tzinfo=manager.tz))

    assert _statuses(manager) == ["error"]


def test_failed_runs_recover_on_other_days_of_the_month_up_to_three(
    tmp_positions, tmp_path, monkeypatch
):
    _patch_screen_failure(monkeypatch)
    manager = _queue_manager(tmp_path)
    first = datetime(2026, 11, 1, 4, 0, tzinfo=manager.tz)
    other_day = datetime(2026, 11, 20, 12, 0, tzinfo=manager.tz)

    _failed_run(manager, first)
    assert manager.should_recover("monthly_multibagger", now=other_day) is True

    _failed_run(manager, other_day)
    next_day = other_day + timedelta(days=1)
    assert manager.should_recover("monthly_multibagger", now=next_day) is True

    _failed_run(manager, next_day)
    assert manager.should_recover("monthly_multibagger", now=next_day + timedelta(days=1)) is False


def test_recovery_run_succeeds_after_failed_run(tmp_positions, tmp_path, monkeypatch):
    _patch_screen_failure(monkeypatch)
    manager = _queue_manager(tmp_path)
    now = datetime(2026, 11, 1, 5, 30, tzinfo=manager.tz)
    _failed_run(manager, now)

    _patch_screen(monkeypatch, [_candidate("NVDA")])
    _run_job(manager, now)

    assert _statuses(manager) == ["error", "success"]
    assert manager.should_recover("monthly_multibagger", now=now) is False


def test_no_holdings_and_no_match_is_success(tmp_positions, tmp_path, monkeypatch):
    _patch_screen(monkeypatch, [])
    calls = _patch_notify(monkeypatch)
    manager = _queue_manager(tmp_path)

    _run_job(manager, datetime(2026, 11, 1, 4, 0, tzinfo=manager.tz))

    assert _statuses(manager) == ["success"]
    assert "該当なし" in calls[0]
    assert "保有なし" in calls[0]


def test_candidates_failing_the_gate_are_not_a_failed_run(tmp_positions, monkeypatch):
    """財務の一部が取れない回や質ゲートで落ちただけの回は失敗にしない。"""
    _FundamentalsWorld(
        monkeypatch, [_trend("BAD"), _trend("AAA")], {"BAD": None, "AAA": _FAILING_INFO}
    )
    _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["fundamentals_failed"] == 1
    multibagger_job.raise_if_multibagger_failed(payload)


def test_unreadable_screen_is_not_reported_as_no_match(tmp_positions, monkeypatch):
    """全銘柄の値段が読めない回は「該当なし」と書かず、ジョブを失敗にする。"""

    def _boom(**_k):
        raise ScreeningDataError("全 543 銘柄の値段を読めず (us)")

    monkeypatch.setattr(multibagger_job, "screen_trend_candidates", _boom)
    calls = _patch_notify(monkeypatch)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert "該当なし" not in calls[0]
    assert "スクリーン失敗" in calls[0]
    with pytest.raises(NonRetryableJobError, match="スクリーンの失敗"):
        multibagger_job.raise_if_multibagger_failed(payload)


def test_notify_exception_counts_as_not_sent(tmp_positions, monkeypatch):
    _patch_screen(monkeypatch, [])
    import src.reporting.discord.discord_utils as discord_utils

    def _boom(*_a, **_k):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(discord_utils, "send_webhook_text_chunked", _boom)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["notified"] is False
    with pytest.raises(NonRetryableJobError, match="知らせの送信の失敗"):
        multibagger_job.raise_if_multibagger_failed(payload)


def test_exit_stays_open_until_notified_and_is_sent_again(tmp_positions, monkeypatch):
    """EXIT の知らせが送れなかった回は保有を開いたまま残し、次の回でもう一度知らせて閉じる。"""
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    _patch_prices(monkeypatch)
    _patch_sim(monkeypatch, _EXIT_EVENTS)
    _patch_notify_false(monkeypatch)

    failed = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert failed["position_actions"][0]["action"] == "EXIT"
    assert [p.symbol for p in list_positions(status="open")] == ["AAPL"]

    calls = _patch_notify(monkeypatch)
    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert "EXIT" in calls[0]
    assert list_positions(status="open") == []
    assert [p.symbol for p in list_positions(status="closed")] == ["AAPL"]
    multibagger_job.raise_if_multibagger_failed(payload)


def test_close_failure_fails_the_run(tmp_positions, monkeypatch):
    add_position("us", "AAPL", "2024-01-01", 100.0)
    _patch_screen(monkeypatch, [])
    _patch_prices(monkeypatch)
    _patch_sim(monkeypatch, _EXIT_EVENTS)
    _patch_notify(monkeypatch)

    def _boom(*_a, **_k):
        raise OSError("ledger down")

    monkeypatch.setattr(multibagger_job, "close_position", _boom)

    payload = multibagger_job.run_monthly_multibagger_screen(market="us")

    assert payload["close_failed"] == ["AAPL"]
    with pytest.raises(NonRetryableJobError, match="閉じられなかった EXIT"):
        multibagger_job.raise_if_multibagger_failed(payload)
