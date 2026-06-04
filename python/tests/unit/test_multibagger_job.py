"""月次 multibagger ジョブ + ポジション永続化の単体テスト（#434）"""

import pandas as pd
import pytest

from src.orchestration import multibagger_job
from src.screening import positions as positions_mod
from src.screening.positions import add_position, close_position, list_positions, update_position
from src.screening.types import MultibaggerCandidate, PositionEvent


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
