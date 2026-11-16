"""朝の目線の告知（reporting/morning_view.py）の純な関数のテスト。"""

from datetime import date

from src.reporting.morning_view import (
    ClaudeView,
    IndexMove,
    NewsItem,
    ViewRecord,
    build_morning_text,
    choose_view,
    futures_view,
    is_consistent,
    judge_outcome,
    tally,
)


def _moves(spx_up: bool = True) -> list[IndexMove]:
    return [
        IndexMove("^GSPC", 5000.0 if spx_up else 4900.0, 4950.0),
        IndexMove("^SOX", 5100.0, 5000.0),
        IndexMove("NIY=F", 39100.0, 39000.0),
        IndexMove("^N225", 38900.0, 38800.0),
    ]


def _record(i: int, view: str, actual: float, source: str = "claude") -> ViewRecord:
    return ViewRecord(
        day=date(2026, 10, 1 + i),
        view=view,
        source=source,
        futures_view="up",
        actual_change_pct=actual,
    )


def test_futures_view_above_is_up_equal_is_down() -> None:
    assert futures_view(39100.0, 39000.0) == "up"
    assert futures_view(39000.0, 39000.0) == "down"
    assert futures_view(38900.0, 39000.0) == "down"


def test_judge_outcome_draw_is_none() -> None:
    assert judge_outcome(100.0, 101.0) == "up"
    assert judge_outcome(100.0, 99.0) == "down"
    assert judge_outcome(100.0, 100.0) is None


def test_tally_excludes_draws_and_counts_baselines() -> None:
    records = [
        _record(0, "up", 1.0),
        _record(1, "down", 1.0, source="fallback"),
        _record(2, "down", -0.5),
        _record(3, "up", 0.0),  # 引き分け
        ViewRecord(date(2026, 10, 9), "up", "claude", "up", None),  # 未答え合わせ
    ]
    t = tally(records)
    assert (t.view.wins, t.view.losses) == (2, 1)
    assert (t.always_up.wins, t.always_up.losses) == (2, 1)
    assert (t.futures.wins, t.futures.losses) == (2, 1)
    assert t.fallback_days == 1
    assert t.announced_days == 5
    assert t.judged == 3


def test_tally_recent_uses_last_20_judged() -> None:
    records = [_record(i, "down", 1.0) for i in range(5)]
    records += [_record(5 + i, "up", 1.0) for i in range(20)]
    t = tally(records)
    assert (t.recent_view.wins, t.recent_view.losses) == (20, 0)
    assert (t.view.wins, t.view.losses) == (20, 5)


def test_is_consistent_detects_contradiction() -> None:
    claude = ClaudeView("up", "米国株高", "連動しやすい", [], {"^GSPC": "up"})
    assert is_consistent(claude, _moves(spx_up=True))
    assert not is_consistent(claude, _moves(spx_up=False))
    unknown = ClaudeView("up", "x", "y", [], {"^XXX": "up"})
    assert not is_consistent(unknown, _moves())


def test_choose_view_falls_back_on_none_or_contradiction() -> None:
    moves = _moves(spx_up=False)
    claude = ClaudeView("down", "理由", "連動", [], {"^GSPC": "up"})
    view, source, used = choose_view(claude, moves, 38800.0)
    assert (view, source, used) == ("up", "fallback", None)
    view, source, used = choose_view(None, moves, 38800.0)
    assert (source, used) == ("fallback", None)
    ok = ClaudeView("down", "理由", "連動", [], {"^GSPC": "down"})
    view, source, used = choose_view(ok, moves, 38800.0)
    assert (view, source, used) == ("down", "claude", ok)


def test_build_text_with_claude_and_verified_urls() -> None:
    news = [
        NewsItem("半導体高", "up", "https://ex.com/a"),
        NewsItem("作り話", "down", "https://ex.com/fake"),
    ]
    claude = ClaudeView("up", "半導体が主導", "東エレク中心に連動", news, {"^SOX": "up"})
    records = [_record(i, "up", 1.0) for i in range(3)]
    text = build_morning_text(
        _moves(), 38800.0, claude, records, verified_urls={"https://ex.com/a"}
    )
    assert "検証中（4 日目）" in text
    assert "上げ目線" in text
    assert "半導体が主導" in text
    assert "S&P500" in text and "+1.01%" in text
    assert "日経先物" in text and "+300" in text
    assert "https://ex.com/a" in text
    assert "作り話" not in text
    assert "3勝0敗（100%）" in text
    assert "いつも上げ" in text and "先物の向き" in text
    assert "うち先物だけの日 0" in text


def test_build_text_without_claude_uses_futures_only() -> None:
    text = build_morning_text(_moves(), 38800.0, None, [], verified_urls=set())
    assert "上げ目線" in text
    assert "ニュースの取得に失敗" in text
    assert "先物の向き" in text
    assert "検証中（1 日目）" in text
    assert "0勝0敗（-）" in text


def test_build_text_no_verification_heading_after_20_judged() -> None:
    records = [_record(i, "up", 1.0, source="fallback") for i in range(20)]
    text = build_morning_text(_moves(), 38800.0, None, records, verified_urls=set())
    assert "検証中" not in text
    assert "うち先物だけの日 20" in text
