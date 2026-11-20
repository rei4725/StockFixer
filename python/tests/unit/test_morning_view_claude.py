"""朝の目線の Claude の道（#1956 段4）を作り物の CLI の出力で確かめる。"""

import json
from datetime import date

from src.market_data.morning_quotes import CloseQuote
from src.orchestration.jobs import morning_view_job
from src.reporting.morning_view import IndexMove
from src.reporting.morning_view_claude import ask_claude_view, parse_stream

TOOL_URL = "https://news.example.com/a"
FAKE_URL = "https://made-up.example.com/b"
DAY = date(2026, 11, 20)


def _result_json(directions: dict[str, str]) -> str:
    return json.dumps(
        {
            "view": "up",
            "reason": "米株高",
            "linkage": "半導体が連動",
            "news": [
                {"title": "本物", "direction": "up", "url": TOOL_URL},
                {"title": "作り物", "direction": "down", "url": FAKE_URL},
            ],
            "index_directions": directions,
        },
        ensure_ascii=False,
    )


def _stream(result_text: str, is_error: bool = False) -> list[str]:
    events = [
        {"type": "system", "subtype": "init"},
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": f"Links: [{{'url': '{TOOL_URL}'}}]"}],
                    }
                ]
            },
        },
        {"type": "assistant", "message": {"content": [{"type": "text", "text": FAKE_URL}]}},
        {"type": "result", "is_error": is_error, "result": result_text},
    ]
    return [json.dumps(e) for e in events]


MOVES = [IndexMove("^GSPC", 110.0, 100.0), IndexMove("^N225", 100.0, 99.0)]


def test_parse_stream_takes_urls_only_from_tool_results():
    text, urls = parse_stream(_stream("x"))
    assert text == "x"
    assert urls == {TOOL_URL}


def test_ask_drops_url_the_tool_did_not_return():
    view = ask_claude_view(MOVES, run=lambda prompt: _stream(_result_json({"^GSPC": "up"})))
    assert view is not None
    assert [n.url for n in view.news] == [TOOL_URL]


def test_ask_returns_none_when_cli_fails():
    def boom(prompt: str) -> list[str]:
        raise RuntimeError("no cli")

    assert ask_claude_view(MOVES, run=boom) is None
    assert ask_claude_view(MOVES, run=lambda p: _stream("x", is_error=True)) is None
    assert ask_claude_view(MOVES, run=lambda p: _stream("not json")) is None


def _quotes() -> list[CloseQuote]:
    return [
        CloseQuote("^GSPC", 110.0, 100.0, DAY),
        CloseQuote("^N225", 100.0, 99.0, date(2026, 11, 19)),
        CloseQuote("NIY=F", 90.0, 99.0, DAY),
    ]


def test_announcement_uses_claude_and_keeps_only_tool_url():
    ask = lambda moves: ask_claude_view(  # noqa: E731
        moves, run=lambda p: _stream(_result_json({"^GSPC": "up"}))
    )
    a = morning_view_job.build_announcement(DAY, _quotes(), [], ask)
    assert a is not None
    assert a.record.source == "claude"
    assert a.record.view == "up"
    assert TOOL_URL in a.text
    assert FAKE_URL not in a.text


def test_direction_mismatch_falls_back():
    ask = lambda moves: ask_claude_view(  # noqa: E731
        moves, run=lambda p: _stream(_result_json({"^GSPC": "down"}))
    )
    a = morning_view_job.build_announcement(DAY, _quotes(), [], ask)
    assert a is not None
    assert a.record.source == "fallback"
    # 先物 90 < 前日の終値 100 なので下げ
    assert a.record.view == "down"
    assert "ニュースの取得に失敗" in a.text
    assert TOOL_URL not in a.text


def test_cli_failure_gives_numbers_only():
    def boom(prompt: str) -> list[str]:
        raise OSError("claude not found")

    ask = lambda moves: ask_claude_view(moves, run=boom)  # noqa: E731
    a = morning_view_job.build_announcement(DAY, _quotes(), [], ask)
    assert a is not None
    assert a.record.source == "fallback"
    assert "ニュースの取得に失敗" in a.text
    assert "うち先物だけの日 0" in a.text


def test_claude_ask_off_by_default(monkeypatch):
    from config.feature_flags import flags

    monkeypatch.setattr(flags, "MORNING_VIEW_CLAUDE_ENABLED", False)
    assert morning_view_job.claude_ask() is None
    monkeypatch.setattr(flags, "MORNING_VIEW_CLAUDE_ENABLED", True)
    assert morning_view_job.claude_ask() is not None
