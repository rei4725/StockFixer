"""朝の目線の Claude の道（#1875 段4、#1956）の、頼みの文と CLI の出力の読み取り。

CLI の呼び出しそのものは持たず、stream-json の行を返す run を引数で受ける。
ニュースの URL は、同じ呼び出しの中で道具（WebSearch / WebFetch）の結果に
現れた URL と一致するものだけ残す。読めない・失敗したときは None を返し、
ジョブは数字と先物の向きだけの告知（source=fallback）にする。
"""

import json
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from src.reporting.morning_view import INDEX_LABELS, ClaudeView, IndexMove, NewsItem
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 頼みの文を受け取り、CLI の stream-json の出力の行を返す道
StreamRunner = Callable[[str], Iterable[str]]

_URL_RE = re.compile(r"https?://[^\s\"'<>\]\[(){}\\,]+")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

SYSTEM_PROMPT = (
    "あなたは東京市場の寄り付き前の目線を書く。WebSearch で夜の米国市場と日本株に効く"
    "主要なニュースを調べ、指定の JSON だけを返す。ニュースの url には、検索の結果に"
    "実際に出た URL だけを書く。"
)


def build_prompt(moves: Sequence[IndexMove]) -> str:
    lines = [
        "夜の値動き（yfinance）:",
        *(
            f"- {m.symbol}（{INDEX_LABELS.get(m.symbol, m.symbol)}）: "
            f"{m.last:.2f}（{m.change_pct:+.2f}%）"
            for m in moves
        ),
        "",
        "今日の日経平均の目線を JSON 1 つで返せ（コードフェンス・前置きは不要）:",
        '{"view": "up|down", "reason": "理由", "linkage": "日本株への連動の見込み",'
        ' "news": [{"title": "題", "direction": "up|down", "url": "記事の URL"}],'
        ' "index_directions": {"文中で触れた上の記号": "up|down"}}',
    ]
    return "\n".join(lines)


def _urls_in(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return {u.rstrip(".") for u in _URL_RE.findall(text)}


def _tool_result_urls(event: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    if "tool_use_result" in event:
        urls |= _urls_in(event["tool_use_result"])
    content = (event.get("message") or {}).get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                urls |= _urls_in(block.get("content"))
    return urls


def parse_stream(lines: Iterable[str]) -> tuple[str | None, set[str]]:
    """(最後の result の文、道具の結果に現れた URL)。失敗の result は文を None にする。"""
    result: str | None = None
    urls: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "user":
            urls |= _tool_result_urls(event)
        elif event.get("type") == "result":
            result = None if event.get("is_error") else str(event.get("result", ""))
    return result, urls


def parse_view(text: str, tool_urls: set[str]) -> ClaudeView | None:
    """result の JSON を ClaudeView にする。道具が返さなかった URL のニュースは捨てる。"""
    try:
        data = json.loads(_FENCE_RE.sub("", text).strip())
    except json.JSONDecodeError:
        logger.warning("朝の目線: Claude の結果が JSON として読めない")
        return None
    if not isinstance(data, dict):
        return None
    news: list[NewsItem] = []
    for item in data.get("news") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", ""))
        if url not in tool_urls:
            logger.info("朝の目線: 道具が返さなかった URL を捨てた %s", url)
            continue
        news.append(NewsItem(str(item.get("title", "")), str(item.get("direction", "")), url))
    directions = data.get("index_directions") or {}
    return ClaudeView(
        view=str(data.get("view", "")),
        reason=str(data.get("reason", "")),
        linkage=str(data.get("linkage", "")),
        news=news,
        index_directions={str(k): str(v) for k, v in dict(directions).items()},
    )


def ask_claude_view(moves: Sequence[IndexMove], run: StreamRunner) -> ClaudeView | None:
    """CLI に目線を頼む。呼び出しの失敗・読めない結果は None（数字だけの告知になる）。"""
    try:
        text, urls = parse_stream(run(build_prompt(moves)))
    except Exception:  # pylint: disable=broad-exception-caught
        logger.error("朝の目線: Claude の CLI の呼び出しに失敗した", exc_info=True)
        return None
    if text is None:
        logger.warning("朝の目線: Claude の CLI が結果を返さなかった")
        return None
    return parse_view(text, urls)
