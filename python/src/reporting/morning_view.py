"""朝 8:30 の目線の告知の、本文の組み立てと答え合わせ・当たり率の純な関数（#1875 段1）。

取得・Claude の呼び出し・表・ジョブは持たない。値はすべて引数で受け取る。
目線の向きは "up"（上げ）か "down"（下げ）。答え合わせは日経平均の前日の終値比で、
同じ値は引き分けとして当たり率から外す。
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from src.utils.logger import get_logger

logger = get_logger(__name__)

FUTURES_SYMBOL = "NIY=F"
RECENT_WINDOW = 20

INDEX_LABELS: dict[str, str] = {
    "^GSPC": "S&P500",
    "^IXIC": "Nasdaq総合",
    "^DJI": "ダウ",
    "^SOX": "SOX",
    "^VIX": "VIX",
    "^TNX": "米10年金利",
    "JPY=X": "ドル円",
    "NIY=F": "日経先物（円建て）",
    "^N225": "日経平均",
}

_DIRECTION_LABELS = {"up": "上げ", "down": "下げ"}


@dataclass(frozen=True)
class IndexMove:
    """指数ごとの夜の騰落（yfinance の値そのまま）。"""

    symbol: str
    last: float
    prev_close: float

    @property
    def change(self) -> float:
        return self.last - self.prev_close

    @property
    def change_pct(self) -> float:
        if self.prev_close == 0:
            return 0.0
        return self.change / self.prev_close * 100.0


@dataclass(frozen=True)
class NewsItem:
    """Claude が返したニュース 1 件（題・材料の向き・URL）。"""

    title: str
    direction: str
    url: str


@dataclass(frozen=True)
class ClaudeView:
    """Claude の結果。index_directions は文中で触れた指数ごとの向き。"""

    view: str
    reason: str
    linkage: str
    news: list[NewsItem] = field(default_factory=list)
    index_directions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewRecord:
    """告知した 1 日分。actual_change_pct が None なら未答え合わせ。"""

    day: date
    view: str
    source: str
    futures_view: str
    actual_change_pct: float | None


@dataclass(frozen=True)
class WinLoss:
    wins: int = 0
    losses: int = 0

    def text(self) -> str:
        total = self.wins + self.losses
        rate = f"{self.wins / total * 100:.0f}%" if total else "-"
        return f"{self.wins}勝{self.losses}敗（{rate}）"


@dataclass(frozen=True)
class Tally:
    """全期間と直近の当たり率、2つの物差し、日数。"""

    view: WinLoss
    always_up: WinLoss
    futures: WinLoss
    recent_view: WinLoss
    recent_always_up: WinLoss
    recent_futures: WinLoss
    announced_days: int
    fallback_days: int
    judged: int


def futures_view(futures_price: float, n225_prev_close: float) -> str:
    """先物の向きの目線。前日の終値より上なら上げ、同じ値と下は下げ。"""
    return "up" if futures_price > n225_prev_close else "down"


def judge_outcome(prev_close: float, close: float) -> str | None:
    """日経平均の前日の終値比の答え。同じ値は引き分けで None。"""
    if close > prev_close:
        return "up"
    if close < prev_close:
        return "down"
    return None


def _outcome_of(record: ViewRecord) -> str | None:
    if record.actual_change_pct is None:
        return None
    return judge_outcome(0.0, record.actual_change_pct)


def _count(pairs: Iterable[tuple[str, str]]) -> WinLoss:
    wins = losses = 0
    for said, outcome in pairs:
        if said == outcome:
            wins += 1
        else:
            losses += 1
    return WinLoss(wins, losses)


def tally(records: Sequence[ViewRecord], recent: int = RECENT_WINDOW) -> Tally:
    """告知した目線すべてで数える。引き分けと未答え合わせは勝敗から外す。"""
    ordered = sorted(records, key=lambda r: r.day)
    judged = [(r, o) for r in ordered if (o := _outcome_of(r)) is not None]
    last = judged[-recent:] if recent > 0 else []

    def _three(rows: list[tuple[ViewRecord, str]]) -> tuple[WinLoss, WinLoss, WinLoss]:
        return (
            _count((r.view, o) for r, o in rows),
            _count(("up", o) for _, o in rows),
            _count((r.futures_view, o) for r, o in rows),
        )

    v, a, f = _three(judged)
    rv, ra, rf = _three(last)
    return Tally(
        view=v,
        always_up=a,
        futures=f,
        recent_view=rv,
        recent_always_up=ra,
        recent_futures=rf,
        announced_days=len(ordered),
        fallback_days=sum(1 for r in ordered if r.source == "fallback"),
        judged=len(judged),
    )


def is_consistent(claude: ClaudeView, moves: Sequence[IndexMove]) -> bool:
    """Claude が返した指数ごとの向きが騰落の符号と合うか。知らない指数は食い違いとみなす。"""
    if claude.view not in _DIRECTION_LABELS:
        return False
    by_symbol = {m.symbol: m for m in moves}
    for symbol, direction in claude.index_directions.items():
        move = by_symbol.get(symbol)
        if move is None or direction not in _DIRECTION_LABELS:
            return False
        actual = "up" if move.change > 0 else "down" if move.change < 0 else None
        if actual != direction:
            return False
    return True


def _futures_price(moves: Sequence[IndexMove]) -> float | None:
    for m in moves:
        if m.symbol == FUTURES_SYMBOL:
            return m.last
    return None


def choose_view(
    claude: ClaudeView | None, moves: Sequence[IndexMove], n225_prev_close: float
) -> tuple[str, str, ClaudeView | None]:
    """(目線, source, 使う Claude の結果)。無い・食い違うときは先物の向きで fallback。"""
    if claude is not None and is_consistent(claude, moves):
        return claude.view, "claude", claude
    if claude is not None:
        logger.warning("Claude の目線が数字と食い違うため fallback にする")
    price = _futures_price(moves)
    view = futures_view(price, n225_prev_close) if price is not None else "down"
    return view, "fallback", None


def _move_line(m: IndexMove) -> str:
    label = INDEX_LABELS.get(m.symbol, m.symbol)
    return f"- {label}: {m.last:,.2f}（{m.change:+,.2f}、{m.change_pct:+.2f}%）"


def _tally_lines(t: Tally) -> list[str]:
    return [
        f"- 直近{RECENT_WINDOW}: この目線 {t.recent_view.text()}"
        f"／いつも上げ {t.recent_always_up.text()}"
        f"／先物の向き {t.recent_futures.text()}",
        f"- 全期間: この目線 {t.view.text()}、うち先物だけの日 {t.fallback_days}"
        f"／いつも上げ {t.always_up.text()}"
        f"／先物の向き {t.futures.text()}",
    ]


def build_morning_text(
    moves: Sequence[IndexMove],
    n225_prev_close: float,
    claude: ClaudeView | None,
    records: Sequence[ViewRecord],
    verified_urls: set[str],
) -> str:
    """告知の本文を組み立てる。records は今日より前の告知と答え合わせ。

    今日の告知を 1 日と数え、検証中の n は告知した日数（今日を含む）。
    ニュースは verified_urls に入った URL のものだけ載せる。
    """
    view, source, used = choose_view(claude, moves, n225_prev_close)
    t = tally(records)
    lines: list[str] = []
    if t.judged < RECENT_WINDOW:
        lines.append(f"【検証中（{t.announced_days + 1} 日目）】")
    lines.append(f"今日の日本株: **{_DIRECTION_LABELS[view]}目線**")
    if used is not None:
        lines.append(f"理由: {used.reason}")
    else:
        lines.append("理由: 先物の向きだけで決めた（日経先物と日経平均の前日の終値の比べ）")

    lines.append("")
    lines.append("■ 夜の動き")
    lines.extend(_move_line(m) for m in moves)
    price = _futures_price(moves)
    if price is not None:
        gap = price - n225_prev_close
        lines.append(
            f"- 日経先物と日経平均の前日の終値の差: {gap:+,.0f} 円"
            f"（前日の終値 {n225_prev_close:,.2f}）"
        )
    if used is not None:
        lines.append(f"連動の見込み: {used.linkage}")

    lines.append("")
    lines.append("■ ニュースと今日の予定")
    if used is None:
        lines.append("- ニュースの取得に失敗（数字と先物の向きだけの告知）")
    else:
        kept = [n for n in used.news if n.url in verified_urls]
        if not kept:
            lines.append("- 確かめられたニュースなし")
        for n in kept:
            label = _DIRECTION_LABELS.get(n.direction, "?")
            lines.append(f"- [{label}の材料] {n.title} {n.url}")

    lines.append("")
    lines.append("■ 当たり率（日経平均の前日の終値比、引き分けは除く）")
    lines.extend(_tally_lines(t))
    if source == "fallback":
        lines.append("※今日の目線は先物の向きだけで決めた")
    return "\n".join(lines)
