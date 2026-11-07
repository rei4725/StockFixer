"""
取っておいた期間の日ごとの勝ち越しの票に数える日を、発注する市場（jp）の銘柄の線で選ぶ（#1714）。

仮の入れ替えの見張り（#1706）と同じく、jp の比べられた行のある銘柄の数が
ウォッチリストの jp の銘柄の数の MIN_COVERAGE 以上の日だけを数える。
分母は行から作らない（段差で表から消えた銘柄の分も縮まない。#1701）。
ウォッチリストが読めない回は universe の jp の銘柄の数を分母にする（#1701 の銘柄の線と同じ）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.domain.jp_market_holidays import is_jp_business_day
from src.prediction.backtest_promotion.judge import MIN_COVERAGE
from src.prediction.backtest_promotion.required_market import REQUIRED_MARKET

# 票から外した日の理由
VOTE_SKIP_JP_HOLIDAY = "jp_holiday"
VOTE_SKIP_JP_SYMBOLS_SHORT = "jp_symbols_short"


def vote_need_symbols(n_universe_jp: int, watchlist: Optional[set[str]]) -> float:
    """票に数える日に要る jp の銘柄の数（ウォッチリストの数 × MIN_COVERAGE）。"""
    n_listed = len(watchlist) if watchlist else n_universe_jp
    return n_listed * MIN_COVERAGE


def split_vote_days(
    table: pd.DataFrame,
    dates: list[str],
    need_symbols: float,
    watchlist: Optional[set[str]],
) -> tuple[list[str], list[dict]]:
    """取っておいた日を、票に数える日と外す日に分ける（どちらも日付の順）。

    日ごとの jp の銘柄は、候補と production の予想がそろった行のある銘柄を1日1銘柄で数える。
    ウォッチリストが読めた回は、その銘柄だけを数える（表にだけある廃止の銘柄を数えない）。

    Args:
        table: 比べられた行（market・symbol・date の列）
        dates: 取っておいた日（YYYY-MM-DD、古い順）
        need_symbols: vote_need_symbols() の線
        watchlist: ウォッチリストの jp の銘柄（読めない・空なら None か空）

    Returns:
        (票に数える日, 外した日の date・reason・n_jp_symbols の dict の並び)
    """
    jp_rows = table[table["market"] == REQUIRED_MARKET]
    if watchlist:
        jp_rows = jp_rows[jp_rows["symbol"].astype(str).str.upper().isin(watchlist)]
    symbols_by_day = jp_rows.groupby("date")["symbol"].nunique()
    counted: list[str] = []
    skipped: list[dict] = []
    for day in dates:
        n_jp = int(symbols_by_day.get(day, 0))
        if n_jp >= need_symbols:
            counted.append(day)
            continue
        holiday = not is_jp_business_day(pd.Timestamp(day).date())
        reason = VOTE_SKIP_JP_HOLIDAY if holiday else VOTE_SKIP_JP_SYMBOLS_SHORT
        skipped.append({"date": day, "reason": reason, "n_jp_symbols": n_jp})
    return counted, skipped
