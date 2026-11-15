"""センチメント特徴量の埋まり具合を市場ごとに数える診断。

ニュースの件数があるのに点が 0 のままの行がどれだけあるかを見るための道具であり、
採点（sentiment_features）と特徴量の値は変えない。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

COVERAGE_COLUMNS = [
    "market",
    "rows",
    "news_rows",
    "news_days",
    "zero_score_rows",
    "zero_score_rate",
    "last_nonzero_date",
]

_LOAD_COLUMNS = ["market", "symbol", "date", "news_count_lag1", "sentiment_score_lag1"]


def sentiment_coverage(
    df: pd.DataFrame,
    since: Optional[str] = None,
    score_col: str = "sentiment_score_lag1",
    news_col: str = "news_count_lag1",
) -> pd.DataFrame:
    """市場ごとのセンチメントの埋まり具合を返す（純粋な関数）。

    Args:
        df: market・date・news_col・score_col の列を持つ行
        since: この日（YYYY-MM-DD）以後の行だけを数える。None なら全部
        score_col: 点の列
        news_col: ニュースの件数の列

    Returns:
        1 市場 1 行の DataFrame。列は COVERAGE_COLUMNS。
          - rows: 行数
          - news_rows: ニュースの件数が 1 以上の行
          - news_days: ニュースのある行の日付の数
          - zero_score_rows: ニュースのある行のうち点がちょうど 0 の行（点の欠けは数えない）
          - zero_score_rate: zero_score_rows / news_rows（news_rows が 0 なら NaN）
          - last_nonzero_date: 点が 0 でない最後の日（無ければ NaT）
    """
    data = df
    if since is not None:
        data = data[pd.to_datetime(data["date"]) >= pd.Timestamp(since)]

    records = []
    for market, group in data.groupby("market", sort=True):
        dates = pd.to_datetime(group["date"])
        news = pd.to_numeric(group[news_col], errors="coerce").fillna(0) > 0
        score = pd.to_numeric(group[score_col], errors="coerce")
        zero = news & (score == 0)
        nonzero = score.notna() & (score != 0)
        news_rows = int(news.sum())
        records.append(
            {
                "market": market,
                "rows": len(group),
                "news_rows": news_rows,
                "news_days": int(dates[news].nunique()),
                "zero_score_rows": int(zero.sum()),
                "zero_score_rate": (int(zero.sum()) / news_rows) if news_rows else float("nan"),
                "last_nonzero_date": dates[nonzero].max() if nonzero.any() else pd.NaT,
            }
        )

    return pd.DataFrame(records, columns=COVERAGE_COLUMNS)


def load_sentiment_rows(since: Optional[str] = None) -> pd.DataFrame:
    """stock_features からセンチメントの診断に要る列だけを読む（読むだけ）。"""
    sql = "SELECT market, symbol, date, news_count_lag1, sentiment_score_lag1 FROM stock_features"
    params: list[str] = []
    if since is not None:
        sql += " WHERE date >= %s"
        params.append(since)
    with _db_connection() as con:
        rows = con.execute(sql, params).fetchall()
    df = pd.DataFrame(rows, columns=_LOAD_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("センチメントの診断用に stock_features を読んだ (%d行)", len(df))
    return df
