"""
取っておいた期間の比べ（holdout）と旧 production の物差し（old_production）が使う、行の部品。

DB にも差し替えの名前にも触れない純粋な関数だけを置く（#2192）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.prediction.baseline_metrics import baseline_metrics


def close_by_day(ohlcv: Optional[pd.DataFrame]) -> dict:
    """生の OHLCV から、日付（タイムゾーン無しの0時）→ 終値の dict を作る。"""
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return {}
    index = pd.DatetimeIndex(pd.to_datetime(ohlcv.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    closes = ohlcv["Close"].to_numpy()
    return {pd.Timestamp(d): float(c) for d, c in zip(index.normalize(), closes) if pd.notna(c)}


def daily_metrics(rows: pd.DataFrame, dates: list[str]) -> list[dict]:
    """日ごとの候補・production の hit_rate_minus_majority と差。"""
    daily = []
    for day in dates:
        day_rows = rows[rows["date"] == day]
        cand = baseline_metrics(day_rows.rename(columns={"candidate_ratio": "predicted_ratio"}))[
            "hit_rate_minus_majority"
        ]
        prod = baseline_metrics(day_rows.rename(columns={"production_ratio": "predicted_ratio"}))[
            "hit_rate_minus_majority"
        ]
        diff = None if cand is None or prod is None else cand - prod
        daily.append({"date": day, "candidate": cand, "production": prod, "diff": diff})
    return daily


def both_metrics(rows: pd.DataFrame) -> tuple[dict, dict]:
    """同じ行での候補と production の baseline_metrics()。"""
    return (
        baseline_metrics(rows.rename(columns={"candidate_ratio": "predicted_ratio"})),
        baseline_metrics(rows.rename(columns={"production_ratio": "predicted_ratio"})),
    )


def with_market_day(rows: pd.DataFrame) -> pd.DataFrame:
    """日の鍵（date）を「市場:取引日」にした写しを返す（#2157）。

    日だけで束ねると、jp と us の水準の差を銘柄の並べの腕とばらつきに数える。影の比べ（#2072）と
    採点の JSON（#2141）と同じ鍵で、全体の daily_rank_corr と prediction_std だけに効く。
    """
    return rows.assign(date=rows["market"].astype(str) + ":" + rows["date"])
