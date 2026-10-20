"""日次の採点の結果を results/accuracy/YYYY-MM-DD.json に書き出す（C-011 の2）。

日次パイプラインの [2/6] の採点の後に呼ばれ、production と challenger それぞれの
ホライズンごとの採点の件数・向きの的中率・平均の誤差、4週の平均、外れの大きい銘柄、
市場と業種ごとの内訳を、DB を引かずに読める JSON に残す。

「今日の採点」は、[3/6] で上書きされる前の prediction_results のスナップショット
（run_accuracy_check が採点した予測そのもの）と prediction_accuracy の行を
(market, symbol, predicted_at) で突き合わせたものとする。採点が 0 件の日も件数 0 で書く。

設計: ai-company docs/superpowers/specs/2026-10-15-prediction-ops-design.md（土台の2）
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Callable, Iterable

import pandas as pd

from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.japan_time import isoformat_utc, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

ACCURACY_REPORT_MODELS: tuple[str, ...] = ("production", "challenger")
ACCURACY_REPORT_HORIZONS: tuple[int, ...] = (1,)
WINDOW_DAYS = 28
TOP_MISSES_N = 10
UNKNOWN_SECTOR = "UNKNOWN"

# run_accuracy_check と同じ件数だけスナップショットを読む
_SNAPSHOT_LIMIT = 5000
# 1日 約700銘柄 × 2モデル × 4週の営業日を十分に覆う件数
_HISTORY_LIMIT = 100_000
_KEY = ["market", "symbol", "predicted_at"]

SectorGetter = Callable[[str, str], str]


def _summary(rows: pd.DataFrame) -> dict:
    """件数・向きの的中率・平均の誤差（|予想の変化率 - 実際の変化率|）を返す。"""
    if rows.empty:
        return {"n_scored": 0, "direction_accuracy": None, "mean_abs_error": None}
    matches = rows["direction_match"].dropna().astype(bool)
    errors = rows["abs_error"].dropna()
    return {
        "n_scored": int(len(rows)),
        "direction_accuracy": float(matches.mean()) if len(matches) else None,
        "mean_abs_error": float(errors.mean()) if len(errors) else None,
    }


def _four_week_summary(rows: pd.DataFrame) -> dict:
    """4週の窓の行を通した平均と、日ごとの的中率の平均を返す。"""
    summary = _summary(rows)
    if rows.empty:
        summary.update({"n_days": 0, "daily_direction_accuracy_mean": None})
        return summary
    days = rows["predicted_at"].str[:8]
    daily = rows.dropna(subset=["direction_match"]).groupby(days)["direction_match"]
    daily_rates = daily.apply(lambda s: s.astype(bool).mean())
    summary.update(
        {
            "n_days": int(days.nunique()),
            "daily_direction_accuracy_mean": (
                float(daily_rates.mean()) if len(daily_rates) else None
            ),
        }
    )
    return summary


def _breakdown(rows: pd.DataFrame, keys: pd.Series) -> dict:
    """keys ごとの件数・的中率・平均の誤差を返す。"""
    if rows.empty:
        return {}
    return {str(key): _summary(group) for key, group in rows.groupby(keys, sort=True)}


def _top_misses(rows: pd.DataFrame, top_n: int) -> list[dict]:
    """外れ（abs_error）の大きい順に top_n 件を返す。"""
    scored = rows.dropna(subset=["abs_error"])
    if scored.empty:
        return []
    top = scored.sort_values("abs_error", ascending=False, kind="stable").head(top_n)
    return [
        {
            "market": str(row["market"]),
            "symbol": str(row["symbol"]),
            "predicted_at": str(row["predicted_at"]),
            "predicted_ratio": float(row["predicted_ratio"]),
            "actual_ratio": float(row["actual_ratio"]),
            "abs_error": float(row["abs_error"]),
        }
        for _, row in top.iterrows()
    ]


def _safe_sector(sector_getter: SectorGetter, market: str, symbol: str) -> str:
    try:
        return str(sector_getter(market, symbol)) or UNKNOWN_SECTOR
    except Exception as e:
        logger.warning("業種の取得失敗 %s/%s: %s", market, symbol, e, exc_info=True)
        return UNKNOWN_SECTOR


def build_horizon_report(
    today_rows: pd.DataFrame,
    window_rows: pd.DataFrame,
    sector_getter: SectorGetter,
    top_n: int = TOP_MISSES_N,
) -> dict:
    """1つのモデル・ホライズンの欄を作る。

    Args:
        today_rows: 今日採点した prediction_accuracy の行
        window_rows: 4週の窓に入る prediction_accuracy の行（今日の分を含む）
        sector_getter: (market, symbol) → 業種名
        top_n: 外れの大きい銘柄の件数
    """
    today = _with_abs_error(today_rows)
    window = _with_abs_error(window_rows)

    sectors: dict[tuple[str, str], str] = {}
    for market, symbol in today[["market", "symbol"]].drop_duplicates().itertuples(index=False):
        sectors[(market, symbol)] = _safe_sector(sector_getter, market, symbol)
    sector_keys = pd.Series(
        [sectors[(m, s)] for m, s in zip(today["market"], today["symbol"])],
        index=today.index,
        dtype=object,
    )

    report = _summary(today)
    report.update(
        {
            "four_week": _four_week_summary(window),
            "top_misses": _top_misses(today, top_n),
            "by_market": _breakdown(today, today["market"]),
            "by_sector": _breakdown(today, sector_keys),
        }
    )
    return report


_ACCURACY_COLUMNS = [
    "market",
    "symbol",
    "model_name",
    "predicted_at",
    "horizon",
    "predicted_ratio",
    "actual_ratio",
    "direction_match",
]


def _with_abs_error(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        rows = pd.DataFrame(columns=_ACCURACY_COLUMNS)
    rows = rows.copy()
    rows["abs_error"] = (
        pd.to_numeric(rows["predicted_ratio"], errors="coerce")
        - pd.to_numeric(rows["actual_ratio"], errors="coerce")
    ).abs()
    return rows


def _today_rows(model_rows: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """prediction_results のスナップショットに載っている予測の採点の行を返す。"""
    if model_rows.empty or snapshot is None or snapshot.empty:
        return model_rows.iloc[0:0]
    keys = snapshot[_KEY].drop_duplicates().astype(str)
    return model_rows.merge(keys, on=_KEY, how="inner")


def _load_accuracy(horizon: int) -> pd.DataFrame:
    from src.prediction.db import load_prediction_accuracy

    rows = load_prediction_accuracy(horizon=horizon, limit=_HISTORY_LIMIT)
    if rows is None or rows.empty:
        return pd.DataFrame(columns=_ACCURACY_COLUMNS)
    rows = rows.copy()
    for col in _KEY + ["model_name"]:
        rows[col] = rows[col].astype(str)
    return rows


def _load_snapshot(model: str) -> pd.DataFrame:
    from src.prediction.db import load_prediction_results

    snapshot = load_prediction_results(limit=_SNAPSHOT_LIMIT, model_version=model)
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=_KEY)
    return snapshot


def write_daily_accuracy_report(
    report_date: date | None = None,
    models: Iterable[str] = ACCURACY_REPORT_MODELS,
    horizons: Iterable[int] = ACCURACY_REPORT_HORIZONS,
    output_dir: str | None = None,
    sector_getter: SectorGetter | None = None,
) -> str:
    """今日の採点の JSON を output_dir/YYYY-MM-DD.json に書き、そのパスを返す。

    [3/6] で prediction_results が上書きされる前に呼ぶこと。

    Args:
        report_date: ファイル名の日付（None なら日本時間の今日）
        models: 書き出すモデル（prediction_accuracy の model_name）
        horizons: 書き出すホライズン
        output_dir: 置き場（None なら results/accuracy）
        sector_getter: (market, symbol) → 業種名（None なら yfinance から引く）
    """
    if report_date is None:
        report_date = now_jst().date()
    if output_dir is None:
        output_dir = os.path.join(get_results_dir(), "accuracy")
    if sector_getter is None:
        from src.utils.sector_constraints import get_symbol_sector

        sector_getter = get_symbol_sector

    models = tuple(models)
    since = (report_date - timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d")
    snapshots = {model: _load_snapshot(model) for model in models}

    sections: dict[str, dict[str, dict]] = {model: {} for model in models}
    for horizon in horizons:
        accuracy = _load_accuracy(horizon)
        for model in models:
            model_rows = accuracy[accuracy["model_name"] == model]
            window_rows = model_rows[model_rows["predicted_at"].str[:8] >= since]
            today_rows = _today_rows(model_rows, snapshots[model])
            sections[model][str(horizon)] = build_horizon_report(
                today_rows, window_rows, sector_getter
            )

    report = {
        "date": report_date.isoformat(),
        "generated_at": isoformat_utc(),
        "window_days": WINDOW_DAYS,
        "models": sections,
    }

    ensure_dir(output_dir)
    path = os.path.join(output_dir, f"{report_date.isoformat()}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp_path, path)

    counts = ", ".join(
        f"{model}/h{h}={section['n_scored']}"
        for model, by_h in sections.items()
        for h, section in by_h.items()
    )
    logger.info("採点の JSON を保存: %s (%s)", path, counts)
    return path
