"""日次の採点の結果を results/accuracy/YYYY-MM-DD.json に書き出す（C-011 の2）。

日次パイプラインの [2/6] の採点の後に呼ばれ、production と challenger それぞれの
ホライズンごとの採点の件数・向きの的中率・平均の誤差、4週の平均、外れの大きい銘柄、
市場と業種ごとの内訳を、DB を引かずに読める JSON に残す。業種の内訳は yfinance を
引くため、日次の道の最後（[6/6] の後）に書き足す（#843）。

「今日の採点」は、[3/6] で上書きされる前の prediction_results のスナップショット
（run_accuracy_check が採点した予測そのもの）のうち predicted_at の一番新しい回の行と、
prediction_accuracy の行を (market, symbol, predicted_at) で突き合わせたものとする。採点が 0 件の日も件数 0 で書く。

設計: ai-company docs/superpowers/specs/2026-10-15-prediction-ops-design.md（土台の2）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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


def _sector_breakdown(today_rows: pd.DataFrame, sector_getter: SectorGetter) -> dict:
    """今日の採点の行の業種ごとの内訳を返す（引けない銘柄は UNKNOWN）。"""
    today = _with_abs_error(today_rows)
    sectors: dict[tuple[str, str], str] = {}
    for market, symbol in today[["market", "symbol"]].drop_duplicates().itertuples(index=False):
        sectors[(market, symbol)] = _safe_sector(sector_getter, market, symbol)
    sector_keys = pd.Series(
        [sectors[(m, s)] for m, s in zip(today["market"], today["symbol"])],
        index=today.index,
        dtype=object,
    )
    return _breakdown(today, sector_keys)


def build_horizon_report(
    today_rows: pd.DataFrame,
    window_rows: pd.DataFrame,
    sector_getter: SectorGetter | None,
    top_n: int = TOP_MISSES_N,
) -> dict:
    """1つのモデル・ホライズンの欄を作る。

    Args:
        today_rows: 今日採点した prediction_accuracy の行
        window_rows: 4週の窓に入る prediction_accuracy の行（今日の分を含む）
        sector_getter: (market, symbol) → 業種名（None なら業種を引かず by_sector は空）
        top_n: 外れの大きい銘柄の件数
    """
    today = _with_abs_error(today_rows)
    window = _with_abs_error(window_rows)

    report = _summary(today)
    report.update(
        {
            "four_week": _four_week_summary(window),
            "top_misses": _top_misses(today, top_n),
            "by_market": _breakdown(today, today["market"]),
            "by_sector": (
                _sector_breakdown(today_rows, sector_getter) if sector_getter is not None else {}
            ),
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


def _today_rows(model_rows: pd.DataFrame, snapshot: pd.DataFrame, model: str) -> pd.DataFrame:
    """prediction_results のスナップショットの一番新しい回の予測の採点の行を返す。

    保存は銘柄ごとの Delete-Insert のため、その回で予想が出なかった銘柄は前の回の
    行がスナップショットに残る（#959）。一番新しい回は predicted_at の最大で決め、
    前の回の行は今日の採点に入れない（1回の予測は全銘柄で同じ predicted_at を持つ）。
    """
    if model_rows.empty or snapshot is None or snapshot.empty:
        return model_rows.iloc[0:0]
    keys = snapshot[_KEY].drop_duplicates().astype(str)
    latest_at = keys["predicted_at"].max()
    is_latest = keys["predicted_at"] == latest_at
    n_old = int((~is_latest).sum())
    if n_old:
        logger.info(
            "採点の JSON: %s のスナップショットの前の回の行を %d件 除く（今の回=%s）",
            model,
            n_old,
            latest_at,
        )
    return model_rows.merge(keys[is_latest], on=_KEY, how="inner")


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


@dataclass
class DailyAccuracyReport:
    """今日の採点の JSON の材料。

    prepare_daily_accuracy_report が [3/6] の保存の前に作り、業種の内訳（by_sector）は
    write に業種の引き方を渡したときに埋める。業種は yfinance を銘柄ごとに引くため
    約700銘柄で数分かかり、日次の道では [6/6] の後に回す（#843）。
    """

    report_date: date
    output_dir: str
    sections: dict[str, dict[str, dict]]
    today_rows: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)

    def write(self, sector_getter: SectorGetter | None) -> str:
        """JSON を output_dir/YYYY-MM-DD.json に書き、そのパスを返す。

        Args:
            sector_getter: (market, symbol) → 業種名（None なら業種を引かず by_sector は空）
        """
        if sector_getter is not None:
            for (model, horizon), rows in self.today_rows.items():
                self.sections[model][horizon]["by_sector"] = _sector_breakdown(rows, sector_getter)

        report = {
            "date": self.report_date.isoformat(),
            "generated_at": isoformat_utc(),
            "window_days": WINDOW_DAYS,
            "models": self.sections,
        }

        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, f"{self.report_date.isoformat()}.json")
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp_path, path)

        counts = ", ".join(
            f"{model}/h{h}={section['n_scored']}"
            for model, by_h in self.sections.items()
            for h, section in by_h.items()
        )
        logger.info(
            "採点の JSON を保存: %s (%s, 業種=%s)",
            path,
            counts,
            "あり" if sector_getter is not None else "なし",
        )
        return path

    def write_with_sectors(self) -> str:
        """業種を yfinance（get_symbol_sector）から引いて JSON を書き直す。"""
        from src.utils.sector_constraints import get_symbol_sector

        return self.write(get_symbol_sector)


def prepare_daily_accuracy_report(
    report_date: date | None = None,
    models: Iterable[str] = ACCURACY_REPORT_MODELS,
    horizons: Iterable[int] = ACCURACY_REPORT_HORIZONS,
    output_dir: str | None = None,
) -> DailyAccuracyReport:
    """今日の採点の JSON の材料を作る（業種は引かない）。

    [3/6] で prediction_results が上書きされる前に呼ぶこと。

    Args:
        report_date: ファイル名の日付（None なら日本時間の今日）
        models: 書き出すモデル（prediction_accuracy の model_name）
        horizons: 書き出すホライズン
        output_dir: 置き場（None なら results/accuracy）
    """
    if report_date is None:
        report_date = now_jst().date()
    if output_dir is None:
        output_dir = os.path.join(get_results_dir(), "accuracy")

    models = tuple(models)
    since = (report_date - timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d")
    snapshots = {model: _load_snapshot(model) for model in models}

    sections: dict[str, dict[str, dict]] = {model: {} for model in models}
    today_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for horizon in horizons:
        accuracy = _load_accuracy(horizon)
        for model in models:
            model_rows = accuracy[accuracy["model_name"] == model]
            window_rows = model_rows[model_rows["predicted_at"].str[:8] >= since]
            today_rows = _today_rows(model_rows, snapshots[model], model)
            sections[model][str(horizon)] = build_horizon_report(today_rows, window_rows, None)
            today_by_key[(model, str(horizon))] = today_rows

    return DailyAccuracyReport(
        report_date=report_date,
        output_dir=output_dir,
        sections=sections,
        today_rows=today_by_key,
    )


def write_daily_accuracy_report(
    report_date: date | None = None,
    models: Iterable[str] = ACCURACY_REPORT_MODELS,
    horizons: Iterable[int] = ACCURACY_REPORT_HORIZONS,
    output_dir: str | None = None,
    sector_getter: SectorGetter | None = None,
) -> str:
    """今日の採点の JSON を業種の内訳つきで一度に書き、そのパスを返す。

    [3/6] で prediction_results が上書きされる前に呼ぶこと。日次の道は業種を引く
    時間で予測を遅らせないよう、prepare_daily_accuracy_report と
    DailyAccuracyReport.write を分けて使う。

    Args:
        report_date: ファイル名の日付（None なら日本時間の今日）
        models: 書き出すモデル（prediction_accuracy の model_name）
        horizons: 書き出すホライズン
        output_dir: 置き場（None なら results/accuracy）
        sector_getter: (market, symbol) → 業種名（None なら yfinance から引く）
    """
    pending = prepare_daily_accuracy_report(report_date, models, horizons, output_dir)
    if sector_getter is None:
        return pending.write_with_sectors()
    return pending.write(sector_getter)
