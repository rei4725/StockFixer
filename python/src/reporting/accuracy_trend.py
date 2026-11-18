"""results/accuracy の日ごとの採点の JSON を読み返し、的中率の流れの表を作る（#1815）。

書く側（prediction/accuracy_report.py）の JSON は1日1ファイルで、的中率の流れを見るには
ファイルを1つずつ開くしかなかった。ここでは読むだけで、日・モデル・ホライズン・市場ごとの
物差しの表と、N 日の移動平均を作る。本番のジョブには組み込まない。

- 物差しは的中率・基準・rank_corr・predicted_up_rate の後ろに、予想と実際の大きさと
  予想の散らばり（mean_abs_predicted_ratio・mean_abs_actual_ratio・prediction_std、#2009）
- 市場の ALL はその日のモデル・ホライズンの全体の値、それ以外は by_market の値
- 欠けた日（ファイルが無い日）は行を作らない。移動平均は、その組の行（値のある日）の
  直近 N 行で取り、値の無い（None・キーの無い古い JSON の）日は平均から外す
- 読めないファイル・形の違う JSON は警告を出して飛ばす
- 4週の窓（各欄の four_week とその by_market、#2013）は別の表（build_four_week_trend）にする。
  four_week の無い古い JSON・空の窓は値を空にした行を残す
"""

import glob
import json
import math
import os
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TREND_METRICS: tuple[str, ...] = (
    "direction_accuracy",
    "majority_baseline_accuracy",
    "hit_rate_minus_majority",
    "rank_corr",
    "predicted_up_rate",
    # 予想の大きさと散らばり（#2009）。鍵があるのは 2026-11-06 以後のファイルだけ
    "mean_abs_predicted_ratio",
    "mean_abs_actual_ratio",
    "prediction_std",
)
ALL_MARKETS = "ALL"
_KEY_COLUMNS = ["date", "model", "horizon", "market"]
_GROUP_COLUMNS = ["model", "horizon", "market"]
_BASE_COLUMNS = _KEY_COLUMNS + ["n_scored"] + list(TREND_METRICS)
# 4週の窓（four_week、#2013）の表の列。鍵の無い欄（古い JSON・by_market の n_days など）は空
FOUR_WEEK_METRICS: tuple[str, ...] = TREND_METRICS + (
    "n_days",
    "daily_direction_accuracy_mean",
)
_FOUR_WEEK_COLUMNS = _KEY_COLUMNS + ["n_scored"] + list(FOUR_WEEK_METRICS)


def load_accuracy_reports(accuracy_dir: str) -> list[dict]:
    """accuracy_dir の YYYY-MM-DD.json を日付の順に読む（読めないものは飛ばす）。"""
    reports: list[dict] = []
    for path in sorted(glob.glob(os.path.join(accuracy_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("採点の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(report, dict):
            logger.warning("採点の JSON の形が違うため飛ばす: %s", path)
            continue
        report.setdefault("date", os.path.splitext(os.path.basename(path))[0])
        reports.append(report)
    return reports


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _row(day: str, model: str, horizon: str, market: str, section: dict) -> dict:
    row: dict[str, Any] = {
        "date": day,
        "model": model,
        "horizon": horizon,
        "market": market,
        "n_scored": _number(section.get("n_scored")),
    }
    for metric in TREND_METRICS:
        row[metric] = _number(section.get(metric))
    return row


def _report_rows(report: dict) -> list[dict]:
    day = str(report.get("date", ""))
    models = report.get("models")
    if not isinstance(models, dict):
        logger.warning("採点の JSON %s に models が無いため飛ばす", day)
        return []
    rows: list[dict] = []
    for model, by_horizon in models.items():
        if not isinstance(by_horizon, dict):
            continue
        for horizon, section in by_horizon.items():
            if not isinstance(section, dict):
                continue
            rows.append(_row(day, str(model), str(horizon), ALL_MARKETS, section))
            by_market = section.get("by_market")
            if not isinstance(by_market, dict):
                continue
            for market, market_section in sorted(by_market.items()):
                if isinstance(market_section, dict):
                    rows.append(_row(day, str(model), str(horizon), str(market), market_section))
    return rows


def build_accuracy_trend(reports: list[dict]) -> pd.DataFrame:
    """日・モデル・ホライズン・市場ごとの n_scored と TREND_METRICS の表を返す。"""
    rows = [row for report in reports for row in _report_rows(report)]
    df = pd.DataFrame(rows, columns=_BASE_COLUMNS)
    for column in ["n_scored"] + list(TREND_METRICS):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def _four_week_row(day: str, model: str, horizon: str, market: str, window: dict) -> dict:
    row: dict[str, Any] = {
        "date": day,
        "model": model,
        "horizon": horizon,
        "market": market,
        "n_scored": _number(window.get("n_scored")),
    }
    for metric in FOUR_WEEK_METRICS:
        row[metric] = _number(window.get(metric))
    return row


def _four_week_report_rows(report: dict) -> list[dict]:
    day = str(report.get("date", ""))
    models = report.get("models")
    if not isinstance(models, dict):
        logger.warning("採点の JSON %s に models が無いため飛ばす", day)
        return []
    rows: list[dict] = []
    for model, by_horizon in models.items():
        if not isinstance(by_horizon, dict):
            continue
        for horizon, section in by_horizon.items():
            if not isinstance(section, dict):
                continue
            window = section.get("four_week")
            # four_week の無い古い JSON は、値を空にした ALL の行を残す
            window = window if isinstance(window, dict) else {}
            rows.append(_four_week_row(day, str(model), str(horizon), ALL_MARKETS, window))
            by_market = window.get("by_market")
            if not isinstance(by_market, dict):
                continue
            for market, market_window in sorted(by_market.items()):
                if isinstance(market_window, dict):
                    rows.append(
                        _four_week_row(day, str(model), str(horizon), str(market), market_window)
                    )
    return rows


def build_four_week_trend(reports: list[dict]) -> pd.DataFrame:
    """各欄の four_week（4週の窓）を、日・モデル・ホライズン・市場ごとの行にした表を返す（#2013）。

    列は n_scored と FOUR_WEEK_METRICS。by_market の行に無い鍵（n_days など）は空にする。
    """
    rows = [row for report in reports for row in _four_week_report_rows(report)]
    df = pd.DataFrame(rows, columns=_FOUR_WEEK_COLUMNS)
    for column in ["n_scored"] + list(FOUR_WEEK_METRICS):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def load_four_week_trend(accuracy_dir: str) -> pd.DataFrame:
    """accuracy_dir の JSON を読み、4週の窓の表を返す（#2013）。"""
    return build_four_week_trend(load_accuracy_reports(accuracy_dir))


def add_moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """モデル・ホライズン・市場の組ごとに、日付の順の直近 window 行の平均の列を足す。

    列の名前は <物差し>_ma<window>。値の無い日は平均から外す（窓に値が1つも無ければ NaN）。
    """
    if window < 1:
        raise ValueError(f"window は1以上: {window}")
    out = df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)
    grouped = out.groupby(_GROUP_COLUMNS, sort=False)
    for metric in TREND_METRICS:
        out[f"{metric}_ma{window}"] = grouped[metric].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return out


def load_accuracy_trend(accuracy_dir: str, window: int) -> pd.DataFrame:
    """accuracy_dir の JSON を読み、移動平均つきの表を返す。"""
    return add_moving_average(build_accuracy_trend(load_accuracy_reports(accuracy_dir)), window)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_trend_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（行が無ければ「データなし」）。"""
    if df.empty:
        return "データなし"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in values) + " |")
    return "\n".join(lines)
