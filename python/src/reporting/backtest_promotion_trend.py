"""results/backtest_promotion の夜間の仮の入れ替えの記録を読み返し、流れの表を作る（#1890）。

書く側（orchestration/jobs/backtest_promotion_job.py の _write_result）の JSON は1回1ファイルで、
負けの理由・日ごとの勝ち越し・予想の上げの割合の流れを見るにはファイルを1つずつ開くしかなかった。
ここでは読むだけで、回ごとに1行の表を作る。本番のジョブには組み込まない。

- 読むのは YYYYMMDD_HHMMSS.json と同じ秒の YYYYMMDD_HHMMSS_<n>.json だけで、同じフォルダの
  state.json・features_version.json は飛ばす
- 読めないファイル・形の違う JSON は警告を出して飛ばす
- evaluation の無い回（skipped・failed など）は、評価の列をすべて欠け（pd.NA）にする
- 値が null の数・鍵の無い数は欠けにし、0 と分ける
- reason の基準ごとの有無は、evaluation のある回だけ True/False にし、無い回は欠けにする
- 右端（reason_other の後ろ）に、判定に使う物差しの列を足す（#2162）。前からの列は変えない
  - candidate_mae・production_mae・candidate_daily_rank_corr・production_daily_rank_corr（全体）
  - <市場>_<candidate か production>_ の頭の daily_rank_corr・hit_rate_minus_majority・
    predicted_up_rate（evaluation.markets。市場の並びは n_compared_ と同じ）
  - daily_rank_corr の鍵の無い古い回・markets の無い回・evaluation の無い回は欠け
  - 全体の daily_rank_corr は 2.88.0 までの回では日ごと、その後の回では市場と取引日ごとに
    束ねた値である（#2157）。JSON の値をそのまま出し、計算し直さない
"""

import json
import os
import re
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_FILE_PATTERN = re.compile(r"^(\d{8}_\d{6})(?:_(\d+))?\.json$")

RUN_COLUMNS: tuple[str, ...] = (
    "started_at",
    "version",
    "force",
    "dry_run",
    "outcome",
    "swapped",
    "lock_timeout",
)
VOTE_COLUMNS: tuple[str, ...] = ("vote_wins", "vote_days")
METRIC_KEYS: tuple[str, ...] = (
    "hit_rate_minus_majority",
    "rank_corr",
    "predicted_up_rate",
    "prediction_std",
)
SIDES: tuple[str, ...] = ("candidate", "production")
METRIC_COLUMNS: tuple[str, ...] = tuple(f"{side}_{key}" for side in SIDES for key in METRIC_KEYS)
# 判定（judge_win）に使う全体の物差しと、市場ごとに出す物差し（#2162）。表の右端に足す
JUDGED_KEYS: tuple[str, ...] = ("mae", "daily_rank_corr")
JUDGED_COLUMNS: tuple[str, ...] = tuple(f"{side}_{key}" for key in JUDGED_KEYS for side in SIDES)
MARKET_METRIC_KEYS: tuple[str, ...] = (
    "daily_rank_corr",
    "hit_rate_minus_majority",
    "predicted_up_rate",
)
N_COMPARED_PREFIX = "n_compared_"
REASON_PREFIX = "reason_"
# reason（"; " でつないだ落ちた基準）の1つずつを、含む語で基準に分ける。どれにも当たらなければ other
REASON_CRITERIA: tuple[tuple[str, str], ...] = (
    ("coverage", "採点行が足りない"),
    ("hit_rate", "基準を引いた的中率"),
    ("mae", "MAE が"),
    ("rank_corr", "順位相関"),
    ("daily_vote", "日ごとの勝ち越し"),
    ("constant", "ほぼ同じ値"),
    ("degraded", "モデルが欠けた行"),
    ("path_check", "道の一致の検査"),
)
REASON_COLUMNS: tuple[str, ...] = tuple(REASON_PREFIX + name for name, _ in REASON_CRITERIA) + (
    REASON_PREFIX + "other",
)
_FLAG_COLUMNS = ("force", "dry_run", "swapped", "lock_timeout") + REASON_COLUMNS


def _sort_key(name: str) -> tuple[str, int]:
    match = _FILE_PATTERN.match(name)
    if match is None:
        return name, 0
    return match.group(1), int(match.group(2) or 1)


def load_backtest_promotion_records(results_dir: str) -> list[dict]:
    """results_dir の回ごとの JSON を、ファイル名の日付の順に読む（読めないものは飛ばす）。"""
    try:
        names = os.listdir(results_dir)
    except OSError as e:
        logger.warning("仮の入れ替えの記録の置き場を読めない: %s (%s)", results_dir, e)
        return []
    records: list[dict] = []
    for name in sorted((n for n in names if _FILE_PATTERN.match(n)), key=_sort_key):
        path = os.path.join(results_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("仮の入れ替えの記録の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(record, dict):
            logger.warning("仮の入れ替えの記録の JSON の形が違うため飛ばす: %s", path)
            continue
        record["_file"] = os.path.splitext(name)[0]
        records.append(record)
    return records


def _number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return pd.NA
    return value


def _flag(value: Any) -> Any:
    return value if isinstance(value, bool) else pd.NA


def _votes(evaluation: dict) -> tuple[Any, Any]:
    vote_dates = evaluation.get("vote_dates")
    daily = evaluation.get("daily")
    if not isinstance(vote_dates, list) or not isinstance(daily, list):
        return pd.NA, pd.NA
    vote_set = set(vote_dates)
    wins = 0
    for day in daily:
        if not isinstance(day, dict) or day.get("date") not in vote_set:
            continue
        diff = _number(day.get("diff"))
        if diff is not pd.NA and diff > 0:
            wins += 1
    return wins, len(vote_dates)


def _reason_flags(reason: Any) -> dict[str, bool]:
    parts = [p.strip() for p in reason.split(";")] if isinstance(reason, str) else []
    flags = {column: False for column in REASON_COLUMNS}
    for part in filter(None, parts):
        matched = False
        for name, word in REASON_CRITERIA:
            if word in part:
                flags[REASON_PREFIX + name] = True
                matched = True
        if not matched:
            flags[REASON_PREFIX + "other"] = True
    return flags


def _row(record: dict) -> tuple[dict, dict[str, Any] | None]:
    row: dict[str, Any] = {"file": record["_file"]}
    for key in RUN_COLUMNS:
        value = record.get(key)
        if key in ("force", "dry_run", "swapped", "lock_timeout"):
            row[key] = _flag(value)
        else:
            row[key] = value if isinstance(value, str) else pd.NA
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        for key in VOTE_COLUMNS + METRIC_COLUMNS + REASON_COLUMNS:
            row[key] = pd.NA
        return row, None
    row["vote_wins"], row["vote_days"] = _votes(evaluation)
    for side in SIDES:
        metrics = evaluation.get(side)
        metrics = metrics if isinstance(metrics, dict) else {}
        for key in METRIC_KEYS:
            row[f"{side}_{key}"] = _number(metrics.get(key))
    row.update(_reason_flags(record.get("reason")))
    markets = evaluation.get("markets")
    n_compared: dict[str, Any] = {}
    if isinstance(markets, dict):
        for market, summary in markets.items():
            value = summary.get("n_compared") if isinstance(summary, dict) else None
            n_compared[str(market)] = _number(value)
    return row, n_compared


def _child(value: Any, key: str) -> dict:
    """dict の中の dict を返す。形が違う・鍵が無いときは空の dict。"""
    child = value.get(key) if isinstance(value, dict) else None
    return child if isinstance(child, dict) else {}


def _judged_metrics(record: dict, markets: list[str]) -> dict[str, Any]:
    """右端に足す列の値（全体の判定の物差しと、市場ごとの物差し）。無いものは欠け。"""
    evaluation = _child(record, "evaluation")
    values: dict[str, Any] = {}
    for key in JUDGED_KEYS:
        for side in SIDES:
            values[f"{side}_{key}"] = _number(_child(evaluation, side).get(key))
    by_market = _child(evaluation, "markets")
    for market in markets:
        for side in SIDES:
            metrics = _child(_child(by_market, market), side)
            for key in MARKET_METRIC_KEYS:
                values[f"{market}_{side}_{key}"] = _number(metrics.get(key))
    return values


def build_backtest_promotion_trend(records: list[dict]) -> pd.DataFrame:
    """回ごとに1行の表を返す。

    市場ごとの n_compared は n_compared_<市場> の列に、判定の物差しと市場ごとの物差しは右端の
    列（JUDGED_COLUMNS と <市場>_<candidate か production>_<MARKET_METRIC_KEYS>）にする。
    """
    rows: list[dict] = []
    compared_by_row: list[dict[str, Any] | None] = []
    for record in records:
        row, n_compared = _row(record)
        rows.append(row)
        compared_by_row.append(n_compared)
    markets = sorted({m for compared in compared_by_row if compared for m in compared})
    for record, row, compared in zip(records, rows, compared_by_row):
        for market in markets:
            row[N_COMPARED_PREFIX + market] = (
                pd.NA if compared is None else compared.get(market, pd.NA)
            )
        row.update(_judged_metrics(record, markets))
    added_columns = list(JUDGED_COLUMNS) + [
        f"{market}_{side}_{key}"
        for market in markets
        for side in SIDES
        for key in MARKET_METRIC_KEYS
    ]
    count_columns = list(VOTE_COLUMNS) + [N_COMPARED_PREFIX + m for m in markets]
    columns = (
        ["file"]
        + list(RUN_COLUMNS)
        + list(VOTE_COLUMNS)
        + list(METRIC_COLUMNS)
        + [N_COMPARED_PREFIX + m for m in markets]
        + list(REASON_COLUMNS)
        + added_columns
    )
    df = pd.DataFrame(rows, columns=columns)
    for column in count_columns:
        df[column] = df[column].astype("Int64")
    for column in list(METRIC_COLUMNS) + added_columns:
        df[column] = df[column].astype("Float64")
    for column in _FLAG_COLUMNS:
        df[column] = df[column].astype("boolean")
    return df.reset_index(drop=True)


def load_backtest_promotion_trend(results_dir: str) -> pd.DataFrame:
    """results_dir の JSON を読み、回ごとの表を返す。"""
    return build_backtest_promotion_trend(load_backtest_promotion_records(results_dir))


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_backtest_promotion_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（欠けは null。行が無ければ「データなし」）。"""
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


def format_backtest_promotion_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書き、0 と分ける）。"""
    return df.to_csv(index=False, na_rep="null")
