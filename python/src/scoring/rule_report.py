"""ルールシグナルと価格予想の日次・週次の採点。

日次: 2026-05-16 からの全期間を毎回採点し直し、採点の行を rule_signal_outcomes に書き、
results/rule_accuracy/YYYY-MM-DD.json に直近 28 日と全期間の2つの窓を書く。
週次: その週に初めて採点された行の集計と、銘柄ごとのルールの選び方の出入りを
results/rule_accuracy/weekly/YYYY-Www.json に書く。

時刻は UTC（scored_at は UTC の naive）。週の境は JST の月曜 0 時。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.scoring import db
from src.scoring.anchor import ANCHOR_HORIZON, HORIZONS, forward_returns, price_frame, score_signals
from src.scoring.version import read_app_version
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.japan_time import isoformat_utc, now_utc
from src.utils.logger import get_logger

logger = get_logger(__name__)

SINCE = date(2026, 5, 16)
WINDOW_DAYS = 28
TOP_SYMBOLS_N = 20
PREDICTION_KEYS: tuple[str, ...] = ("production", "challenger")
_DIRECTIONS: tuple[tuple[str, int], ...] = (("buy", 1), ("sell", -1))
_CLAIMS: tuple[tuple[str, int], ...] = (("h1_claim", 1), ("h5_claim", 5))
_JST_OFFSET = timedelta(hours=9)
_PRED_COLS = ["date", "symbol", "key", "direction", "horizon_of_claim"]
_SIG_COLS = ["date", "symbol", "key", "direction"]


@dataclass(frozen=True)
class ScoringLoaders:
    """DB の読み込み（テストで差し替える）。"""

    bars: Callable[[str, date], pd.DataFrame] = db.load_bars
    rule_signals: Callable[[str, date], pd.DataFrame] = db.load_rule_signals
    prediction_h1: Callable[[str, date], pd.DataFrame] = db.load_prediction_signals_h1
    prediction_history: Callable[[str, date], pd.DataFrame] = db.load_prediction_history
    rule_best: Callable[[str], pd.DataFrame] = db.load_rule_best
    rule_best_history: Callable[[str], pd.DataFrame] = db.load_rule_best_history
    rule_outcomes_scored: Callable[[str, datetime, datetime], pd.DataFrame] = (
        db.load_rule_outcomes_scored
    )


# ---------------------------------------------------------------------------
# シグナルの形をそろえる
# ---------------------------------------------------------------------------


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def _day(s: pd.Series) -> pd.Series:
    """merge_asof の鍵にする日時（分解能をそろえる。pandas は入力ごとに単位が変わる）。"""
    return pd.to_datetime(s).astype("datetime64[ns]")


def _parse_predicted_at(s: pd.Series) -> pd.Series:
    """YYYYMMDD_HHMMSS（UTC）を UTC の時刻にする。読めなければ ISO 8601 として読む。"""
    text = s.astype(str)
    ts = pd.to_datetime(text, format="%Y%m%d_%H%M%S", errors="coerce", utc=True)
    rest = ts.isna()
    if rest.any():
        ts[rest] = pd.to_datetime(text[rest], format="ISO8601", errors="coerce", utc=True)
    return ts


def prediction_signals(h1: pd.DataFrame, history: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """予想を [date, symbol, key, direction, horizon_of_claim] のシグナルにする。

    date は predicted_at の JST の日付より前で一番新しい、その銘柄の足の日。
    1日の予想は predicted_ratio、5日の予想は diff_ratio_5d の符号。0 と NaN は捨てる。
    """
    frames = []
    if not h1.empty:
        frames.append(
            pd.DataFrame(
                {
                    "symbol": h1["symbol"],
                    "key": h1["model_name"],
                    "predicted_at": h1["predicted_at"],
                    "value": h1["predicted_ratio"],
                    "horizon_of_claim": 1,
                }
            )
        )
    if not history.empty:
        frames.append(
            pd.DataFrame(
                {
                    "symbol": history["symbol"],
                    "key": history["model_version"],
                    "predicted_at": history["predicted_at"],
                    "value": history["diff_ratio_5d"],
                    "horizon_of_claim": 5,
                }
            )
        )
    empty = pd.DataFrame(columns=_PRED_COLS)
    if not frames or px.empty:
        return empty
    claims = pd.concat(frames, ignore_index=True)
    claims = claims[claims["key"].isin(PREDICTION_KEYS)]
    value = pd.to_numeric(claims["value"], errors="coerce")
    claims = claims.assign(direction=np.sign(value))
    claims = claims[claims["direction"].notna() & (claims["direction"] != 0)]
    ts = _parse_predicted_at(claims["predicted_at"])
    claims = claims.assign(
        ts=ts,
        jst_day=_day(pd.to_datetime((ts + _JST_OFFSET).dt.strftime("%Y-%m-%d"), errors="coerce")),
    ).dropna(subset=["jst_day"])
    if claims.empty:
        return empty
    bars = px[["symbol", "date"]].drop_duplicates()
    bars = bars.assign(bar_day=_day(bars["date"])).sort_values("bar_day")
    m = pd.merge_asof(
        claims.sort_values("jst_day"),
        bars[["symbol", "bar_day"]],
        left_on="jst_day",
        right_on="bar_day",
        by="symbol",
        direction="backward",
        allow_exact_matches=False,
    ).dropna(subset=["bar_day"])
    # 同じ足に同じ版の予想が複数あれば、最後の予想を使う
    m = m.sort_values("ts").drop_duplicates(
        ["bar_day", "symbol", "key", "horizon_of_claim"], keep="last"
    )
    out = pd.DataFrame(
        {
            "date": m["bar_day"].dt.date,
            "symbol": m["symbol"],
            "key": m["key"],
            "direction": m["direction"].astype(int),
            "horizon_of_claim": m["horizon_of_claim"].astype(int),
        }
    )
    return out.sort_values(["horizon_of_claim", "key", "date", "symbol"]).reset_index(drop=True)


def _rule_signal_frame(raw: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """rule_daily_signals の行を [date, symbol, key, direction, app_version] にする。"""
    if raw.empty:
        return pd.DataFrame(columns=_SIG_COLS + ["app_version"])
    out = pd.DataFrame(
        {
            "date": _to_date(raw["signal_date"]),
            "symbol": raw["symbol"].astype(str),
            "key": raw["rule_name"].astype(str),
            "direction": raw["signal"].astype(int),
            "app_version": raw["app_version"]
            .astype(object)
            .where(raw["app_version"].notna(), None),
        }
    )
    out = out[(out["date"] >= SINCE) & (out["date"] <= as_of) & (out["direction"] != 0)]
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _score(signals: pd.DataFrame, fwd: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if signals.empty:
        cols = ["date", "symbol", "key", "direction", "horizon", "hit", "baseline", "signed_return"]
        return pd.DataFrame(columns=cols), 0
    return score_signals(signals[_SIG_COLS], fwd, HORIZONS)


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _stats(sub: pd.DataFrame) -> dict[str, Any]:
    n = int(len(sub))
    if n == 0:
        return {
            "n": 0,
            "hit_rate": None,
            "baseline_rate": None,
            "edge": None,
            "mean_signed_return": None,
        }
    hit = float(sub["hit"].astype(float).mean())
    base = _num(pd.to_numeric(sub["baseline"], errors="coerce").mean())
    return {
        "n": n,
        "hit_rate": hit,
        "baseline_rate": base,
        "edge": None if base is None else hit - base,
        "mean_signed_return": _num(pd.to_numeric(sub["signed_return"], errors="coerce").mean()),
    }


def _by_horizon(sub: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(h): _stats(sub[sub["horizon"] == h]) for h in HORIZONS}


def _by_direction(sub: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    return {name: _by_horizon(sub[sub["direction"] == d]) for name, d in _DIRECTIONS}


def _rules_block(scored: pd.DataFrame, rules: list[str]) -> dict[str, Any]:
    return {r: _by_direction(scored[scored["key"] == r]) for r in rules}


def _window(
    rule_scored: pd.DataFrame, pred_scored: pd.DataFrame, rules: list[str], start: date
) -> dict[str, Any]:
    rs = rule_scored[rule_scored["date"] >= start]
    ps = pred_scored[pred_scored["date"] >= start]
    return {
        "rules": _rules_block(rs, rules),
        "overall": _by_direction(rs),
        "predictions": {
            key: {
                claim: _by_horizon(ps[(ps["key"] == key) & (ps["claim"] == h)])
                for claim, h in _CLAIMS
            }
            for key in PREDICTION_KEYS
        },
    }


def _backtest_gap(
    rule_scored: pd.DataFrame, best_history: pd.DataFrame, best_now: pd.DataFrame
) -> dict[str, Any]:
    """ルールごとに、選んだときのバックテストの勝率と、錨の5日の的中率を並べる。

    シグナルの日以前で一番新しい rule_best_history の評価を当てる。その銘柄に評価が
    まだ無いシグナルは今の rule_best_by_symbol を当て、approx を立てる。
    評価のルールがシグナルのルールと違う行は、勝率が無いので数えない（その数を
    n_unmatched に出す）。
    """
    h5 = rule_scored[rule_scored["horizon"] == ANCHOR_HORIZON]
    rules = sorted(h5["key"].unique())
    if h5.empty:
        return {}
    left = h5.assign(day=_day(h5["date"])).sort_values("day")
    if best_history.empty:
        m = left.assign(evaluated_on=pd.NaT, best_rule=None, win_rate=np.nan)
    else:
        hist = best_history[["evaluated_on", "symbol", "best_rule", "win_rate"]].copy()
        hist["evaluated_on"] = _day(hist["evaluated_on"])
        hist["symbol"] = hist["symbol"].astype(str)
        m = pd.merge_asof(
            left,
            hist.sort_values("evaluated_on"),
            left_on="day",
            right_on="evaluated_on",
            by="symbol",
            direction="backward",
        )
    approx = m["evaluated_on"].isna()
    if not best_now.empty:
        now = best_now[["symbol", "best_rule", "win_rate"]].copy()
        now["symbol"] = now["symbol"].astype(str)
        fb = m[["symbol"]].merge(now, on="symbol", how="left").set_index(m.index)
        m.loc[approx, "best_rule"] = fb.loc[approx, "best_rule"]
        m.loc[approx, "win_rate"] = fb.loc[approx, "win_rate"]
    m = m.assign(approx=approx)
    unmatched = m[m["best_rule"].notna() & (m["best_rule"] != m["key"])]["key"].value_counts()
    m = m[(m["best_rule"] == m["key"]) & m["win_rate"].notna()]
    out: dict[str, Any] = {}
    for r in rules:
        sub = m[m["key"] == r]
        n = int(len(sub))
        out[r] = {
            "backtest_win_rate": _num(sub["win_rate"].astype(float).mean()) if n else None,
            "live_hit_rate_h5": float(sub["hit"].astype(float).mean()) if n else None,
            "n_h5": n,
            "n_unmatched": int(unmatched.get(r, 0)),
            "approx": bool(sub["approx"].any()) if n else True,
        }
    return out


def _top_symbols(rule_scored: pd.DataFrame) -> list[dict[str, Any]]:
    h5 = rule_scored[rule_scored["horizon"] == ANCHOR_HORIZON]
    if h5.empty:
        return []
    g = h5.groupby(["symbol", "key", "direction"], as_index=False).agg(
        n=("hit", "size"),
        hit_rate=("hit", "mean"),
        baseline=("baseline", "mean"),
    )
    g = g.sort_values(["n", "symbol", "key", "direction"], ascending=[False, True, True, False])
    return [
        {
            "symbol": str(r.symbol),
            "rule_name": str(r.key),
            "direction": "buy" if r.direction > 0 else "sell",
            "n_h5": int(r.n),
            "hit_rate_h5": float(r.hit_rate),
            "edge_h5": _num(r.hit_rate - r.baseline),
        }
        for r in g.head(TOP_SYMBOLS_N).itertuples()
    ]


# ---------------------------------------------------------------------------
# 日次
# ---------------------------------------------------------------------------


def build_rule_accuracy(
    market: str, as_of: date, *, loaders: ScoringLoaders | None = None
) -> tuple[dict, pd.DataFrame]:
    """(rule_accuracy の JSON の dict, rule_signal_outcomes に書く行) を返す。"""
    ld = loaders or ScoringLoaders()
    px = price_frame(ld.bars(market, SINCE))
    px = px[px["date"] <= as_of].reset_index(drop=True)
    fwd = forward_returns(px, HORIZONS)

    signals = _rule_signal_frame(ld.rule_signals(market, SINCE), as_of)
    rule_scored, n_excluded = _score(signals, fwd)
    rules = sorted(signals["key"].unique())

    preds = prediction_signals(
        ld.prediction_h1(market, SINCE), ld.prediction_history(market, SINCE), px
    )
    preds = preds[(preds["date"] >= SINCE) & (preds["date"] <= as_of)]
    pred_frames = []
    for _, h in _CLAIMS:
        scored, _n = _score(preds[preds["horizon_of_claim"] == h], fwd)
        pred_frames.append(scored.assign(claim=h))
    pred_scored = pd.concat(pred_frames, ignore_index=True)

    report: dict[str, Any] = {
        "date": as_of.isoformat(),
        "generated_at": isoformat_utc(),
        "market": market,
        "anchor_horizon": ANCHOR_HORIZON,
        "baseline": "per_day_same_direction",
        "n_excluded_split": int(n_excluded),
        "windows": {
            "28d": _window(rule_scored, pred_scored, rules, as_of - timedelta(days=WINDOW_DAYS)),
            "all": _window(rule_scored, pred_scored, rules, SINCE),
        },
        "backtest_gap": _backtest_gap(
            rule_scored, ld.rule_best_history(market), ld.rule_best(market)
        ),
        "top_symbols": _top_symbols(rule_scored),
    }
    return report, _outcome_rows(rule_scored, signals, market)


def _outcome_rows(rule_scored: pd.DataFrame, signals: pd.DataFrame, market: str) -> pd.DataFrame:
    cols = [
        "signal_date",
        "market",
        "symbol",
        "rule_name",
        "signal",
        "horizon",
        "hit",
        "baseline",
        "signed_return",
        "app_version",
        "scored_at",
    ]
    if rule_scored.empty:
        return pd.DataFrame(columns=cols)
    versions = signals[["date", "symbol", "key", "app_version"]]
    m = rule_scored.merge(versions, on=["date", "symbol", "key"], how="left")
    rows = pd.DataFrame(
        {
            "signal_date": m["date"],
            "market": market,
            "symbol": m["symbol"],
            "rule_name": m["key"],
            "signal": m["direction"].astype(int),
            "horizon": m["horizon"].astype(int),
            "hit": m["hit"].astype(bool),
            "baseline": m["baseline"],
            "signed_return": m["signed_return"],
            "app_version": m["app_version"].astype(object).where(m["app_version"].notna(), None),
            "scored_at": now_utc().replace(tzinfo=None),
        }
    )
    return rows.sort_values(["signal_date", "symbol", "horizon"]).reset_index(drop=True)[cols]


def _write_json(data: dict, path: Path) -> Path:
    ensure_dir(str(path.parent))
    tmp = Path(f"{path}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)
    return path


def _out_dir(output_dir: Path | None) -> Path:
    return output_dir or Path(get_results_dir()) / "rule_accuracy"


def write_rule_accuracy(report: dict, as_of: date, output_dir: Path | None = None) -> Path:
    """results/rule_accuracy/YYYY-MM-DD.json に書く（上書き）。"""
    return _write_json(report, _out_dir(output_dir) / f"{as_of.isoformat()}.json")


def run_daily_scoring(market: str = "jp", as_of: date | None = None) -> Path:
    """全期間を採点し直し、rule_signal_outcomes と rule_accuracy の JSON を書く。"""
    as_of = as_of or (now_utc() + _JST_OFFSET).date()
    report, rows = build_rule_accuracy(market, as_of)
    n, n_deleted = db.replace_rule_outcomes(rows, market, SINCE, as_of)
    path = write_rule_accuracy(report, as_of)
    logger.info(
        "rule_accuracy: %s market=%s outcomes=%d deleted=%d app_version=%s",
        path,
        market,
        n,
        n_deleted,
        read_app_version(),
    )
    return path


# ---------------------------------------------------------------------------
# 週次
# ---------------------------------------------------------------------------


def _rule_best_diff(history: pd.DataFrame, week_end: date) -> dict[str, Any]:
    """week_end 以前で一番新しい評価と、その前の評価の、銘柄ごとのルールの出入り。"""
    out: dict[str, Any] = {
        "evaluated_on": None,
        "previous_evaluated_on": None,
        "added": [],
        "removed": [],
        "changed": [],
    }
    if history.empty:
        return out
    h = history.assign(evaluated_on=_to_date(history["evaluated_on"]))
    days = sorted(d for d in h["evaluated_on"].unique() if d <= week_end)
    if not days:
        return out
    out["evaluated_on"] = days[-1].isoformat()
    cur = {
        str(s): str(r)
        for s, r in h.loc[h["evaluated_on"] == days[-1], ["symbol", "best_rule"]].to_numpy()
    }
    prev: dict[str, str] = {}
    if len(days) > 1:
        out["previous_evaluated_on"] = days[-2].isoformat()
        prev = {
            str(s): str(r)
            for s, r in h.loc[h["evaluated_on"] == days[-2], ["symbol", "best_rule"]].to_numpy()
        }
    out["added"] = [{"symbol": s, "rule": cur[s]} for s in sorted(cur.keys() - prev.keys())]
    out["removed"] = [{"symbol": s, "rule": prev[s]} for s in sorted(prev.keys() - cur.keys())]
    out["changed"] = [
        {"symbol": s, "from": prev[s], "to": cur[s]}
        for s in sorted(cur.keys() & prev.keys())
        if cur[s] != prev[s]
    ]
    return out


def write_weekly_rule_accuracy(
    market: str,
    iso_year: int,
    iso_week: int,
    output_dir: Path | None = None,
    *,
    loaders: ScoringLoaders | None = None,
) -> Path:
    """results/rule_accuracy/weekly/YYYY-Www.json に、その週の採点とルールの出入りを書く。"""
    ld = loaders or ScoringLoaders()
    monday = date.fromisocalendar(iso_year, iso_week, 1)
    sunday = monday + timedelta(days=6)
    # 週の境は JST の月曜 0 時。scored_at は UTC の naive なので 9 時間前にずらす
    start = datetime(monday.year, monday.month, monday.day) - _JST_OFFSET
    end = start + timedelta(days=7)
    outcomes = ld.rule_outcomes_scored(market, start, end)
    if outcomes.empty:
        scored = pd.DataFrame(columns=["key", "direction", "horizon", "hit", "baseline"])
        rules: list[str] = []
    else:
        scored = outcomes.rename(columns={"rule_name": "key", "signal": "direction"})
        rules = sorted(scored["key"].astype(str).unique())
    report = {
        "iso_week": f"{iso_year}-W{iso_week:02d}",
        "week_start": monday.isoformat(),
        "week_end": sunday.isoformat(),
        "generated_at": isoformat_utc(),
        "market": market,
        "anchor_horizon": ANCHOR_HORIZON,
        "n_outcomes": int(len(outcomes)),
        "outcomes": _rules_block(scored, rules),
        "rule_best_diff": _rule_best_diff(ld.rule_best_history(market), sunday),
    }
    path = _out_dir(output_dir) / "weekly" / f"{iso_year}-W{iso_week:02d}.json"
    return _write_json(report, path)
