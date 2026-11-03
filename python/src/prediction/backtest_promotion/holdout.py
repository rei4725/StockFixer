"""
取っておいた期間（直近 20 営業日）で、候補と今の production の予想を比べる。

予想は本番と同じ芯（ensemble_diff_ratio）と同じ行の作り方（feature_row）で出し、
物差しは両方の予想がそろった行だけで baseline_metrics() から出す。
最新の日については、本番の予想の関数とこの道の production の予想が一致するかを確かめる
（道の一致の検査）。prediction_accuracy に記録された production の予想との突き合わせは
参考として書くだけで、合否には使わない。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from src.prediction.backtest_promotion.judge import (
    HOLDOUT_TRADING_DAYS,
    PATH_CHECK_SAMPLE,
    PATH_CHECK_TOL,
    judge_win,
)
from src.prediction.baseline_metrics import baseline_metrics
from src.prediction.db import load_model_weights, load_prediction_accuracy
from src.prediction.predict_unified import ensemble_diff_ratio, feature_row
from src.prediction.prediction_pipeline import _base_bar_date
from src.prediction.types import (
    UNIFIED_PREDICTION_MODEL_NAMES,
    EnsembleOutcome,
    HoldoutEvaluation,
    PredictionResult,
    WinDecision,
)
from src.utils.db import get_all_symbols, get_target_dates, load_raw_ohlcv, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HORIZON = 1
_PATH_CHECK_FAILED = "道の一致の検査に落ちた"
_DEGRADED_CANDIDATE = "候補のアンサンブルにモデルが欠けた行がある（{n} 行）"


def holdout_dates(feature_dates: pd.Series, n: int = HOLDOUT_TRADING_DAYS) -> list[pd.Timestamp]:
    """取っておく日を、重複なしの新しい方から n 個、古い順で返す。締め日はその先頭。

    Args:
        feature_dates: y が埋まった行の日付（銘柄をまたいだ重複があってよい）
        n: 取っておく日数

    Raises:
        ValueError: 日付が n 個に満たないとき
    """
    days = pd.to_datetime(pd.Series(feature_dates), errors="coerce").dropna().dt.normalize()
    unique = sorted(days.unique())
    if len(unique) < n:
        raise ValueError(f"y が埋まった日付が {len(unique)} 個しかない（{n} 個要る）")
    return [pd.Timestamp(d) for d in unique[-n:]]


def latest_holdout_dates(n: int = HOLDOUT_TRADING_DAYS) -> list[pd.Timestamp]:
    """stock_features で y が埋まった日付から、取っておく日を返す。"""
    return holdout_dates(pd.Series(get_target_dates()), n)


def _close_by_day(ohlcv: Optional[pd.DataFrame]) -> dict:
    """生の OHLCV から、日付（タイムゾーン無しの0時）→ 終値の dict を作る。"""
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return {}
    index = pd.DatetimeIndex(pd.to_datetime(ohlcv.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    closes = ohlcv["Close"].to_numpy()
    return {pd.Timestamp(d): float(c) for d, c in zip(index.normalize(), closes) if pd.notna(c)}


def _check_weight_names(models: list[tuple[str, Any]], label: str) -> None:
    """重みを引く名前が本番の名前であることを確かめる（入れ替えた後の本番の重みと同じにする）。"""
    unknown = [name for name, _ in models if name not in UNIFIED_PREDICTION_MODEL_NAMES]
    if unknown:
        raise ValueError(
            f"{label}の重みの名前が本番の名前ではない: {unknown}"
            f"（{UNIFIED_PREDICTION_MODEL_NAMES} のどれかにする）"
        )


class _WeightsRecorder:
    """load_model_weights を呼び、返った重みを名前ごとに集める（結果に小さく書くため）。"""

    def __init__(self) -> None:
        self.calls = 0
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.vectors: set = set()

    def __call__(self, market: str, symbol: str, names: list[str]) -> list[float]:
        weights = load_model_weights(market, symbol, names)
        self.calls += 1
        self.vectors.add(tuple(zip(names, (round(float(w), 6) for w in weights))))
        for name, weight in zip(names, weights):
            self.sums[name] = self.sums.get(name, 0.0) + float(weight)
            self.counts[name] = self.counts.get(name, 0) + 1
        return weights

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "distinct": len(self.vectors),
            "mean": {name: self.sums[name] / self.counts[name] for name in sorted(self.sums)},
        }


def _production_outcome(
    latest_X: pd.DataFrame,
    price: float,
    market: str,
    symbol: str,
    models: list,
    weights_fn: Callable[[str, str, list[str]], list[float]] = load_model_weights,
) -> tuple[Optional[EnsembleOutcome], bool]:
    """production の予想。全モデルが例外なら (None, True) を返す（その行は飛ばす）。"""
    try:
        return (
            ensemble_diff_ratio(
                latest_X, price, market, symbol, _HORIZON, models, weights_fn=weights_fn
            ),
            False,
        )
    except RuntimeError:
        logger.error(
            "production の予想が全モデルで失敗したため行を飛ばす: market=%s symbol=%s",
            market,
            symbol,
            exc_info=True,
        )
        return None, True


def _recorded_reference(backtest_production: pd.DataFrame, n_days: int) -> dict:
    """prediction_accuracy の production（horizon=1）の記録と、この道の production を突き合わせる。

    同じ (market, symbol, 基準の日) が何回も記録されていれば新しい回を使う。参考のみ。
    """
    empty = {"n_matched": 0, "direction_agreement": None, "median_ratio_diff": None}
    recorded = load_prediction_accuracy(
        horizon=_HORIZON, limit=0, model_name="production", days=n_days + 5
    )
    if recorded is None or recorded.empty or backtest_production.empty:
        return empty
    if not {"market", "symbol", "predicted_at", "predicted_ratio"}.issubset(recorded.columns):
        return empty

    days: list[Optional[str]] = []
    for predicted_at, market in zip(recorded["predicted_at"], recorded["market"]):
        try:
            days.append(_base_bar_date(str(predicted_at), str(market)).strftime("%Y-%m-%d"))
        except ValueError:
            days.append(None)
    recorded = recorded.assign(date=days, recorded_ratio=recorded["predicted_ratio"])
    recorded = recorded.dropna(subset=["date", "recorded_ratio"])
    recorded = recorded.sort_values("predicted_at", ascending=False).drop_duplicates(
        subset=["market", "symbol", "date"], keep="first"
    )
    merged = backtest_production.merge(
        recorded[["market", "symbol", "date", "recorded_ratio"]],
        on=["market", "symbol", "date"],
        how="inner",
    )
    if merged.empty:
        return empty
    recorded_ratio = pd.to_numeric(merged["recorded_ratio"], errors="coerce")
    backtest_ratio = merged["predicted_ratio"].astype(float)
    return {
        "n_matched": int(len(merged)),
        "direction_agreement": float(((recorded_ratio >= 0) == (backtest_ratio >= 0)).mean()),
        "median_ratio_diff": float((recorded_ratio - backtest_ratio).median()),
    }


def _daily(rows: pd.DataFrame, dates: list[str]) -> list[dict]:
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


def evaluate_holdout(
    cutoff_dates: list[pd.Timestamp],
    candidate_models: list[tuple[str, Any]],
    production_models: list[tuple[str, Any]],
    live_predict: Callable[[str, str], Optional[PredictionResult]],
) -> HoldoutEvaluation:
    """取っておいた日について候補と production を比べ、勝ちを判定する。

    Args:
        cutoff_dates: 取っておく日（holdout_dates の結果。先頭が締め日）
        candidate_models: 候補の（重みを引く名前, モデルの実物）。名前は本番の名前にする
        production_models: production の（重みを引く名前, モデルの実物）
        live_predict: 本番の予想の関数（market, symbol）。道の一致の検査に使う

    Returns:
        HoldoutEvaluation

    Raises:
        ValueError: 日・銘柄・比べられる行が無いとき、重みの名前が本番の名前でないとき
        RuntimeError: 候補の予想が全モデルで失敗した行があるとき（負けでなく失敗にする）
        Exception: 特徴量の読み込みの失敗（DB の例外）
    """
    if not cutoff_dates:
        raise ValueError("取っておく日が無い")
    _check_weight_names(candidate_models, "候補")
    _check_weight_names(production_models, "production")

    days = [pd.Timestamp(d).normalize() for d in cutoff_dates]
    day_set = set(days)
    date_strs = [d.strftime("%Y-%m-%d") for d in days]
    universe = sorted(get_all_symbols())
    if not universe:
        raise ValueError("stock_features に銘柄が無い")

    n_cand_models = len(candidate_models)
    n_prod_models = len(production_models)
    degraded = {"candidate": 0, "production": 0}
    skipped = {"no_price": 0, "candidate_none": 0, "production_none": 0, "production_error": 0}
    rows: list[dict] = []
    path_diffs: list[float] = []
    # 取っておいた日に y の埋まった行がある銘柄（上場廃止などで行の無い銘柄は数えない）
    n_symbols = 0
    path_sampled = 0
    cand_weights = _WeightsRecorder()
    prod_weights = _WeightsRecorder()

    for market, symbol in universe:
        df = load_stock_features(market, symbol, raise_on_error=True)
        if df is None or df.empty or "date" not in df.columns or "y" not in df.columns:
            continue
        closes = _close_by_day(load_raw_ohlcv(market, symbol, start_date=days[0]))
        row_days = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        # 同じ日の行が重なっていれば後ろの行を使う
        positions = {
            pd.Timestamp(d): i for i, d in enumerate(row_days) if pd.notna(d) and d in day_set
        }
        target_positions = [
            (day, pos) for day, pos in sorted(positions.items()) if pd.notna(df["y"].iloc[pos])
        ]
        if not target_positions:
            continue
        n_symbols += 1
        for day, pos in target_positions:
            actual = df["y"].iloc[pos]
            price = closes.get(day)
            if price is None:
                skipped["no_price"] += 1
                continue
            latest_X = feature_row(df, pos, market)
            cand = ensemble_diff_ratio(
                latest_X,
                price,
                market,
                symbol,
                _HORIZON,
                candidate_models,
                weights_fn=cand_weights,
            )
            prod, prod_failed = _production_outcome(
                latest_X, price, market, symbol, production_models, prod_weights
            )
            if prod_failed:
                skipped["production_error"] += 1
                continue
            if cand is None:
                skipped["candidate_none"] += 1
                continue
            if prod is None:
                skipped["production_none"] += 1
                continue
            if cand.model_count < n_cand_models:
                degraded["candidate"] += 1
            if prod.model_count < n_prod_models:
                degraded["production"] += 1
            rows.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "date": day.strftime("%Y-%m-%d"),
                    "actual_ratio": float(actual),
                    "candidate_ratio": cand.diff_ratio,
                    "production_ratio": prod.diff_ratio,
                }
            )

        # 先頭の PATH_CHECK_SAMPLE 銘柄（market, symbol の順）で道の一致を確かめる
        if path_sampled < PATH_CHECK_SAMPLE:
            path_sampled += 1
            diff = _path_diff(df, market, symbol, production_models, live_predict)
            if diff is not None:
                path_diffs.append(diff)

    if not rows:
        raise ValueError("候補と production の予想がそろった行が無い")

    table = pd.DataFrame(rows)
    candidate = baseline_metrics(table.rename(columns={"candidate_ratio": "predicted_ratio"}))
    production = baseline_metrics(table.rename(columns={"production_ratio": "predicted_ratio"}))
    daily = _daily(table, date_strs)

    max_abs_diff = max(path_diffs) if path_diffs else None
    path_check = {
        "compared": len(path_diffs),
        "max_abs_diff": max_abs_diff,
        "passed": bool(path_diffs) and max_abs_diff is not None and max_abs_diff <= PATH_CHECK_TOL,
    }

    # 物差しの無い日は勝ちに数えず、日数には入れる
    daily_diff = [0.0 if d["diff"] is None else float(d["diff"]) for d in daily]
    decision = judge_win(candidate, production, daily_diff, len(days) * n_symbols)
    reasons = list(decision.reasons)
    if degraded["candidate"]:
        reasons.append(_DEGRADED_CANDIDATE.format(n=degraded["candidate"]))
    if not path_check["passed"]:
        reasons.append(_PATH_CHECK_FAILED)
    decision = WinDecision(won=not reasons, reasons=reasons)

    reference = _recorded_reference(
        table[["market", "symbol", "date", "production_ratio"]].rename(
            columns={"production_ratio": "predicted_ratio"}
        ),
        len(days),
    )

    return HoldoutEvaluation(
        cutoff=date_strs[0],
        dates=date_strs,
        n_symbols=n_symbols,
        candidate=candidate,
        production=production,
        daily=daily,
        path_check=path_check,
        recorded_reference=reference,
        decision=decision,
        degraded_rows=degraded,
        skipped_rows=skipped,
        weights={"candidate": cand_weights.summary(), "production": prod_weights.summary()},
    )


def _path_diff(
    df: pd.DataFrame,
    market: str,
    symbol: str,
    production_models: list,
    live_predict: Callable[[str, str], Optional[PredictionResult]],
) -> Optional[float]:
    """最後の行について、この道の production と本番の予想の diff_ratio の差の絶対値。

    値段の出どころの違いを道の違いと取り違えないよう、この道にも本番の予想の基準の値段
    （current_price）を渡す。本番が None か、この道で予想が出ない（モデルが無い・
    全モデルが例外）なら None。
    """
    live = live_predict(market, symbol)
    if live is None:
        return None
    outcome, failed = _production_outcome(
        feature_row(df, -1, market), float(live.current_price), market, symbol, production_models
    )
    if failed or outcome is None:
        return None
    return abs(float(live.diff_ratio) - outcome.diff_ratio)
