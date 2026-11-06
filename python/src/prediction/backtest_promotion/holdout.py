"""
取っておいた期間（直近 20 営業日）で、候補と今の production の予想を比べる。

予想は本番と同じ芯（ensemble_diff_ratio）と同じ行の作り方（feature_row）で出し、
物差しは両方の予想がそろった行だけで baseline_metrics() から出す。
発注する市場（jp）は全市場の線とは別に、比べられた行と行のある銘柄の数を線と比べる（#1693）。
行のある日の数も、東証の営業日の数を分母にして線と比べる（#1699）。
最新の日については、本番の予想の関数とこの道の production の予想が一致するかを確かめる
（道の一致の検査）。prediction_accuracy に記録された production の予想との突き合わせは
参考として書くだけで、合否には使わない。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from src.domain.jp_market_holidays import is_jp_business_day
from src.prediction.backtest_promotion.judge import (
    HOLDOUT_TRADING_DAYS,
    MIN_COVERAGE,
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
_PATH_CHECK_NOTHING = "道の一致の検査で比べられた銘柄が無い（本番の予想が出ない）"
_PATH_CHECK_NO_JP = "道の一致の検査で jp の比べられた銘柄が無い"
# 発注する市場。ここを1銘柄も比べられなかった回は passed にしない（#1675）
_PATH_CHECK_REQUIRED_MARKET = "jp"
# 道の一致の検査で銘柄を飛ばした理由
_PATH_SKIP_NO_BASE_PRICE = "no_base_price"
_PATH_SKIP_STALE = "stale_features"
_PATH_SKIP_LIVE_NONE = "live_none"
_PATH_SKIP_PRODUCTION_NONE = "production_none"
_PATH_SKIP_REASONS = (
    _PATH_SKIP_NO_BASE_PRICE,
    _PATH_SKIP_STALE,
    _PATH_SKIP_LIVE_NONE,
    _PATH_SKIP_PRODUCTION_NONE,
)
_DEGRADED_CANDIDATE = "候補のアンサンブルにモデルが欠けた行がある（{n} 行）"
_REQUIRED_NO_SYMBOLS = f"発注する市場 {_PATH_CHECK_REQUIRED_MARKET} の銘柄が無い"
_REQUIRED_ROWS_SHORT = (
    f"発注する市場 {_PATH_CHECK_REQUIRED_MARKET} の比べられた行が足りない"
    "（{n} 行、見込み {expected} 行、必要 {need:g} 以上）"
)
_REQUIRED_SYMBOLS_SHORT = (
    f"発注する市場 {_PATH_CHECK_REQUIRED_MARKET} の行のある銘柄が足りない"
    "（{n} 銘柄、全 {universe} 銘柄、必要 {need:g} 以上）"
)
_REQUIRED_DAYS_SHORT = (
    f"発注する市場 {_PATH_CHECK_REQUIRED_MARKET} の行のある営業日が足りない"
    "（{n} 日、東証の営業日 {business} 日、必要 {need:g} 以上）"
)


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
    weights_fn: Callable[[str, str, list[str]], list[float]],
) -> tuple[Optional[EnsembleOutcome], bool]:
    """production の予想。全モデルが例外なら (None, True) を返す（その行は飛ばす）。

    weights_fn は呼び出し元で渡す（既定値にすると定義の時に束縛され、差し替えが届かない）。
    """
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


def _path_check_quotas(universe: list[tuple[str, str]]) -> dict[str, int]:
    """道の一致の検査に使う銘柄の数を、銘柄のある市場で PATH_CHECK_SAMPLE を等しく分けて決める。

    割り切れない分は市場の名前の順に1つずつ足す（毎回同じ分け方にする）。
    """
    markets = sorted({market for market, _ in universe})
    base, extra = divmod(PATH_CHECK_SAMPLE, len(markets))
    return {market: base + (1 if i < extra else 0) for i, market in enumerate(markets)}


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


def _both_metrics(rows: pd.DataFrame) -> tuple[dict, dict]:
    """同じ行での候補と production の baseline_metrics()。"""
    return (
        baseline_metrics(rows.rename(columns={"candidate_ratio": "predicted_ratio"})),
        baseline_metrics(rows.rename(columns={"production_ratio": "predicted_ratio"})),
    )


def _market_summaries(
    table: pd.DataFrame,
    universe: list[tuple[str, str]],
    market_symbols: dict[str, int],
    market_days: dict[str, set],
    days: list[pd.Timestamp],
) -> dict[str, dict]:
    """universe の市場ごとの、行の見込みと、比べられた行の数と物差し。

    行のある銘柄が1つも無い市場も書く（その市場が丸ごと欠けた回を結果から読めるようにする）。
    採点の行の見込みは（その市場で y のある日の数）×（行のある銘柄の数）。日は jp と us を
    合わせたものなので、全部の日 × 全部の銘柄だと休場日の分だけ足りなくなる。
    発注する市場には、取っておいた日のうちの東証の営業日の数と、そのうち行のある日の数も書く。
    表に無い年は平日を全部営業日に数えるので、分母が大きくなる側に倒れる。
    """
    summaries: dict[str, dict] = {}
    for market in sorted({m for m, _ in universe}):
        n_symbols = market_symbols.get(market, 0)
        n_dates = len(market_days.get(market, ()))
        candidate, production = _both_metrics(table[table["market"] == market])
        summaries[market] = {
            "n_universe_symbols": sum(1 for m, _ in universe if m == market),
            "n_symbols": n_symbols,
            "n_dates": n_dates,
            "expected_rows": n_dates * n_symbols,
            "n_compared": min(candidate["n_baseline"], production["n_baseline"]),
            "candidate": candidate,
            "production": production,
        }
    if _PATH_CHECK_REQUIRED_MARKET in summaries:
        business_days = {day for day in days if is_jp_business_day(day.date())}
        rows_days = market_days.get(_PATH_CHECK_REQUIRED_MARKET, set())
        summaries[_PATH_CHECK_REQUIRED_MARKET].update(
            n_business_dates=len(business_days),
            n_business_dates_with_rows=len(business_days & rows_days),
        )
    return summaries


def _required_market_reasons(markets: dict[str, dict]) -> list[str]:
    """発注する市場の行か銘柄か日が線に届かないときの、勝ちにしない理由を返す。

    銘柄の分母は universe の銘柄の数、日の分母は東証の営業日の数。行の見込みは行のある銘柄と
    日から数えるので、銘柄や日がごっそり欠けると一緒に縮み、行の線だけでは止まらない。
    universe に発注する市場が無い回は、道の一致の検査（#1675）と同じく勝ちにしない。
    """
    summary = markets.get(_PATH_CHECK_REQUIRED_MARKET)
    if summary is None:
        return [_REQUIRED_NO_SYMBOLS]
    reasons: list[str] = []
    need_rows = summary["expected_rows"] * MIN_COVERAGE
    if summary["n_compared"] < need_rows:
        reasons.append(
            _REQUIRED_ROWS_SHORT.format(
                n=summary["n_compared"], expected=summary["expected_rows"], need=need_rows
            )
        )
    need_symbols = summary["n_universe_symbols"] * MIN_COVERAGE
    if summary["n_symbols"] < need_symbols:
        reasons.append(
            _REQUIRED_SYMBOLS_SHORT.format(
                n=summary["n_symbols"], universe=summary["n_universe_symbols"], need=need_symbols
            )
        )
    need_days = summary["n_business_dates"] * MIN_COVERAGE
    if summary["n_business_dates_with_rows"] < need_days:
        reasons.append(
            _REQUIRED_DAYS_SHORT.format(
                n=summary["n_business_dates_with_rows"],
                business=summary["n_business_dates"],
                need=need_days,
            )
        )
    return reasons


def evaluate_holdout(
    cutoff_dates: list[pd.Timestamp],
    candidate_models: list[tuple[str, Any]],
    production_models: list[tuple[str, Any]],
    live_predict: Callable[..., Optional[PredictionResult]],
) -> HoldoutEvaluation:
    """取っておいた日について候補と production を比べ、勝ちを判定する。

    Args:
        cutoff_dates: 取っておく日（holdout_dates の結果。先頭が締め日）
        candidate_models: 候補の（重みを引く名前, モデルの実物）。名前は本番の名前にする
        production_models: production の（重みを引く名前, モデルの実物）
        live_predict: 本番の予想の関数（market, symbol, base_price=, on_stale=）。
            道の一致の検査に使う

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
    # 市場ごとの、y の埋まった行がある銘柄の数と日（休場日の違いで日数が市場ごとに違う）
    market_symbols: dict[str, int] = {}
    market_days: dict[str, set] = {}
    # 道の一致の検査は市場ごとに分けて取る（market, symbol の順だと jp だけになる）
    path_quotas = _path_check_quotas(universe)
    path_compared: dict[str, int] = {market: 0 for market in path_quotas}
    path_skipped: dict[str, int] = {reason: 0 for reason in _PATH_SKIP_REASONS}
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
        market_symbols[market] = market_symbols.get(market, 0) + 1
        market_days.setdefault(market, set()).update(day for day, _ in target_positions)
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
            # 欠けは予想した行ごとに数える（相手側で飛ばす行でも、候補の欠けは勝ちを止める）
            if cand is not None and cand.model_count < n_cand_models:
                degraded["candidate"] += 1
            prod, prod_failed = _production_outcome(
                latest_X, price, market, symbol, production_models, prod_weights
            )
            if prod is not None and prod.model_count < n_prod_models:
                degraded["production"] += 1
            if prod_failed:
                skipped["production_error"] += 1
                continue
            if cand is None:
                skipped["candidate_none"] += 1
                continue
            if prod is None:
                skipped["production_none"] += 1
                continue
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

        # 市場ごとに、比べられた銘柄が割り当ての数になるまで（symbol の順に）道の一致を確かめる。
        # 飛ばした銘柄は割り当てを使わないので、先頭の銘柄が飛んでも次の銘柄へ進む
        if path_compared[market] < path_quotas[market]:
            diff, skip_reason = _path_diff(
                df, closes.get(row_days.iloc[-1]), market, symbol, production_models, live_predict
            )
            if diff is not None:
                path_diffs.append(diff)
                path_compared[market] += 1
            elif skip_reason is not None:
                path_skipped[skip_reason] += 1

    if not rows:
        raise ValueError("候補と production の予想がそろった行が無い")

    table = pd.DataFrame(rows)
    candidate, production = _both_metrics(table)
    daily = _daily(table, date_strs)

    max_abs_diff = max(path_diffs) if path_diffs else None
    required_compared = path_compared.get(_PATH_CHECK_REQUIRED_MARKET, 0) > 0
    path_check = {
        "compared": len(path_diffs),
        "max_abs_diff": max_abs_diff,
        "passed": required_compared and max_abs_diff is not None and max_abs_diff <= PATH_CHECK_TOL,
        "compared_by_market": path_compared,
        "skipped_by_reason": path_skipped,
    }

    markets = _market_summaries(table, universe, market_symbols, market_days, days)
    expected_rows = sum(m["expected_rows"] for m in markets.values())

    # 物差しの無い日は勝ちに数えず、日数には入れる
    daily_diff = [0.0 if d["diff"] is None else float(d["diff"]) for d in daily]
    decision = judge_win(candidate, production, daily_diff, expected_rows)
    reasons = list(decision.reasons)
    if degraded["candidate"]:
        reasons.append(_DEGRADED_CANDIDATE.format(n=degraded["candidate"]))
    if not path_diffs:
        reasons.append(_PATH_CHECK_NOTHING)
    else:
        if not required_compared:
            reasons.append(_PATH_CHECK_NO_JP)
        if max_abs_diff is not None and max_abs_diff > PATH_CHECK_TOL:
            reasons.append(_PATH_CHECK_FAILED)
    reasons.extend(_required_market_reasons(markets))
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
        expected_rows=expected_rows,
        markets=markets,
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
    base_price: Optional[float],
    market: str,
    symbol: str,
    production_models: list,
    live_predict: Callable[..., Optional[PredictionResult]],
) -> tuple[Optional[float], Optional[str]]:
    """最後の行について、この道の production と本番の予想の diff_ratio の差の絶対値を出す。

    基準の値段は market_data_raw の最後の行の日の終値を本番の予想へ渡す。yfinance の最後の足は
    22:00 JST には特徴量の最後の行より新しいことがあり、それだと比べられない（#1675）。
    値段の出どころの違いを道の違いと取り違えないよう、この道にも本番の予想の基準の値段
    （current_price）を渡す。

    Args:
        df: その銘柄の特徴量
        base_price: 最後の行の日の終値。無ければ比べない

    Returns:
        (差の絶対値, None)。比べられなければ (None, 飛ばした理由)
    """
    if base_price is None:
        return None, _PATH_SKIP_NO_BASE_PRICE
    stale: list[str] = []
    live = live_predict(
        market, symbol, base_price=base_price, on_stale=lambda _m, s: stale.append(s)
    )
    if live is None:
        return None, _PATH_SKIP_STALE if stale else _PATH_SKIP_LIVE_NONE
    outcome, failed = _production_outcome(
        feature_row(df, -1, market),
        float(live.current_price),
        market,
        symbol,
        production_models,
        weights_fn=load_model_weights,
    )
    if failed or outcome is None:
        return None, _PATH_SKIP_PRODUCTION_NONE
    return abs(float(live.diff_ratio) - outcome.diff_ratio), None
