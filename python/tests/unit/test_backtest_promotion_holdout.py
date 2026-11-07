"""
取っておいた期間の評価（evaluate_holdout）と道の一致の検査のテスト

DB の読み込みはすべて差し替える。候補は特徴量 feat_a をそのまま言い（y と同じ）、
production はその逆を言う。
"""

from dataclasses import asdict
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain import jp_market_holidays
from src.prediction import predict_unified
from src.prediction.backtest_promotion import holdout, required_market
from src.prediction.backtest_promotion.holdout import (
    evaluate_holdout,
    holdout_dates,
    latest_holdout_dates,
)
from src.prediction.backtest_promotion.judge import PATH_CHECK_TOL
from src.prediction.predict_unified import ensemble_diff_ratio as real_ensemble
from src.prediction.predict_unified import feature_row
from src.prediction.types import HoldoutEvaluation, PredictionResult

NAMES = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
SYMBOLS = [("jp", "1111"), ("us", "AAA")]
DAYS = pd.bdate_range("2026-10-01", periods=25)
FEATURES = ["feat_a", "atr_lag1", "market_encoded"]


class _Model:
    """feat_a に scale を掛けた値を言うモデル。raise_on に入った feat_a では例外を出す"""

    def __init__(self, scale: float, raise_on: Optional[set] = None):
        self.feature_names_in_ = list(FEATURES)
        self.scale = scale
        self.raise_on = raise_on or set()

    def predict(self, X: pd.DataFrame):
        value = float(X["feat_a"].iloc[0])
        if value in self.raise_on:
            raise ValueError("壊れた")
        return [value * self.scale]


def _y(i: int, j: int) -> float:
    # 1e-5 * i で行ごとに値を変え、feat_a で行を見分けられるようにする
    return 0.01 * ((i + j) % 5 - 2) + 0.001 * (j + 1) + 1e-5 * i


def _features(j: int, atr: float = 10.0) -> pd.DataFrame:
    ys = [_y(i, j) for i in range(len(DAYS))]
    feat = list(ys)
    ys[-1] = np.nan  # 最後の日は翌日の値がまだ無い
    return pd.DataFrame(
        {
            "date": DAYS,
            "feat_a": feat,
            "atr_lag1": [atr] * len(DAYS),
            "market_encoded": [0] * len(DAYS),
            "y": ys,
        }
    )


def _ohlcv(drop: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    days = [d for d in DAYS if d != drop]
    df = pd.DataFrame({"Close": [100.0] * len(days)}, index=pd.DatetimeIndex(days, name="Date"))
    return df


def _patch(
    monkeypatch, *, weights_calls=None, drop_price=None, accuracy=None, atr=10.0, extra=None
):
    frames = {s: _features(j, atr) for j, s in enumerate(SYMBOLS)}
    frames.update(extra or {})
    monkeypatch.setattr(holdout, "get_all_symbols", lambda: sorted(frames))
    # 本物の config/watchlist.json を読まない。既定は表の jp の銘柄と同じ並びにする
    monkeypatch.setattr(
        holdout,
        "load_watchlist_symbols",
        lambda market: {s for m, s in frames if m == market},
    )
    monkeypatch.setattr(
        holdout, "load_stock_features", lambda m, s, raise_on_error=False: frames[(m, s)].copy()
    )
    monkeypatch.setattr(
        holdout,
        "load_raw_ohlcv",
        lambda m, s, start_date=None: _ohlcv(drop_price if s == "AAA" else None),
    )

    def _weights(market, symbol, names):
        if weights_calls is not None:
            weights_calls.append(list(names))
        return [1.0 / len(names)] * len(names)

    monkeypatch.setattr(holdout, "load_model_weights", _weights)
    monkeypatch.setattr(
        holdout,
        "load_prediction_accuracy",
        lambda **kw: accuracy if accuracy is not None else pd.DataFrame(),
    )
    return frames


def _models(scale: float, raise_on: Optional[set] = None):
    return [(name, _Model(scale, raise_on)) for name in NAMES]


def _live_matching(offset: float = 0.0):
    """production（-feat_a）と同じ値に offset を足した本番の予想を返す"""

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        j = SYMBOLS.index((market, symbol))
        diff = -_features(j)["feat_a"].iloc[-1] + offset
        return PredictionResult(
            market=market,
            symbol=symbol,
            current_price=100.0,
            avg_pred_price=100.0 * (1 + diff),
            diff_ratio=diff,
            model_count=2,
        )

    return _live


def _dates() -> list:
    return list(DAYS[-21:-1])


def test_holdout_dates_are_latest_n_dates_with_target():
    # 25日分、最後の日は y が NaN → 最後から2番目までの20日（2銘柄ぶんの重複も1つに）
    df = pd.concat([_features(0), _features(1)])
    dates = holdout_dates(df.loc[df["y"].notna(), "date"], 20)
    assert dates == _dates()
    assert dates[0] == DAYS[4]


def test_holdout_dates_raise_when_too_few():
    with pytest.raises(ValueError):
        holdout_dates(pd.Series(DAYS[:5]), 20)


def test_latest_holdout_dates_reads_target_dates(monkeypatch):
    monkeypatch.setattr(holdout, "get_target_dates", lambda: list(DAYS[:-1]))
    assert latest_holdout_dates(20) == _dates()


def test_candidate_wins_when_better_and_path_matches(monkeypatch):
    _patch(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert isinstance(ev, HoldoutEvaluation)
    assert ev.cutoff == "2026-10-07"
    assert ev.dates[0] == "2026-10-07" and len(ev.dates) == 20
    assert ev.n_symbols == 2
    assert ev.candidate["n_baseline"] == ev.production["n_baseline"] == 40
    assert ev.path_check["compared"] == 2
    assert ev.path_check["passed"] is True
    assert len(ev.daily) == 20
    assert all(d["diff"] > 0 for d in ev.daily)
    assert ev.decision.won is True, ev.decision.reasons
    assert ev.degraded_rows == {"candidate": 0, "production": 0}


def test_metrics_use_only_rows_both_models_predicted(monkeypatch):
    # 候補が1銘柄の1日でモデル無し（None）を返す → その行は production の物差しからも外れる
    _patch(monkeypatch)
    cand = _models(1.0)
    target = _features(1)["feat_a"].iloc[-5]

    def _ensemble(latest_X, current_price, market, symbol, horizon, models, weights_fn):
        if models is cand and symbol == "AAA" and latest_X["feat_a"].iloc[0] == target:
            return None
        return real_ensemble(
            latest_X, current_price, market, symbol, horizon, models, weights_fn=weights_fn
        )

    monkeypatch.setattr(holdout, "ensemble_diff_ratio", _ensemble)
    ev = evaluate_holdout(_dates(), cand, _models(-1.0), _live_matching())
    assert ev.candidate["n_baseline"] == ev.production["n_baseline"] == 39
    assert ev.skipped_rows["candidate_none"] == 1


def test_missing_price_skips_row_for_both(monkeypatch):
    _patch(monkeypatch, drop_price=DAYS[10])
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.candidate["n_baseline"] == ev.production["n_baseline"] == 39
    assert ev.skipped_rows["no_price"] == 1


def test_candidate_uses_production_weight_names(monkeypatch):
    # weights_fn に渡る名前が候補でも ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    calls: list = []
    _patch(monkeypatch, weights_calls=calls)
    evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert calls and all(names == NAMES for names in calls)


def test_candidate_names_other_than_production_raise(monkeypatch):
    _patch(monkeypatch)
    cand = [(f"{name}_candidate", _Model(1.0)) for name in NAMES]
    with pytest.raises(ValueError):
        evaluate_holdout(_dates(), cand, _models(-1.0), _live_matching())


def test_path_check_failure_blocks_win(monkeypatch):
    # live_predict が 0.001 ずれた diff_ratio を返す → passed False、decision.won False
    _patch(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching(0.001))
    assert ev.path_check["compared"] == 2
    assert ev.path_check["max_abs_diff"] == pytest.approx(0.001)
    assert ev.path_check["passed"] is False
    assert ev.decision.won is False
    assert "道の一致の検査に落ちた" in ev.decision.reasons


def test_path_check_ignores_live_none(monkeypatch):
    _patch(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), lambda m, s, **kw: None)
    assert ev.path_check["compared"] == 0
    assert ev.path_check["passed"] is False
    assert ev.decision.won is False
    # 食い違いではなく、比べられなかったと分かる理由にする
    assert "道の一致の検査で比べられた銘柄が無い（本番の予想が出ない）" in ev.decision.reasons
    assert "道の一致の検査に落ちた" not in ev.decision.reasons


def test_degraded_candidate_cannot_win(monkeypatch):
    _patch(monkeypatch)
    broken = _features(0)["feat_a"].iloc[-3]
    cand = [
        (NAMES[0], _Model(1.0, raise_on={broken})),
        (NAMES[1], _Model(1.0)),
    ]
    ev = evaluate_holdout(_dates(), cand, _models(-1.0), _live_matching())
    assert ev.degraded_rows["candidate"] == 1
    assert ev.decision.won is False
    assert any("欠け" in r for r in ev.decision.reasons)


def test_candidate_all_models_raising_propagates(monkeypatch):
    _patch(monkeypatch)
    broken = _features(0)["feat_a"].iloc[-3]
    with pytest.raises(RuntimeError):
        evaluate_holdout(_dates(), _models(1.0, raise_on={broken}), _models(-1.0), _live_matching())


def test_production_all_models_raising_skips_row(monkeypatch):
    _patch(monkeypatch)
    broken = _features(0)["feat_a"].iloc[-3]
    ev = evaluate_holdout(
        _dates(), _models(1.0), _models(-1.0, raise_on={broken}), _live_matching()
    )
    assert ev.skipped_rows["production_error"] == 1
    assert ev.candidate["n_baseline"] == ev.production["n_baseline"] == 39


def test_empty_universe_is_failure(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(holdout, "get_all_symbols", lambda: [])
    with pytest.raises(ValueError):
        evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())


def test_recorded_reference_compares_recorded_production(monkeypatch):
    # jp の 10:00 UTC は引けの後なので、その日の足の予想になる
    feats = _features(0)
    rows = [
        {
            "market": "jp",
            "symbol": "1111",
            "predicted_at": f"{d:%Y%m%d}_100000",
            "predicted_ratio": -float(feats.loc[feats["date"] == d, "feat_a"].iloc[0]) + 1e-5,
        }
        for d in _dates()[:3]
    ]
    _patch(monkeypatch, accuracy=pd.DataFrame(rows))
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    ref = ev.recorded_reference
    assert ref["n_matched"] == 3
    assert ref["direction_agreement"] == pytest.approx(1.0)
    assert ref["median_ratio_diff"] == pytest.approx(1e-5)


def test_symbol_without_rows_in_window_is_not_counted(monkeypatch):
    # 上場廃止で取っておいた期間に行の無い銘柄は n_symbols と expected_rows に入れない
    delisted = _features(0).assign(date=pd.bdate_range("2025-01-01", periods=len(DAYS)))
    _patch(monkeypatch, extra={("us", "ZZZ"): delisted})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.n_symbols == 2
    assert ev.decision.won is True, ev.decision.reasons


def test_path_check_uses_live_price(monkeypatch):
    # atr_lag1=0.5・価格100 で上限は ±0.015 となりクリップが効く。本番の値段は DB の終値より 1% 高い
    calls: list = []
    _patch(monkeypatch, atr=0.5, weights_calls=calls)
    prod = _models(-1.0)

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        row = feature_row(_features(SYMBOLS.index((market, symbol)), 0.5), -1, market)
        out = real_ensemble(row, 101.0, market, symbol, 1, prod, weights_fn=lambda *a: [0.5, 0.5])
        at_db_price = real_ensemble(
            row, 100.0, market, symbol, 1, prod, weights_fn=lambda *a: [0.5, 0.5]
        )
        # 値段の違いだけで許容差を超える（DB の終値で比べると落ちる）
        assert abs(out.diff_ratio - at_db_price.diff_ratio) > PATH_CHECK_TOL
        return PredictionResult(
            market=market,
            symbol=symbol,
            current_price=101.0,
            avg_pred_price=101.0 * (1 + out.diff_ratio),
            diff_ratio=out.diff_ratio,
            model_count=2,
        )

    ev = evaluate_holdout(_dates(), _models(1.0), prod, _live)
    assert ev.path_check["compared"] == 2
    assert ev.path_check["passed"] is True
    # 道の検査も差し替えた重みを引く（取っておいた行 40 × 2 組 + 道の検査 2 銘柄）
    assert len(calls) == 40 * 2 + 2


def test_weights_summary_is_recorded(monkeypatch):
    _patch(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    for label in ("candidate", "production"):
        summary = ev.weights[label]
        assert summary["calls"] == 40
        assert summary["distinct"] == 1
        assert summary["mean"] == pytest.approx({name: 0.5 for name in NAMES})


def test_degraded_candidate_on_row_production_failed_still_blocks_win(monkeypatch):
    # production が全モデルで落ちて飛ばす行でも、候補の欠けは数えて勝ちを止める
    _patch(monkeypatch)
    broken = _features(0)["feat_a"].iloc[-3]
    cand = [
        (NAMES[0], _Model(1.0, raise_on={broken})),
        (NAMES[1], _Model(1.0)),
    ]
    ev = evaluate_holdout(_dates(), cand, _models(-1.0, raise_on={broken}), _live_matching())
    assert ev.skipped_rows["production_error"] == 1
    assert ev.degraded_rows["candidate"] == 1
    assert ev.decision.won is False


def test_expected_rows_follow_each_markets_dates(monkeypatch):
    # jp だけ 9 日休場（行が無い）。日は jp と us を合わせた 20 日なので、
    # 20 日 × 2 銘柄 = 40 の 80%（32）だと jp の休場だけで負ける。市場ごとに数えれば勝てる。
    # 本物の表にこの長さの休場は無いので、9 日を休場として差し替える（#1699 の日の線を越える）
    holiday = set(_dates()[3:12])
    jp = _features(0)
    jp = jp[~jp["date"].isin(holiday)].reset_index(drop=True)
    _patch(monkeypatch, extra={("jp", "1111"): jp})
    monkeypatch.setattr(holdout, "is_jp_business_day", lambda d: pd.Timestamp(d) not in holiday)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.candidate["n_baseline"] == 31
    keys = ("n_symbols", "n_dates", "expected_rows")
    assert {m: {k: v[k] for k in keys} for m, v in ev.markets.items()} == {
        "jp": {"n_symbols": 1, "n_dates": 11, "expected_rows": 11},
        "us": {"n_symbols": 1, "n_dates": 20, "expected_rows": 20},
    }
    assert ev.expected_rows == 31
    assert ev.decision.won is True, ev.decision.reasons


def test_path_check_samples_each_market(monkeypatch):
    # 銘柄は jp が先に並ぶ。上限 2 を市場で分け、jp と us から 1 銘柄ずつ確かめる
    extra = {("jp", "2222"): _features(0), ("jp", "3333"): _features(0)}
    _patch(monkeypatch, extra=extra)
    monkeypatch.setattr(holdout, "PATH_CHECK_SAMPLE", 2)
    asked: list = []
    live = _live_matching()

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        asked.append((market, symbol))
        return live(market, symbol) if (market, symbol) in SYMBOLS else None

    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live)
    assert asked == [("jp", "1111"), ("us", "AAA")]
    assert ev.path_check["compared"] == 2
    assert ev.path_check["compared_by_market"] == {"jp": 1, "us": 1}


def test_path_check_quotas_split_evenly_in_market_order(monkeypatch):
    universe = [("jp", "1"), ("jp", "2"), ("us", "A")]
    monkeypatch.setattr(holdout, "PATH_CHECK_SAMPLE", 3)
    assert holdout._path_check_quotas(universe) == {"jp": 2, "us": 1}
    monkeypatch.setattr(holdout, "PATH_CHECK_SAMPLE", 50)
    assert holdout._path_check_quotas(universe) == {"jp": 25, "us": 25}


def _path_skips(**counts: int) -> dict:
    return {
        "no_base_price": 0,
        "stale_features": 0,
        "live_none": 0,
        "production_none": 0,
        **counts,
    }


def _live_through_real_path(monkeypatch, prod: list, history: pd.DataFrame):
    """本番の予想の関数を、yfinance の足だけ差し替えて通す。"""
    by_name = dict(prod)
    frames = {key: _features(j) for j, key in enumerate(SYMBOLS)}
    ticker = MagicMock()
    ticker.history.return_value = history
    monkeypatch.setattr(predict_unified.yf, "Ticker", MagicMock(return_value=ticker))
    monkeypatch.setattr(predict_unified, "get_service_url", lambda: None)
    monkeypatch.setattr(predict_unified, "get_ticker", lambda m, s: s)
    monkeypatch.setattr(predict_unified, "get_cached_model", by_name.get)
    monkeypatch.setattr(predict_unified, "load_model_weights", lambda m, s, n: [0.5, 0.5])
    monkeypatch.setattr(predict_unified, "load_feature_data", lambda m, s: frames[(m, s)].copy())
    monkeypatch.setattr(predict_unified, "count_weekdays_behind", lambda *a, **kw: 1)
    return predict_unified.predict_with_unified_model


NEWER_BAR = pd.DataFrame(
    {"Close": [999.0]}, index=pd.DatetimeIndex([DAYS[-1] + pd.Timedelta(days=1)])
)


@pytest.mark.parametrize(
    "history",
    [NEWER_BAR, pd.DataFrame()],
    ids=["営業日の22時で足が特徴量より新しい", "休場日で足が無い"],
)
def test_path_check_compares_jp_without_relying_on_the_last_bar(monkeypatch, history):
    _patch(monkeypatch)
    prod = _models(-1.0)
    live = _live_through_real_path(monkeypatch, prod, history)
    ev = evaluate_holdout(_dates(), _models(1.0), prod, live)
    assert ev.path_check["compared_by_market"] == {"jp": 1, "us": 1}
    assert ev.path_check["max_abs_diff"] <= PATH_CHECK_TOL
    assert ev.path_check["passed"] is True
    assert ev.path_check["skipped_by_reason"] == _path_skips()
    assert ev.decision.won is True, ev.decision.reasons


def test_daily_path_with_the_same_bar_still_skips(monkeypatch):
    # 引数を渡さない日次の呼び方は、足の日付の判定（#845）で今までどおり飛ばす
    live = _live_through_real_path(monkeypatch, _models(-1.0), NEWER_BAR)
    assert live("jp", "1111") is None


def test_path_check_gives_the_close_of_the_last_feature_row(monkeypatch):
    _patch(monkeypatch)
    prices: list = []
    live = _live_matching()

    def _live(market: str, symbol: str, **kwargs) -> Optional[PredictionResult]:
        prices.append(kwargs["base_price"])
        return live(market, symbol)

    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live)
    assert prices == [100.0, 100.0]
    assert ev.path_check["passed"] is True


def test_no_jp_compared_is_not_passed_even_if_us_matches(monkeypatch):
    _patch(monkeypatch)
    live = _live_matching()

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        return None if market == "jp" else live(market, symbol)

    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live)
    assert ev.path_check["compared_by_market"] == {"jp": 0, "us": 1}
    assert ev.path_check["max_abs_diff"] <= PATH_CHECK_TOL
    assert ev.path_check["passed"] is False
    assert ev.path_check["skipped_by_reason"] == _path_skips(live_none=1)
    assert ev.decision.won is False
    assert "道の一致の検査で jp の比べられた銘柄が無い" in ev.decision.reasons
    assert "道の一致の検査に落ちた" not in ev.decision.reasons


def test_no_us_compared_still_passes(monkeypatch):
    # 発注するのは jp なので、us の 0 件は passed の条件に入れない
    _patch(monkeypatch)
    live = _live_matching()

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        return None if market == "us" else live(market, symbol)

    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live)
    assert ev.path_check["compared_by_market"] == {"jp": 1, "us": 0}
    assert ev.path_check["passed"] is True


def test_skipped_symbols_do_not_use_up_the_quota(monkeypatch):
    # 先頭の jp が飛んでも、割り当ては比べられた銘柄で数えるので次の jp へ進む
    _patch(monkeypatch, extra={("jp", "2222"): _features(0)})
    monkeypatch.setattr(holdout, "PATH_CHECK_SAMPLE", 2)
    live = _live_matching()
    asked: list = []

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        asked.append((market, symbol))
        if symbol == "1111":
            return None
        return live("jp", "1111") if market == "jp" else live(market, symbol)

    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live)
    assert asked == [("jp", "1111"), ("jp", "2222"), ("us", "AAA")]
    assert ev.path_check["compared_by_market"] == {"jp": 1, "us": 1}
    assert ev.path_check["skipped_by_reason"] == _path_skips(live_none=1)
    assert ev.path_check["passed"] is True


def test_path_skip_reasons_are_counted(monkeypatch):
    # 1111 は最後の行の日の終値が無い、2222 は特徴量が古い、3333 は本番の予想が無い、
    # 4444 はこの道の予想が無い
    extra = {("jp", s): _features(0) for s in ("2222", "3333", "4444")}
    _patch(monkeypatch, extra=extra)
    monkeypatch.setattr(
        holdout,
        "load_raw_ohlcv",
        lambda m, s, start_date=None: _ohlcv(DAYS[-1] if s == "1111" else None),
    )
    live = _live_matching()

    def _live(market, symbol, on_stale=None, **_kwargs) -> Optional[PredictionResult]:
        if symbol == "2222":
            on_stale(market, symbol)
            return None
        if symbol == "3333":
            return None
        return live("jp", "1111") if market == "jp" else live(market, symbol)

    real = holdout._production_outcome

    def _outcome(latest_X, price, market, symbol, models, weights_fn):
        if symbol == "4444" and weights_fn is holdout.load_model_weights:
            return None, True
        return real(latest_X, price, market, symbol, models, weights_fn)

    monkeypatch.setattr(holdout, "_production_outcome", _outcome)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live)
    assert ev.path_check["skipped_by_reason"] == _path_skips(
        no_base_price=1, stale_features=1, live_none=1, production_none=1
    )
    assert ev.path_check["compared_by_market"] == {"jp": 0, "us": 1}
    assert ev.path_check["passed"] is False


def _live_by_market():
    """銘柄が増えても、jp は 1111、us は AAA と同じ値を返す本番の予想。"""
    live = _live_matching()

    def _live(market: str, _symbol: str, **_kwargs) -> Optional[PredictionResult]:
        return live("jp", "1111") if market == "jp" else live("us", "AAA")

    return _live


def _patch_prices_missing(monkeypatch, market: str, missing: list) -> None:
    def _load(m, _s, start_date=None):
        df = _ohlcv()
        return df[~df.index.isin(missing)] if m == market else df

    monkeypatch.setattr(holdout, "load_raw_ohlcv", _load)


def _delisted() -> pd.DataFrame:
    return _features(0).assign(date=pd.bdate_range("2025-01-01", periods=len(DAYS)))


def _no_features() -> pd.DataFrame:
    return pd.DataFrame()


def test_jp_rows_short_blocks_win_even_if_total_is_over_the_line(monkeypatch):
    # jp 2 銘柄は 10 日ずつ欠ける（20 / 40 行）。us 4 銘柄はそろう（80 行）。
    # 全市場は 100 / 120 行で線（96）を越えるが、jp は線（32）に届かない
    extra = {("jp", "2222"): _features(0)}
    extra.update({("us", s): _features(1) for s in ("BBB", "CCC", "DDD")})
    _patch(monkeypatch, extra=extra)
    _patch_prices_missing(monkeypatch, "jp", _dates()[:10])
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.candidate["n_baseline"] == 100
    assert ev.expected_rows == 120
    assert ev.path_check["passed"] is True
    assert ev.decision.won is False, "全市場の線を越えても jp の線で止める"
    assert ev.markets["jp"]["expected_rows"] == 40
    assert ev.markets["jp"]["n_compared"] == 20
    assert "発注する市場 jp の比べられた行が足りない（20 行、見込み 40 行、必要 32 以上）" in (
        ev.decision.reasons
    )
    assert ev.required_market_shortfalls == [
        "発注する市場 jp の比べられた行が足りない（20 行、見込み 40 行、必要 32 以上）"
    ]


@pytest.mark.parametrize("gone", [_delisted, _no_features], ids=["行が無い", "特徴量が無い"])
def test_jp_symbols_without_rows_block_win_though_expected_rows_shrink(monkeypatch, gone):
    # jp の 5 銘柄のうち 4 銘柄が飛ばされ、見込みも一緒に縮むので行の線では止まらない
    _patch(monkeypatch, extra={("jp", s): gone() for s in ("2222", "3333", "4444", "5555")})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.decision.won is False, "飛ばした銘柄で分母が縮んでも止める"
    assert ev.markets["jp"]["expected_rows"] == ev.markets["jp"]["n_compared"] == 20
    assert ev.markets["jp"]["n_symbols"] == 1
    assert ev.markets["jp"]["n_universe_symbols"] == 5
    assert "発注する市場 jp の行のある銘柄が足りない（1 銘柄、全 5 銘柄、必要 4 以上）" in (
        ev.decision.reasons
    )
    assert ev.required_market_shortfalls == [
        "発注する市場 jp の行のある銘柄が足りない（1 銘柄、全 5 銘柄、必要 4 以上）"
    ]


def test_jp_at_the_line_still_wins(monkeypatch):
    # jp の行は 16 / 20、行のある銘柄は 4 / 5 で、どちらも線の上
    extra = {("jp", s): _spread(_features(0), k) for k, s in enumerate(("2222", "3333", "4444"), 1)}
    extra[("jp", "5555")] = _delisted()
    _patch(monkeypatch, extra=extra)
    _patch_prices_missing(monkeypatch, "jp", _dates()[:4])
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.decision.won is True, ev.decision.reasons
    assert ev.markets["jp"]["n_compared"] == 4 * 16
    assert ev.markets["jp"]["expected_rows"] == 4 * 20
    assert ev.required_market_shortfalls == []


def test_loss_with_jp_over_the_line_has_no_jp_reason(monkeypatch):
    _patch(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(-1.0), _models(-1.0), _live_matching())
    assert ev.decision.won is False
    assert not any("発注する市場" in r for r in ev.decision.reasons)
    assert ev.required_market_shortfalls == []


def test_all_jp_symbols_without_rows_is_written_and_blocks_win(monkeypatch):
    _patch(monkeypatch, extra={("jp", "1111"): _delisted()})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.decision.won is False
    assert ev.markets["jp"]["n_symbols"] == ev.markets["jp"]["n_compared"] == 0
    assert ev.markets["jp"]["n_universe_symbols"] == 1
    assert ev.markets["jp"]["candidate"]["n_baseline"] == 0
    assert "発注する市場 jp の行のある銘柄が足りない（0 銘柄、全 1 銘柄、必要 0.8 以上）" in (
        ev.decision.reasons
    )


def test_no_jp_in_universe_does_not_win(monkeypatch):
    # 道の一致の検査と同じく、発注する市場を確かめられない回は勝ちにしない
    _patch(monkeypatch)
    monkeypatch.setattr(holdout, "get_all_symbols", lambda: [("us", "AAA")])
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.decision.won is False
    assert "jp" not in ev.markets
    assert "発注する市場 jp の銘柄が無い" in ev.decision.reasons
    assert "道の一致の検査で jp の比べられた銘柄が無い" in ev.decision.reasons


def _patch_watchlist(monkeypatch, symbols: set):
    monkeypatch.setattr(holdout, "load_watchlist_symbols", lambda market: set(symbols))


def _spread(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """最後の行より前の feat_a と y を 1e-3 * k ずらす（jp の銘柄の間に予想のばらつきを作る）。

    最後の行は道の一致の検査に使うので変えない。
    """
    out = df.copy()
    head = out.index[:-1]
    out.loc[head, "feat_a"] += 1e-3 * k
    out.loc[head, "y"] += 1e-3 * k
    return out


def _jp_frames(n: int) -> dict:
    """1111 に続けて、行のそろった jp の銘柄を足して n 銘柄にする（_patch の extra に渡す）。"""
    return {("jp", str(2000 + i)): _spread(_features(0), i + 1) for i in range(n - 1)}


def _jp_keys(frames: dict) -> set:
    return {s for m, s in frames if m == "jp"}


def test_jp_symbols_gone_from_table_block_win_by_watchlist_denominator(monkeypatch):
    # 段差で jp の行が消え、表の jp は 130 銘柄の全部に行がある。今の分母は 130 で 130 / 130 になる
    frames = _patch(monkeypatch, extra=_jp_frames(130))
    gone = {str(5000 + i) for i in range(120)}
    _patch_watchlist(monkeypatch, _jp_keys(frames) | gone)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    reason = "発注する市場 jp の行のある銘柄が足りない（130 銘柄、全 250 銘柄、必要 200 以上）"
    assert ev.decision.won is False, "表から消えた銘柄の分だけ分母が縮んでも止める"
    assert reason in ev.decision.reasons
    assert ev.required_market_shortfalls == [reason]
    assert ev.markets["jp"]["n_universe_symbols"] == ev.markets["jp"]["n_symbols"] == 130


@pytest.mark.parametrize("n_rows, won", [(200, True), (199, False)], ids=["線の上", "線の下"])
def test_jp_watchlist_line_is_min_coverage_of_watchlist(monkeypatch, n_rows, won):
    frames = _patch(monkeypatch, extra=_jp_frames(n_rows))
    gone = {str(5000 + i) for i in range(250 - n_rows)}
    _patch_watchlist(monkeypatch, _jp_keys(frames) | gone)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.decision.won is won, ev.decision.reasons
    assert bool(ev.required_market_shortfalls) is not won


def test_jp_over_watchlist_line_keeps_candidate_loss_without_jp_reason(monkeypatch):
    frames = _patch(monkeypatch, extra=_jp_frames(5))
    _patch_watchlist(monkeypatch, _jp_keys(frames))
    ev = evaluate_holdout(_dates(), _models(-1.0), _models(-1.0), _live_by_market())
    assert ev.decision.won is False
    assert ev.required_market_shortfalls == []


def test_delisted_jp_symbol_only_in_table_is_not_counted(monkeypatch):
    # ウォッチリストの 5 銘柄のうち行があるのは 1 銘柄。表にだけある廃止の 3 銘柄を足すと 4 / 5 に
    # 見えるが、分子に数えないので 1 / 5 で止める
    extra = {("jp", s): _features(0) for s in ("8001", "8002", "8003")}
    _patch(monkeypatch, extra=extra)
    _patch_watchlist(monkeypatch, {"1111", "2222", "3333", "4444", "5555"})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.markets["jp"]["n_watchlist_symbols_with_rows"] == 1
    assert ev.markets["jp"]["n_symbols"] == 4, "行の見込みの銘柄の数は変えない"
    assert ev.required_market_shortfalls == [
        "発注する市場 jp の行のある銘柄が足りない（1 銘柄、全 5 銘柄、必要 4 以上）"
    ]


def test_watchlist_symbol_without_rows_in_table_is_not_counted(monkeypatch):
    # 表に銘柄はあるが取っておいた日の行が無い銘柄は、行のある銘柄に数えない
    _patch(monkeypatch, extra={("jp", s): _delisted() for s in ("2222", "3333", "4444")})
    _patch_watchlist(monkeypatch, {"1111", "2222", "3333", "4444"})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.markets["jp"]["n_watchlist_symbols_with_rows"] == 1
    assert ev.decision.won is False


@pytest.mark.parametrize("empty", [None, set()], ids=["読めない", "空"])
def test_unreadable_watchlist_falls_back_to_universe_with_warning(monkeypatch, caplog, empty):
    # 表の jp 5 銘柄のうち行があるのは 4 銘柄。今の universe の数で判定して勝ちのまま
    extra = {("jp", s): _spread(_features(0), k) for k, s in enumerate(("2222", "3333", "4444"), 1)}
    extra[("jp", "5555")] = _delisted()
    _patch(monkeypatch, extra=extra)
    monkeypatch.setattr(holdout, "load_watchlist_symbols", lambda market: empty)
    with caplog.at_level("WARNING", logger=holdout.__name__):
        ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.decision.won is True, ev.decision.reasons
    assert ev.markets["jp"]["n_watchlist_symbols"] is None
    assert ev.markets["jp"]["n_watchlist_symbols_with_rows"] is None
    assert "universe の銘柄の数で判定する" in caplog.text


def test_unreadable_watchlist_still_blocks_short_universe(monkeypatch):
    _patch(monkeypatch, extra={("jp", s): _delisted() for s in ("2222", "3333", "4444", "5555")})
    monkeypatch.setattr(holdout, "load_watchlist_symbols", lambda market: None)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.required_market_shortfalls == [
        "発注する市場 jp の行のある銘柄が足りない（1 銘柄、全 5 銘柄、必要 4 以上）"
    ]


def test_watchlist_denominator_is_written_to_result_json(monkeypatch):
    frames = _patch(monkeypatch)
    _patch_watchlist(monkeypatch, _jp_keys(frames) | {"2222"})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    jp = asdict(ev)["markets"]["jp"]
    assert jp["n_watchlist_symbols"] == 2
    assert jp["n_watchlist_symbols_with_rows"] == 1
    assert "n_watchlist_symbols" not in asdict(ev)["markets"]["us"]


def test_load_watchlist_symbols_reads_market_and_normalizes(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text('{"us": ["aaa"], "jp": ["1111", "1111", "130a"]}', encoding="utf-8")
    monkeypatch.setattr(required_market, "get_watchlist_path", lambda: str(path))
    assert required_market.load_watchlist_symbols("jp") == {"1111", "130A"}


@pytest.mark.parametrize(
    "content",
    ["{壊れた", "[]", '{"jp": 1}'],
    ids=["JSON でない", "辞書でない", "市場が並びでない"],
)
def test_load_watchlist_symbols_returns_none_when_unreadable(
    monkeypatch, tmp_path, caplog, content
):
    path = tmp_path / "watchlist.json"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(required_market, "get_watchlist_path", lambda: str(path))
    with caplog.at_level("WARNING", logger=required_market.__name__):
        assert required_market.load_watchlist_symbols("jp") is None
    assert "ウォッチリスト" in caplog.text


def test_load_watchlist_symbols_returns_none_when_file_is_missing(monkeypatch, tmp_path):
    path = tmp_path / "absent.json"
    monkeypatch.setattr(required_market, "get_watchlist_path", lambda: str(path))
    assert required_market.load_watchlist_symbols("jp") is None


def test_markets_carry_each_markets_rows_and_metrics(monkeypatch):
    _patch(monkeypatch)
    _patch_prices_missing(monkeypatch, "jp", _dates()[:2])
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    jp = asdict(ev)["markets"]["jp"]
    assert jp["n_compared"] == 18
    assert jp["candidate"]["n_baseline"] == jp["production"]["n_baseline"] == 18
    assert jp["candidate"]["rank_corr"] > jp["production"]["rank_corr"]
    assert ev.markets["us"]["n_compared"] == 20
    assert ev.candidate["n_baseline"] == 38


# _dates() の 20 日のうち 10/12 と 11/3 が東証の休場で、営業日は 18 日
JP_HOLIDAYS = [pd.Timestamp("2026-10-12"), pd.Timestamp("2026-11-03")]
JP_OPEN_DAYS = [d for d in _dates() if d not in JP_HOLIDAYS]


def _jp_without(missing: list) -> pd.DataFrame:
    jp = _features(0)
    return jp[~jp["date"].isin(missing)].reset_index(drop=True)


def _patch_jp_tail_missing(monkeypatch):
    """jp の全銘柄で後ろ 15 日の行を消す。本番の予想は jp の残った最後の行に合わせて返す。"""
    jp = _jp_without(list(DAYS[-16:]))
    _patch(monkeypatch, extra={("jp", "1111"): jp, ("jp", "2222"): _spread(jp, 1)})
    us_live = _live_matching()

    def _live(market: str, symbol: str, **_kwargs) -> Optional[PredictionResult]:
        if market == "us":
            return us_live(market, symbol)
        diff = -float(jp["feat_a"].iloc[-1])
        return PredictionResult(
            market=market,
            symbol=symbol,
            current_price=100.0,
            avg_pred_price=100.0 * (1 + diff),
            diff_ratio=diff,
            model_count=2,
        )

    return _live


def test_jp_tail_days_missing_block_win_though_expected_rows_shrink(monkeypatch):
    # jp の見込みも 5 日 × 2 銘柄へ縮むので、行の線でも銘柄の線でも止まらない。
    # 残る 5 日のうち 10/12 は休場なので、行のある営業日は 4 日
    live = _patch_jp_tail_missing(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), live)
    assert ev.path_check["passed"] is True
    assert ev.markets["jp"]["n_dates"] == 5
    assert ev.markets["jp"]["expected_rows"] == ev.markets["jp"]["n_compared"] == 10
    assert ev.decision.won is False, "jp の後ろの日を比べていない回は勝ちにしない"
    assert ev.decision.reasons == [
        "発注する市場 jp の行のある営業日が足りない（4 日、東証の営業日 18 日、必要 14.4 以上）"
    ]
    assert ev.required_market_shortfalls == ev.decision.reasons


def test_jp_missing_only_holidays_still_wins(monkeypatch):
    # 本番の前の JSON の形（jp 18 日・us 20 日）。休場だけの差では勝ち負けが変わらない
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(JP_HOLIDAYS)})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.markets["jp"]["n_dates"] == 18
    assert ev.markets["us"]["n_dates"] == 20
    assert ev.decision.won is True, ev.decision.reasons


def test_jp_missing_only_holidays_still_loses_when_candidate_is_worse(monkeypatch):
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(JP_HOLIDAYS)})
    ev = evaluate_holdout(_dates(), _models(-1.0), _models(-1.0), _live_matching())
    assert ev.decision.won is False
    assert not any("発注する市場" in r for r in ev.decision.reasons)


@pytest.mark.parametrize("n_kept, won", [(15, True), (14, False)], ids=["線の上", "線の下"])
def test_jp_day_line_is_min_coverage_of_tse_open_days(monkeypatch, n_kept, won):
    # 営業日 18 日の 80% は 14.4 日。古い方の営業日と休場の日の行を消す
    missing = JP_HOLIDAYS + JP_OPEN_DAYS[: len(JP_OPEN_DAYS) - n_kept]
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(missing)})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.markets["jp"]["n_dates"] == n_kept
    assert ev.decision.won is won, ev.decision.reasons


def test_jp_rows_on_holidays_do_not_count_as_open_days(monkeypatch):
    # 10/12 と 11/3 に行があっても営業日には数えず、行のある営業日は 14 日で線の下
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(JP_OPEN_DAYS[:4])})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.markets["jp"]["n_dates"] == 16
    assert ev.decision.won is False


def test_markets_carry_tse_open_days_for_jp_only(monkeypatch):
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(JP_HOLIDAYS)})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    markets = asdict(ev)["markets"]
    assert markets["jp"]["n_business_dates"] == 18
    assert markets["jp"]["n_business_dates_with_rows"] == 18
    assert "n_business_dates" not in markets["us"]


def test_year_missing_from_holiday_table_counts_weekdays_and_does_not_pass(monkeypatch):
    # 表が切れた年は平日を全部営業日に数える（分母が大きくなる側なので、候補を黙って通さない）
    live = _patch_jp_tail_missing(monkeypatch)
    monkeypatch.setattr(jp_market_holidays, "_HOLIDAYS_BY_YEAR", {})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), live)
    assert ev.markets["jp"]["n_business_dates"] == 20
    assert ev.decision.reasons == [
        "発注する市場 jp の行のある営業日が足りない（5 日、東証の営業日 20 日、必要 16 以上）"
    ]


def _capture_daily_diff(monkeypatch) -> list:
    """judge_win に渡った daily_diff を集める（票の分母を見る）。"""
    seen: list = []
    real = holdout.judge_win

    def _judge(candidate, production, daily_diff, expected_rows):
        seen.append(list(daily_diff))
        return real(candidate, production, daily_diff, expected_rows)

    monkeypatch.setattr(holdout, "judge_win", _judge)
    return seen


def test_vote_counts_only_jp_open_days_when_holdout_has_jp_holidays(monkeypatch):
    # 20 日のうち 10/12 と 11/3 は東証の休場で jp の行が無い（us だけの日）。票は 18 日で数える
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(JP_HOLIDAYS)})
    seen = _capture_daily_diff(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert len(seen) == 1
    assert len(seen[0]) == 18
    result = asdict(ev)
    assert result["vote_dates"] == [d.strftime("%Y-%m-%d") for d in JP_OPEN_DAYS]
    assert [s["date"] for s in result["vote_skipped"]] == ["2026-10-12", "2026-11-03"]
    assert {s["reason"] for s in result["vote_skipped"]} == {"jp_holiday"}
    assert all(s["n_jp_symbols"] == 0 for s in result["vote_skipped"])
    assert len(ev.daily) == 20, "日ごとの物差しは取っておいた20日の全部を書く"
    assert ev.decision.won is True, ev.decision.reasons


def _patch_jp_prices_missing_for(monkeypatch, symbols: set, missing: list) -> None:
    def _load(m, s, start_date=None):
        df = _ohlcv()
        return df[~df.index.isin(missing)] if m == "jp" and s in symbols else df

    monkeypatch.setattr(holdout, "load_raw_ohlcv", _load)


def test_vote_skips_open_days_where_jp_symbols_are_under_the_line(monkeypatch):
    # jp 5 銘柄のうち 4 銘柄は 2 日だけ値段が無く、その日の jp は 1 銘柄（線は 4 銘柄）
    frames = _patch(monkeypatch, extra=_jp_frames(5))
    _patch_watchlist(monkeypatch, _jp_keys(frames))
    short_days = JP_OPEN_DAYS[5:7]
    _patch_jp_prices_missing_for(monkeypatch, _jp_keys(frames) - {"1111"}, short_days)
    seen = _capture_daily_diff(monkeypatch)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert len(seen[0]) == 18
    assert ev.vote_dates == [d.strftime("%Y-%m-%d") for d in _dates() if d not in short_days]
    assert ev.vote_skipped == [
        {"date": d.strftime("%Y-%m-%d"), "reason": "jp_symbols_short", "n_jp_symbols": 1}
        for d in short_days
    ]
    assert ev.vote_need_symbols == 4


def test_vote_jp_symbol_count_uses_watchlist_symbols_only(monkeypatch):
    # 表にだけある廃止の jp 3 銘柄は、日ごとの jp の銘柄に数えない。分母はウォッチリストの 2 銘柄
    extra = {("jp", s): _features(0) for s in ("8001", "8002", "8003")}
    extra[("jp", "2222")] = _features(0)
    _patch(monkeypatch, extra=extra)
    _patch_watchlist(monkeypatch, {"1111", "2222"})
    short_days = _dates()[:1]
    _patch_jp_prices_missing_for(monkeypatch, {"2222"}, short_days)
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.vote_need_symbols == pytest.approx(1.6)
    assert [s["date"] for s in ev.vote_skipped] == ["2026-10-07"]
    assert ev.vote_skipped[0]["n_jp_symbols"] == 1


def test_vote_change_keeps_overall_metrics(monkeypatch):
    # 票から外した日の行も、全体の物差しには残す
    _patch(monkeypatch, extra={("jp", "1111"): _jp_without(JP_HOLIDAYS)})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.candidate["n_baseline"] == ev.production["n_baseline"] == 38
    assert ev.expected_rows == 38


def test_jp_constant_candidate_cannot_win_though_mixed_std_is_over_the_line(monkeypatch):
    # jp の 5 銘柄はどの日も同じ値、us は jp と違う値。全市場を混ぜた日ごとの標準偏差は
    # 線を越えるが、jp の中ではばらつきが無いので勝ちにしない（#1718）
    extra = {("jp", str(2000 + i)): _features(0) for i in range(4)}
    frames = _patch(monkeypatch, extra=extra)
    _patch_watchlist(monkeypatch, _jp_keys(frames))
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_by_market())
    assert ev.candidate["prediction_std"] > 1e-4, "全市場の確かめは通る"
    assert ev.markets["jp"]["candidate"]["prediction_std"] == 0.0
    assert ev.decision.won is False
    jp_reasons = [r for r in ev.decision.reasons if "発注する市場 jp の中で" in r]
    assert len(jp_reasons) == 1, ev.decision.reasons
    assert "ほぼ同じ値" in jp_reasons[0]
    assert not any(r.startswith("候補の予想が全銘柄で") for r in ev.decision.reasons)


@pytest.mark.parametrize(
    "markets",
    [{}, {"jp": {"candidate": {"prediction_std": None}}}],
    ids=["jp が無い", "jp の標準偏差が出ない"],
)
def test_jp_constant_reason_is_none_without_jp_std(markets):
    # jp が無い回は required_market_reasons が止め、標準偏差が出ない回は測れないので理由にしない
    assert required_market.required_market_constant_reason(markets) is None
