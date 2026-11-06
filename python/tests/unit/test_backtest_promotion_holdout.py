"""
取っておいた期間の評価（evaluate_holdout）と道の一致の検査のテスト

DB の読み込みはすべて差し替える。候補は特徴量 feat_a をそのまま言い（y と同じ）、
production はその逆を言う。
"""

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.prediction import predict_unified
from src.prediction.backtest_promotion import holdout
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
    # 20 日 × 2 銘柄 = 40 の 80%（32）だと jp の休場だけで負ける。市場ごとに数えれば勝てる
    holiday = set(_dates()[3:12])
    jp = _features(0)
    jp = jp[~jp["date"].isin(holiday)].reset_index(drop=True)
    _patch(monkeypatch, extra={("jp", "1111"): jp})
    ev = evaluate_holdout(_dates(), _models(1.0), _models(-1.0), _live_matching())
    assert ev.candidate["n_baseline"] == 31
    assert ev.markets == {
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
