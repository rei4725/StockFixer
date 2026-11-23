"""統合の学習の特徴量の準備が、値を変えずに表の写しの重なりを減らすことのテスト（#2134）。"""

import tracemalloc
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.sentiment_sources import any_sentiment_source_configured, is_sentiment_feature
from src.market_data.technical import add_earnings_flag
from src.prediction import unified_model_pipeline as ump

_EXCLUDED = ["f003", "f010", "f011"]


def _old_prepare(df, excluded):
    """#2134 で直す前の prepare_unified_features（比べる先）。"""
    if "market_encoded" not in df.columns and "market" in df.columns:
        df["market_encoded"] = df["market"].map({"us": 0, "jp": 1}).fillna(-1).astype(int)
    y = df["y"]
    feature_cols = [
        c
        for c in df.columns
        if c not in ["market", "symbol", "y", "date"]
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    X = df[feature_cols]
    X = X.drop(
        columns=[c for c in X.columns if str(c).startswith(ump._CORPORATE_ACTION_LAG_PREFIXES)]
    )
    markets = df["market"].dropna().unique() if "market" in df.columns else []
    if not any_sentiment_source_configured(markets):
        X = X.drop(columns=[c for c in X.columns if is_sentiment_feature(c)])
    valid_idx = X.notna().all(axis=1) & y.notna()
    if "market" in df.columns and "symbol" in df.columns:
        ump._warn_dropped_rows(valid_idx, df["market"], df["symbol"])
    X = X[valid_idx]
    y = y[valid_idx]
    if excluded:
        remaining = [col for col in X.columns if col not in excluded]
        X = X[remaining].copy() if remaining else X
    return X, y


def _old_mask(df, port):
    """#2134 で直す前の決算の前後の行の除き方（比べる先）。"""
    frames = []
    for (market, symbol), group in df.groupby(["market", "symbol"], sort=False):
        work = group.copy()
        work.index = pd.DatetimeIndex(pd.to_datetime(work["date"], errors="coerce"))
        dates = port.get_earnings_dates(str(market), str(symbol), raise_on_error=True)
        work = port.add_earnings_flag(work, dates, lookaround_days=ump.EARNINGS_MASK_WINDOW_DAYS)
        work = work[work["earnings_flag"] == 0].copy()
        work["date"] = work.index
        frames.append(work.drop(columns=["earnings_flag"], errors="ignore"))
    return pd.concat(frames, ignore_index=True)


def _table(n_per_symbol=60, n_features=40, nan_rows=True, market=True):
    """float32 の特徴量に float64 と整数の列が混ざる、本番の表の形の模型。"""
    rng = np.random.default_rng(7)
    symbols = [("jp", "7203"), ("jp", "6758"), ("us", "AAPL"), ("us", "MSFT")]
    n = n_per_symbol * len(symbols)
    values = rng.normal(size=(n, n_features)).astype("float32")
    features = pd.DataFrame(values, columns=[f"f{i:03d}" for i in range(n_features)])
    if nan_rows:
        features.iloc[[0, 5, n - 1], 2] = np.nan
        features.iloc[9, 10] = np.nan  # 保存した除外の列の NaN でも行は落ちる
    features.insert(5, "Close", rng.normal(size=n))
    features.insert(8, "count", np.arange(n))
    features["Dividends_lag1"] = np.float32(np.nan)
    features["asof"] = pd.Timestamp("2026-01-01")
    features["date"] = list(pd.date_range("2024-01-01", periods=n_per_symbol, freq="B")) * 4
    if market:
        features.insert(0, "market", [m for m, _ in symbols for _ in range(n_per_symbol)])
        features.insert(1, "symbol", [s for _, s in symbols for _ in range(n_per_symbol)])
    features["y"] = rng.normal(size=n)
    if nan_rows:
        features.loc[20, "y"] = np.nan
    return features.copy()


def _set_excluded(monkeypatch, excluded):
    monkeypatch.setattr(ump, "load_excluded_features", lambda *a, **k: list(excluded))


@pytest.mark.parametrize("excluded", [[], _EXCLUDED, "all"])
@pytest.mark.parametrize("market", [True, False])
@pytest.mark.parametrize("nan_rows", [True, False])
@pytest.mark.parametrize("index", ["range", "gap", "duplicated"])
def test_same_x_y_as_before(monkeypatch, caplog, excluded, market, nan_rows, index):
    df = _table(market=market, nan_rows=nan_rows)
    if index == "gap":  # drop_rows_reaching_cutoff の後の形
        df = df[np.arange(len(df)) % 7 != 3]
    elif index == "duplicated":
        df.index = [i // 2 for i in range(len(df))]
    if excluded == "all":
        excluded = [*df.columns, "market_encoded"]
    _set_excluded(monkeypatch, excluded)
    old_df, new_df = df.copy(), df.copy()

    old_X, old_y = _old_prepare(old_df, excluded)
    old_messages = [r.getMessage() for r in caplog.records]
    caplog.clear()
    new_X, new_y = ump.prepare_unified_features(new_df)

    pd.testing.assert_frame_equal(new_X, old_X, check_exact=True, check_index_type="exact")
    pd.testing.assert_series_equal(new_y, old_y, check_exact=True, check_index_type="exact")
    pd.testing.assert_frame_equal(new_df, old_df, check_exact=True)
    assert [r.getMessage() for r in caplog.records] == old_messages


def test_saved_exclusions_not_applied_when_off(monkeypatch):
    _set_excluded(monkeypatch, _EXCLUDED)
    old_X, _ = _old_prepare(_table(), [])
    new_X, _ = ump.prepare_unified_features(_table(), apply_saved_exclusions=False)
    pd.testing.assert_frame_equal(new_X, old_X, check_exact=True)


def test_returned_x_does_not_share_memory_with_input(monkeypatch):
    _set_excluded(monkeypatch, [])
    df = _table(nan_rows=False)
    before = df["f000"].copy()
    X, _ = ump.prepare_unified_features(df)
    X.iloc[0, 0] = 999.0
    pd.testing.assert_series_equal(df["f000"], before)


def test_missing_y_message_is_unchanged():
    with pytest.raises(
        ValueError,
        match="^y列がデータに含まれていません。特徴量生成済みのCSVを使用してください。$",
    ):
        ump.prepare_unified_features(_table().drop(columns="y"))


def _peak(func, *args):
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        func(*args)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def test_prepare_peak_is_lower_than_before(monkeypatch):
    excluded = [f"f{i:03d}" for i in range(0, 300, 5)]
    _set_excluded(monkeypatch, excluded)
    table = _table(n_per_symbol=3000, n_features=300)
    table["market_encoded"] = 1
    old = _peak(_old_prepare, table, excluded)
    new = _peak(ump.prepare_unified_features, table)
    size = int(table.memory_usage().sum())
    assert new < old * 0.6, f"表 {size} 直す前 {old} 直した後 {new}"
    print(f"prepare: 表 {size} 直す前 {old} 直した後 {new}")


def _port():
    port = MagicMock()
    port.get_earnings_dates.side_effect = lambda market, symbol, **k: pd.DatetimeIndex(
        ["2024-02-01", "2024-03-06"] if market == "us" else []
    )
    port.add_earnings_flag.side_effect = add_earnings_flag
    return port


def test_mask_returns_same_table_as_before(monkeypatch):
    port = _port()
    monkeypatch.setattr(ump, "get_market_data_port", lambda: port)
    # 銘柄の行が離れていても、今までどおり銘柄ごとにまとまる。日付の無い行も残る
    df = _table().sample(frac=1, random_state=1)
    df.iloc[3, df.columns.get_loc("date")] = pd.NaT
    before = df.copy()

    masked = ump._mask_earnings_rows_for_unified(df)

    pd.testing.assert_frame_equal(
        masked, _old_mask(df, port), check_exact=True, check_index_type="exact"
    )
    assert len(masked) < len(df)
    pd.testing.assert_frame_equal(df, before, check_exact=True)


def test_mask_peak_is_lower_than_before(monkeypatch):
    port = _port()
    monkeypatch.setattr(ump, "get_market_data_port", lambda: port)
    table = _table(n_per_symbol=3000, n_features=300)
    old = _peak(_old_mask, table, port)
    new = _peak(ump._mask_earnings_rows_for_unified, table)
    size = int(table.memory_usage().sum())
    assert new < old * 0.6, f"表 {size} 直す前 {old} 直した後 {new}"
    print(f"mask: 表 {size} 直す前 {old} 直した後 {new}")
