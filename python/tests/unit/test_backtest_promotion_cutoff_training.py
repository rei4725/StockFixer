"""統合モデルの学習に締め日を足す（バックテスト暫定昇格 Task 1）のテスト"""

import numpy as np
import pandas as pd
import pytest

from src.prediction import unified_model_pipeline as ump
from src.prediction.manager import ModelManager


def test_drop_rows_reaching_cutoff_keeps_only_rows_whose_next_day_is_before_cutoff():
    df = pd.DataFrame(
        {
            "market": ["us"] * 4 + ["jp"] * 2,
            "symbol": ["A"] * 4 + ["B"] * 2,
            "date": pd.to_datetime(
                ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-06", "2026-10-01", "2026-10-03"]
            ),
        }
    )
    out = ump.drop_rows_reaching_cutoff(df, pd.Timestamp("2026-10-03"))
    # us/A 10-02 は次が 10-03、jp/B 10-01 も次が 10-03 なので落ちる。最後の行も落ちる
    assert set(map(tuple, out[["market", "symbol", "date"]].astype(str).values)) == {
        ("us", "A", "2026-10-01"),
    }


def test_drop_rows_reaching_cutoff_keeps_original_index_and_handles_unsorted_input():
    df = pd.DataFrame(
        {
            "market": ["us"] * 3,
            "symbol": ["A"] * 3,
            "date": pd.to_datetime(["2026-10-03", "2026-10-01", "2026-10-02"]),
        },
        index=[7, 8, 9],
    )
    out = ump.drop_rows_reaching_cutoff(df, pd.Timestamp("2026-10-03"))
    assert list(out.index) == [8]


def _make_frame(cutoff: pd.Timestamp) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-08-03", periods=60)
    frames = []
    for market, symbol in [("us", "A"), ("jp", "B")]:
        f = pd.DataFrame(
            {
                "market": market,
                "symbol": symbol,
                "date": dates,
                "f1": rng.normal(size=60),
                "f2": rng.normal(size=60),
                "f3": rng.normal(size=60),
            }
        )
        f["y"] = f["f1"] * 0.01
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    after = df["date"] >= cutoff
    # 締め日以降の行は y と特徴量に目印を入れる
    df.loc[after, "y"] = 1e6
    df.loc[after, "f1"] = 1e6
    return df


def _setup(monkeypatch, df, calls):
    monkeypatch.setattr(ump, "load_all_stock_data", lambda data_dir=None: df.copy())
    monkeypatch.setattr(ump, "load_excluded_features", lambda *a, **k: set())
    monkeypatch.setattr(
        ump, "save_feature_selection", lambda *a, **k: calls["save_selection"].append(a)
    )
    monkeypatch.setattr(ump, "any_sentiment_source_configured", lambda markets: True)

    orig_split = ump._split_train_val_by_date

    def spy_split(X, y, day):
        calls["split_max"].append(day.max())
        return orig_split(X, y, day)

    monkeypatch.setattr(ump, "_split_train_val_by_date", spy_split)

    orig_create = ModelManager.create_model

    def small_create(self, model_type, model_name, **kwargs):
        model = orig_create(self, model_type, model_name, n_estimators=5, **kwargs)
        calls["models"].append(model)
        return model

    monkeypatch.setattr(ModelManager, "create_model", small_create)


def test_cutoff_training_ignores_marked_targets_after_cutoff(monkeypatch, tmp_path):
    cutoff = pd.Timestamp("2026-10-01")
    df = _make_frame(cutoff)
    calls = {"save_selection": [], "split_max": [], "models": []}
    _setup(monkeypatch, df, calls)

    ump.train_unified_model(
        model_name="with_cutoff",
        save_dir=str(tmp_path / "a"),
        train_end_date=cutoff,
        apply_saved_exclusions=False,
        save_selection=False,
    )

    # 締め日以降の行と、y が締め日に届く行（締め日の直前の行）を最初から除いたデータ
    pre_trimmed = ump.drop_rows_reaching_cutoff(df, cutoff)
    assert (pre_trimmed["date"] < cutoff).all()
    monkeypatch.setattr(ump, "load_all_stock_data", lambda data_dir=None: pre_trimmed.copy())
    ump.train_unified_model(
        model_name="pre_trimmed",
        save_dir=str(tmp_path / "b"),
        apply_saved_exclusions=False,
        save_selection=False,
    )

    X_probe = pre_trimmed[["f1", "f2", "f3"]].assign(market_encoded=0).head(10)
    m_cutoff, m_pre = calls["models"]
    pred_with_cutoff = np.asarray(m_cutoff.predict(X_probe))
    pred_pre_trimmed = np.asarray(m_pre.predict(X_probe))

    assert np.allclose(pred_with_cutoff, pred_pre_trimmed)
    assert max(calls["split_max"]) < cutoff
    assert calls["save_selection"] == []


def test_save_selection_default_still_saves(monkeypatch, tmp_path):
    cutoff = pd.Timestamp("2026-10-01")
    calls = {"save_selection": [], "split_max": [], "models": []}
    _setup(monkeypatch, _make_frame(cutoff), calls)
    monkeypatch.setattr(ump, "FEATURE_SELECTION_MIN_FEATURES", 1)
    ump.train_unified_model(model_name="d", save_dir=str(tmp_path), train_end_date=cutoff)
    assert len(calls["save_selection"]) == 1


def test_apply_saved_exclusions_false_skips_both_calls(monkeypatch, tmp_path):
    cutoff = pd.Timestamp("2026-10-01")
    calls = {"save_selection": [], "split_max": [], "models": []}
    _setup(monkeypatch, _make_frame(cutoff), calls)
    loaded = []
    monkeypatch.setattr(ump, "load_excluded_features", lambda *a, **k: loaded.append(a) or {"f2"})
    ump.train_unified_model(
        model_name="x", save_dir=str(tmp_path), apply_saved_exclusions=False, save_selection=False
    )
    assert loaded == []


def test_cutoff_with_multi_horizon_raises():
    with pytest.raises(ValueError):
        ump.train_unified_model(horizon=3, train_end_date=pd.Timestamp("2026-10-01"))
