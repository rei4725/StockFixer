"""results/accuracy の日ごとの JSON を読む表と移動平均のテスト（#1815）。"""

import json

import pandas as pd
import pytest

from src.reporting.accuracy_trend import (
    TREND_METRICS,
    add_moving_average,
    build_accuracy_trend,
    format_trend_markdown,
    load_accuracy_reports,
    load_accuracy_trend,
)


def _section(acc, base, diff, corr, up, by_market=None):
    return {
        "n_scored": 10,
        "direction_accuracy": acc,
        "mean_abs_error": 0.01,
        "majority_baseline_accuracy": base,
        "hit_rate_minus_majority": diff,
        "rank_corr": corr,
        "predicted_up_rate": up,
        "four_week": {"n_scored": 100},
        "top_misses": [],
        "by_market": by_market or {},
        "by_sector": {},
    }


def _write(dir_path, day, models):
    report = {"date": day, "generated_at": f"{day}T09:00:00+00:00", "window_days": 28}
    report["models"] = models
    (dir_path / f"{day}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def accuracy_dir(tmp_path):
    jp = _section(0.6, 0.5, 0.1, 0.2, 0.7)
    _write(
        tmp_path,
        "2026-11-02",
        {
            "production": {"1": _section(0.5, 0.5, 0.0, 0.1, 0.8, {"jp": jp})},
            "challenger": {"1": _section(0.4, 0.5, -0.1, 0.0, 0.9)},
        },
    )
    # 11-03 は欠けた日（ファイルが無い）
    _write(
        tmp_path,
        "2026-11-04",
        {"production": {"1": _section(0.7, 0.6, 0.1, 0.3, 0.6)}},
    )
    _write(
        tmp_path,
        "2026-11-05",
        {"production": {"1": _section(0.6, 0.5, 0.1, None, 0.5)}},
    )
    return tmp_path


def test_rows_per_day_model_market(accuracy_dir):
    df = build_accuracy_trend(load_accuracy_reports(str(accuracy_dir)))

    assert list(df.columns[:5]) == ["date", "model", "horizon", "market", "n_scored"]
    assert len(df) == 5
    first = df[(df["date"] == "2026-11-02") & (df["model"] == "production")]
    assert list(first["market"]) == ["ALL", "jp"]
    jp = first[first["market"] == "jp"].iloc[0]
    assert jp["direction_accuracy"] == pytest.approx(0.6)
    assert jp["majority_baseline_accuracy"] == pytest.approx(0.5)
    assert jp["hit_rate_minus_majority"] == pytest.approx(0.1)
    assert jp["rank_corr"] == pytest.approx(0.2)
    assert jp["predicted_up_rate"] == pytest.approx(0.7)
    chal = df[df["model"] == "challenger"].iloc[0]
    assert chal["direction_accuracy"] == pytest.approx(0.4)
    assert chal["horizon"] == "1"


def test_moving_average_per_group(accuracy_dir):
    df = load_accuracy_trend(str(accuracy_dir), window=2)
    prod = df[(df["model"] == "production") & (df["market"] == "ALL")].reset_index(drop=True)

    assert list(prod["date"]) == ["2026-11-02", "2026-11-04", "2026-11-05"]
    assert list(prod["direction_accuracy_ma2"]) == pytest.approx([0.5, 0.6, 0.65])
    # None の日は移動平均から外す（窓の中の値のある日だけで平均する）
    assert list(prod["rank_corr_ma2"]) == pytest.approx([0.1, 0.2, 0.3])
    # 別のモデルの値は混ざらない
    chal = df[df["model"] == "challenger"].iloc[0]
    assert chal["direction_accuracy_ma2"] == pytest.approx(0.4)
    for metric in TREND_METRICS:
        assert f"{metric}_ma2" in df.columns


def test_old_json_without_keys_does_not_fail(tmp_path):
    old = {
        "n_scored": 0,
        "direction_accuracy": None,
        "mean_abs_error": None,
        "four_week": {"n_scored": 0},
        "top_misses": [],
        "by_market": {},
        "by_sector": {},
    }
    _write(tmp_path, "2026-10-22", {"production": {"1": old}})
    (tmp_path / "2026-10-23.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "2026-10-24.json").write_text(json.dumps({"date": "2026-10-24"}), encoding="utf-8")

    df = load_accuracy_trend(str(tmp_path), window=3)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["date"] == "2026-10-22"
    for metric in TREND_METRICS:
        assert pd.isna(row[metric])
        assert pd.isna(row[f"{metric}_ma3"])


def test_missing_dir_and_empty(tmp_path):
    df = load_accuracy_trend(str(tmp_path / "none"), window=5)
    assert df.empty
    assert "direction_accuracy_ma5" in df.columns
    assert "データなし" in format_trend_markdown(df)


def test_add_moving_average_rejects_bad_window():
    with pytest.raises(ValueError):
        add_moving_average(build_accuracy_trend([]), window=0)


def test_markdown_has_header_and_rows(accuracy_dir):
    df = load_accuracy_trend(str(accuracy_dir), window=2)
    text = format_trend_markdown(df)
    lines = text.splitlines()
    assert lines[0].startswith("| date | model | horizon | market |")
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert len(lines) == 2 + len(df)
    assert "| 2026-11-05 | production | 1 | ALL |" in text
    assert "0.6500" in text
