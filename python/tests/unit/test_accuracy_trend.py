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


_MAGNITUDE_METRICS = ("mean_abs_predicted_ratio", "mean_abs_actual_ratio", "prediction_std")
_LEGACY_COLUMNS = [
    "date",
    "model",
    "horizon",
    "market",
    "n_scored",
    "direction_accuracy",
    "majority_baseline_accuracy",
    "hit_rate_minus_majority",
    "rank_corr",
    "predicted_up_rate",
]


def _magnitude(section, pred, actual, std):
    section = dict(section)
    section["mean_abs_predicted_ratio"] = pred
    section["mean_abs_actual_ratio"] = actual
    section["prediction_std"] = std
    return section


def test_magnitude_metrics_after_legacy_columns(accuracy_dir):
    df = build_accuracy_trend(load_accuracy_reports(str(accuracy_dir)))

    # 既存の5つの物差しと n_scored の列の順は変えず、3つは後ろに足す
    assert list(df.columns) == _LEGACY_COLUMNS + list(_MAGNITUDE_METRICS)
    assert tuple(TREND_METRICS[-3:]) == _MAGNITUDE_METRICS


def test_magnitude_metrics_values_and_moving_average(tmp_path):
    # 11-05 は鍵の無い古い日、11-06 と 11-07 は鍵のある日
    _write(tmp_path, "2026-11-05", {"production": {"1": _section(0.6, 0.5, 0.1, 0.2, 0.5)}})
    jp = _magnitude(_section(0.6, 0.5, 0.1, 0.2, 0.7), 0.05, 0.02, 0.01)
    _write(
        tmp_path,
        "2026-11-06",
        {
            "production": {
                "1": _magnitude(
                    _section(0.5, 0.5, 0.0, 0.1, 0.8, {"jp": jp}), 0.0655, 0.0176, 0.0183
                )
            },
            "challenger": {
                "1": _magnitude(_section(0.4, 0.5, -0.1, 0.0, 0.9), 0.0067, 0.0176, 0.0020)
            },
        },
    )
    _write(
        tmp_path,
        "2026-11-07",
        {
            "production": {
                "1": _magnitude(_section(0.7, 0.6, 0.1, 0.3, 0.6), 0.0645, 0.0180, 0.0170)
            }
        },
    )

    df = load_accuracy_trend(str(tmp_path), window=2)
    prod = df[(df["model"] == "production") & (df["market"] == "ALL")].reset_index(drop=True)

    assert list(prod["date"]) == ["2026-11-05", "2026-11-06", "2026-11-07"]
    for metric in _MAGNITUDE_METRICS:
        assert pd.isna(prod.loc[0, metric])
        assert pd.isna(prod.loc[0, f"{metric}_ma2"])
    assert list(prod["mean_abs_predicted_ratio"][1:]) == pytest.approx([0.0655, 0.0645])
    assert list(prod["mean_abs_actual_ratio"][1:]) == pytest.approx([0.0176, 0.0180])
    assert list(prod["prediction_std"][1:]) == pytest.approx([0.0183, 0.0170])
    # 鍵の無い日は移動平均から外す
    assert list(prod["mean_abs_predicted_ratio_ma2"][1:]) == pytest.approx([0.0655, 0.065])
    assert list(prod["mean_abs_actual_ratio_ma2"][1:]) == pytest.approx([0.0176, 0.0178])
    assert list(prod["prediction_std_ma2"][1:]) == pytest.approx([0.0183, 0.01765])
    # 既存の物差しの値は変わらない
    assert list(prod["direction_accuracy"]) == pytest.approx([0.6, 0.5, 0.7])
    assert list(prod["direction_accuracy_ma2"]) == pytest.approx([0.6, 0.55, 0.6])

    jp_row = df[(df["model"] == "production") & (df["market"] == "jp")].iloc[0]
    assert jp_row["mean_abs_predicted_ratio"] == pytest.approx(0.05)
    assert jp_row["mean_abs_actual_ratio"] == pytest.approx(0.02)
    assert jp_row["prediction_std"] == pytest.approx(0.01)
    chal = df[df["model"] == "challenger"].iloc[0]
    assert chal["mean_abs_predicted_ratio_ma2"] == pytest.approx(0.0067)
    assert chal["prediction_std_ma2"] == pytest.approx(0.0020)


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


def _four_week(n, acc, diff, n_days, by_market=None):
    return {
        "n_scored": n,
        "direction_accuracy": acc,
        "majority_baseline_accuracy": 0.5,
        "hit_rate_minus_majority": diff,
        "rank_corr": 0.05,
        "predicted_up_rate": 0.6,
        "n_days": n_days,
        "daily_direction_accuracy_mean": acc,
        "by_market": by_market or {},
    }


def test_four_week_rows_per_day_model_market(tmp_path):
    """four_week と four_week.by_market から日・モデル・ホライズン・市場ごとの4週の行を作る（#2013）。"""
    from src.reporting.accuracy_trend import FOUR_WEEK_METRICS, build_four_week_trend

    jp = _four_week(60, 0.52, 0.02, None)
    us = _four_week(40, 0.48, -0.03, None)
    first = _section(0.5, 0.5, 0.0, 0.1, 0.8)
    first["four_week"] = _four_week(100, 0.5, -0.01, 18, {"jp": jp, "us": us})
    second = _section(0.6, 0.5, 0.1, 0.1, 0.8)
    second["four_week"] = _four_week(120, 0.51, 0.01, 19, {"jp": jp})
    _write(tmp_path, "2026-11-17", {"production": {"1": first}})
    _write(tmp_path, "2026-11-18", {"production": {"1": second}})

    df = build_four_week_trend(load_accuracy_reports(str(tmp_path)))

    assert "n_days" in FOUR_WEEK_METRICS
    assert "hit_rate_minus_majority" in FOUR_WEEK_METRICS
    assert list(df.columns) == ["date", "model", "horizon", "market", "n_scored"] + list(
        FOUR_WEEK_METRICS
    )
    keys = list(zip(df["date"], df["market"]))
    assert keys == [
        ("2026-11-17", "ALL"),
        ("2026-11-17", "jp"),
        ("2026-11-17", "us"),
        ("2026-11-18", "ALL"),
        ("2026-11-18", "jp"),
    ]
    all_rows = df[df["market"] == "ALL"]
    assert all_rows["hit_rate_minus_majority"].tolist() == pytest.approx([-0.01, 0.01])
    assert all_rows["n_days"].tolist() == [18.0, 19.0]
    assert all_rows["n_scored"].tolist() == [100.0, 120.0]
    us_row = df[df["market"] == "us"].iloc[0]
    assert us_row["hit_rate_minus_majority"] == pytest.approx(-0.03)
    assert pd.isna(us_row["n_days"])


def test_four_week_old_json_and_empty_window(tmp_path):
    """four_week の無い古い JSON と空の窓は値を空にして行を残し、壊れた JSON は飛ばす（#2013）。"""
    from src.reporting.accuracy_trend import FOUR_WEEK_METRICS, build_four_week_trend

    old = _section(0.5, 0.5, 0.0, 0.1, 0.8)
    del old["four_week"]
    empty = _section(0.5, 0.5, 0.0, 0.1, 0.8)
    empty["four_week"] = {
        "n_scored": 0,
        "direction_accuracy": None,
        "n_days": 0,
        "daily_direction_accuracy_mean": None,
        "by_market": {},
    }
    _write(tmp_path, "2026-10-01", {"production": {"1": old}})
    _write(tmp_path, "2026-10-02", {"production": {"1": empty}})
    (tmp_path / "2026-10-03.json").write_text("{broken", encoding="utf-8")

    df = build_four_week_trend(load_accuracy_reports(str(tmp_path)))

    assert df["date"].tolist() == ["2026-10-01", "2026-10-02"]
    first = df.iloc[0]
    assert all(pd.isna(first[c]) for c in ["n_scored"] + list(FOUR_WEEK_METRICS))
    second = df.iloc[1]
    assert second["n_scored"] == 0.0
    assert second["n_days"] == 0.0
    assert pd.isna(second["hit_rate_minus_majority"])
    assert build_four_week_trend([]).empty
