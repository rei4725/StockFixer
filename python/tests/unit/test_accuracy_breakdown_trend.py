"""採点の JSON の業種・予想の大きさの内訳を日をまたいで並べる表のテスト（#1927）。

期待値は、下の偽の入力から手で計算した値で書く。
"""

import json
import math

import pytest

from src.reporting.accuracy_breakdown_trend import (
    BREAKDOWN_METRICS,
    build_breakdown_trend,
    load_breakdown_trend,
    summarize_breakdown_trend,
)
from src.reporting.accuracy_trend import load_accuracy_reports


def _bucket(n, acc, base, diff, up, corr):
    return {
        "n_scored": n,
        "direction_accuracy": acc,
        "mean_abs_error": 0.01,
        "majority_baseline_accuracy": base,
        "hit_rate_minus_majority": diff,
        "predicted_up_rate": up,
        "rank_corr": corr,
    }


def _section(by_sector=None, by_magnitude=None, has_sector=True):
    section = {"n_scored": 100, "direction_accuracy": 0.5, "by_market": {}}
    if has_sector:
        section["by_sector"] = by_sector if by_sector is not None else {}
    section["by_magnitude"] = by_magnitude if by_magnitude is not None else {}
    return section


def _write(dir_path, day, models):
    report = {"date": day, "generated_at": f"{day}T09:00:00+00:00", "models": models}
    (dir_path / f"{day}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def accuracy_dir(tmp_path):
    _write(
        tmp_path,
        "2026-11-02",
        {
            "production": {
                "1": _section(
                    by_sector={
                        "Tech": _bucket(10, 0.4, 0.6, -0.2, 1.0, 0.1),
                        "Energy": _bucket(5, 0.8, 0.6, 0.2, 0.4, None),
                    },
                    by_magnitude={
                        "q1": _bucket(4, 0.5, 0.5, 0.0, 0.25, None),
                        "q2": _bucket(4, 0.75, 0.5, 0.25, 1.0, 0.3),
                    },
                )
            }
        },
    )
    # 11-03 は by_sector の鍵が無い日、by_magnitude は空
    _write(tmp_path, "2026-11-03", {"production": {"1": _section(has_sector=False)}})
    # 11-04 は by_sector が空
    _write(tmp_path, "2026-11-04", {"production": {"1": _section(by_sector={})}})
    tech_no_corr = _bucket(30, 0.6, 0.5, 0.1, 0.8, None)
    del tech_no_corr["rank_corr"]  # 鍵の無い値
    _write(
        tmp_path,
        "2026-11-05",
        {
            "production": {
                "1": _section(
                    by_sector={
                        "Tech": tech_no_corr,
                        "Energy": _bucket(None, 0.5, 0.5, 0.0, 0.5, 0.2),
                    }
                )
            },
            "challenger": {"1": _section(by_sector={"Tech": _bucket(2, 1.0, 0.5, 0.5, 1.0, 0.4)})},
        },
    )
    return tmp_path


def _one(df, **keys):
    mask = None
    for column, value in keys.items():
        hit = df[column] == value
        mask = hit if mask is None else mask & hit
    rows = df[mask]
    assert len(rows) == 1, rows
    return rows.iloc[0]


def test_sector_rows_skip_missing_and_empty_days(accuracy_dir):
    df = load_breakdown_trend(str(accuracy_dir), kind="sector")

    # 11-02 の2行・11-05 の production 2行と challenger 1行。11-03（鍵なし）・11-04（空）は0行
    assert len(df) == 5
    assert sorted(df["date"].unique()) == ["2026-11-02", "2026-11-05"]
    assert set(df["kind"]) == {"sector"}

    tech = _one(df, date="2026-11-02", model="production", bucket="Tech")
    assert tech["horizon"] == "1"
    assert tech["n_scored"] == 10.0
    assert tech["direction_accuracy"] == pytest.approx(0.4)
    assert tech["majority_baseline_accuracy"] == pytest.approx(0.6)
    assert tech["hit_rate_minus_majority"] == pytest.approx(-0.2)
    assert tech["predicted_up_rate"] == pytest.approx(1.0)
    assert tech["rank_corr"] == pytest.approx(0.1)

    energy_0202 = _one(df, date="2026-11-02", model="production", bucket="Energy")
    assert math.isnan(energy_0202["rank_corr"])  # null は NaN
    tech_1105 = _one(df, date="2026-11-05", model="production", bucket="Tech")
    assert math.isnan(tech_1105["rank_corr"])  # 鍵の無い値は NaN
    energy_1105 = _one(df, date="2026-11-05", model="production", bucket="Energy")
    assert math.isnan(energy_1105["n_scored"])
    assert energy_1105["direction_accuracy"] == pytest.approx(0.5)


def test_magnitude_rows_skip_empty_breakdown(accuracy_dir):
    df = load_breakdown_trend(str(accuracy_dir), kind="magnitude")

    # by_magnitude は 11-02 の q1・q2 だけ。ほかの日は空
    assert len(df) == 2
    assert list(df["bucket"]) == ["q1", "q2"]
    q2 = _one(df, bucket="q2")
    assert q2["n_scored"] == 4.0
    assert q2["direction_accuracy"] == pytest.approx(0.75)
    assert math.isnan(_one(df, bucket="q1")["rank_corr"])


def test_weighted_average_over_window(accuracy_dir):
    df = load_breakdown_trend(str(accuracy_dir), kind="sector")
    summary = summarize_breakdown_trend(df, window=5)

    assert len(summary) == 3
    tech = _one(summary, model="production", bucket="Tech")
    assert tech["n_days"] == 2
    assert tech["n_scored_total"] == 40.0
    assert tech["first_date"] == "2026-11-02"
    assert tech["last_date"] == "2026-11-05"
    # (0.4*10 + 0.6*30) / 40 = 0.55
    assert tech["direction_accuracy_wavg"] == pytest.approx(0.55)
    # (0.6*10 + 0.5*30) / 40 = 0.525
    assert tech["majority_baseline_accuracy_wavg"] == pytest.approx(0.525)
    # (-0.2*10 + 0.1*30) / 40 = 0.025
    assert tech["hit_rate_minus_majority_wavg"] == pytest.approx(0.025)
    # (1.0*10 + 0.8*30) / 40 = 0.85
    assert tech["predicted_up_rate_wavg"] == pytest.approx(0.85)
    # 11-05 の rank_corr は欠けなので 11-02 の 0.1 だけ
    assert tech["rank_corr_wavg"] == pytest.approx(0.1)

    energy = _one(summary, model="production", bucket="Energy")
    assert energy["n_days"] == 2
    assert energy["n_scored_total"] == 5.0
    # 11-05 は n_scored が null なので平均から外れ、11-02 の 0.8 だけ
    assert energy["direction_accuracy_wavg"] == pytest.approx(0.8)
    # rank_corr は 11-02 が null、11-05 は重みが無いので値なし
    assert math.isnan(energy["rank_corr_wavg"])

    challenger = _one(summary, model="challenger", bucket="Tech")
    assert challenger["n_days"] == 1
    assert challenger["direction_accuracy_wavg"] == pytest.approx(1.0)


def test_weighted_average_window_takes_recent_days(accuracy_dir):
    df = load_breakdown_trend(str(accuracy_dir), kind="sector")
    summary = summarize_breakdown_trend(df, window=1)

    # production の行のある日は 11-02 と 11-05。直近1日は 11-05 だけ
    tech = _one(summary, model="production", bucket="Tech")
    assert tech["n_days"] == 1
    assert tech["n_scored_total"] == 30.0
    assert tech["direction_accuracy_wavg"] == pytest.approx(0.6)
    energy = _one(summary, model="production", bucket="Energy")
    assert math.isnan(energy["n_scored_total"])
    assert math.isnan(energy["direction_accuracy_wavg"])


def test_summary_rejects_bad_window(accuracy_dir):
    df = load_breakdown_trend(str(accuracy_dir), kind="sector")
    with pytest.raises(ValueError):
        summarize_breakdown_trend(df, window=0)


def test_build_rejects_unknown_kind(accuracy_dir):
    with pytest.raises(ValueError):
        build_breakdown_trend(load_accuracy_reports(str(accuracy_dir)), kind="market")


def test_empty_dir_gives_empty_tables(tmp_path):
    df = load_breakdown_trend(str(tmp_path), kind="sector")
    assert df.empty
    assert list(df.columns[-len(BREAKDOWN_METRICS) :]) == list(BREAKDOWN_METRICS)
    assert summarize_breakdown_trend(df, window=5).empty


def test_report_without_models_is_skipped(tmp_path):
    (tmp_path / "2026-11-02.json").write_text(json.dumps({"date": "2026-11-02"}), encoding="utf-8")
    assert load_breakdown_trend(str(tmp_path), kind="magnitude").empty
