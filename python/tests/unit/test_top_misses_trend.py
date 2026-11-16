"""top_misses_trend のテスト（#1928）。期待値は偽の入力から手で数えた値。"""

import json
import math

import pytest

from src.reporting.top_misses_trend import (
    TREND_COLUMNS,
    build_top_misses_trend,
    load_top_misses_trend,
)


def _miss(symbol, predicted, actual, error, market="jp"):
    return {
        "market": market,
        "symbol": symbol,
        "predicted_at": "2026-10-31",
        "predicted_ratio": predicted,
        "actual_ratio": actual,
        "abs_error": error,
    }


def _write(tmp_path, day, models):
    (tmp_path / f"{day}.json").write_text(
        json.dumps({"date": day, "models": models}), encoding="utf-8"
    )


@pytest.fixture
def accuracy_dir(tmp_path):
    # 1日目: 7936（上に外れ）・4385（下に外れ）・形の違う要素2つ（文字列・symbol 無し）
    _write(
        tmp_path,
        "2026-11-01",
        {
            "production": {
                "1d": {
                    "top_misses": [
                        _miss("7936", 0.05, -0.10, 0.15),
                        _miss("4385", -0.02, 0.06, 0.08),
                        "bad",
                        {"market": "jp", "abs_error": 0.9},
                    ]
                }
            }
        },
    )
    # 2日目: production に top_misses が無い。challenger に 7936（上に外れ）
    _write(
        tmp_path,
        "2026-11-02",
        {
            "production": {"1d": {"n_scored": 10}},
            "challenger": {"1d": {"top_misses": [_miss("7936", 0.01, -0.04, 0.05)]}},
        },
    )
    # 3日目: 7936 が2日目に出る（上に外れ）。5631 は値が null
    _write(
        tmp_path,
        "2026-11-03",
        {
            "production": {
                "1d": {
                    "top_misses": [
                        _miss("7936", 0.03, -0.07, 0.10),
                        _miss("5631", None, 0.02, None),
                    ]
                }
            }
        },
    )
    return str(tmp_path)


def test_hand_counted_values(accuracy_dir):
    df = load_top_misses_trend(accuracy_dir)

    assert list(df.columns) == TREND_COLUMNS
    keys = list(zip(df["model"], df["symbol"]))
    # 出た日数の多い順、同じ日数は max_abs_error の大きい順、欠けは最後
    assert keys == [
        ("production", "7936"),
        ("production", "4385"),
        ("challenger", "7936"),
        ("production", "5631"),
    ]

    top = df.iloc[0]
    assert top["horizon"] == "1d" and top["market"] == "jp"
    assert top["n_days"] == 2
    assert top["first_date"] == "2026-11-01"
    assert top["last_date"] == "2026-11-03"
    assert top["mean_abs_error"] == pytest.approx(0.125)
    assert top["max_abs_error"] == pytest.approx(0.15)
    assert (top["n_over"], top["n_under"]) == (2, 0)

    under = df.iloc[1]
    assert under["n_days"] == 1
    assert under["first_date"] == under["last_date"] == "2026-11-01"
    assert under["mean_abs_error"] == pytest.approx(0.08)
    assert (under["n_over"], under["n_under"]) == (0, 1)

    challenger = df.iloc[2]
    assert challenger["first_date"] == "2026-11-02"
    assert challenger["mean_abs_error"] == pytest.approx(0.05)
    assert (challenger["n_over"], challenger["n_under"]) == (1, 0)

    missing = df.iloc[3]
    assert missing["n_days"] == 1
    assert math.isnan(missing["mean_abs_error"]) and math.isnan(missing["max_abs_error"])
    assert (missing["n_over"], missing["n_under"]) == (0, 0)


def test_same_day_duplicate_counts_one_day():
    reports = [
        {
            "date": "2026-11-01",
            "models": {
                "production": {
                    "1d": {"top_misses": [_miss("1", 0.1, 0.0, 0.1), _miss("1", -0.1, 0.1, 0.2)]}
                }
            },
        }
    ]
    row = build_top_misses_trend(reports).iloc[0]
    assert row["n_days"] == 1
    assert row["mean_abs_error"] == pytest.approx(0.15)
    assert (row["n_over"], row["n_under"]) == (1, 1)


def test_no_misses_returns_empty_table():
    reports = [
        {"date": "2026-11-01", "models": {"production": {"1d": {"top_misses": []}}}},
        {"date": "2026-11-02", "models": "bad"},
        {"date": "2026-11-03", "models": {"production": "bad"}},
    ]
    df = build_top_misses_trend(reports)
    assert df.empty
    assert list(df.columns) == TREND_COLUMNS


def test_missing_dir_returns_empty(tmp_path):
    assert load_top_misses_trend(str(tmp_path / "none")).empty
