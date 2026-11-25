"""夜間の仮の入れ替えの記録の流れの表（#1890）のテスト。"""

import json

import pandas as pd

from src.reporting.backtest_promotion_trend import (
    build_backtest_promotion_trend,
    format_backtest_promotion_csv,
    format_backtest_promotion_markdown,
    load_backtest_promotion_records,
    load_backtest_promotion_trend,
)


def _metrics(hit, rank_corr, up_rate, std):
    return {
        "hit_rate_minus_majority": hit,
        "rank_corr": rank_corr,
        "predicted_up_rate": up_rate,
        "prediction_std": std,
        "n_baseline": 100,
    }


def _evaluated(started_at, reason, outcome="lost"):
    return {
        "started_at": started_at,
        "version": "abc",
        "force": False,
        "dry_run": False,
        "outcome": outcome,
        "reason": reason,
        "lock_timeout": False,
        "swapped": False,
        "evaluation": {
            "candidate": _metrics(0.01, 0.05, 0.958, 0.002),
            "production": _metrics(0.02, 0.06, 0.991, None),
            "daily": [
                {"date": "2026-11-01", "diff": 0.1},
                {"date": "2026-11-02", "diff": -0.1},
                {"date": "2026-11-03", "diff": None},
                {"date": "2026-11-04", "diff": 0.2},
            ],
            "vote_dates": ["2026-11-01", "2026-11-02", "2026-11-03"],
            "markets": {
                "jp": {"n_compared": 0},
                "us": {"n_compared": 50},
            },
        },
    }


def _skipped(started_at):
    return {
        "started_at": started_at,
        "version": None,
        "force": False,
        "dry_run": False,
        "outcome": "failed",
        "reason": "学習か評価か入れ替えに失敗（負けではない）",
        "lock_timeout": False,
        "swapped": False,
        "evaluation": None,
    }


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_runs_are_sorted_by_date_and_other_files_are_skipped(tmp_path):
    _write(tmp_path / "20261111_220000.json", _evaluated("c", "x"))
    _write(tmp_path / "20261104_220000_2.json", _evaluated("b2", "x"))
    _write(tmp_path / "20261104_220000.json", _evaluated("b", "x"))
    _write(tmp_path / "20261104_014102.json", _evaluated("a", "x"))
    _write(tmp_path / "state.json", {"watch": None})
    _write(tmp_path / "features_version.json", {"version": "abc"})
    (tmp_path / "20261109_220000.json").write_text("{broken", encoding="utf-8")
    _write(tmp_path / "20261110_220000.json", [1, 2])

    records = load_backtest_promotion_records(str(tmp_path))

    assert [r["started_at"] for r in records] == ["a", "b", "b2", "c"]
    df = load_backtest_promotion_trend(str(tmp_path))
    assert list(df["file"]) == [
        "20261104_014102",
        "20261104_220000",
        "20261104_220000_2",
        "20261111_220000",
    ]


def test_missing_directory_gives_empty_table(tmp_path):
    df = load_backtest_promotion_trend(str(tmp_path / "none"))
    assert df.empty
    assert format_backtest_promotion_markdown(df) == "データなし"


def test_run_without_evaluation_is_a_missing_row():
    records = [
        dict(_evaluated("a", "x"), _file="20261104_014102"),
        dict(_skipped("b"), _file="20261105_220000"),
    ]
    df = build_backtest_promotion_trend(records)

    row = df.iloc[1]
    assert row["outcome"] == "failed"
    for column in (
        "vote_wins",
        "vote_days",
        "candidate_hit_rate_minus_majority",
        "production_predicted_up_rate",
        "n_compared_jp",
        "n_compared_us",
        "reason_daily_vote",
        "reason_other",
    ):
        assert row[column] is pd.NA, column
    assert pd.isna(row["version"])
    assert not row["swapped"]


def test_missing_is_kept_apart_from_zero():
    df = build_backtest_promotion_trend([dict(_evaluated("a", "x"), _file="20261104_014102")])
    row = df.iloc[0]

    assert row["n_compared_jp"] == 0
    assert row["n_compared_us"] == 50
    assert row["production_prediction_std"] is pd.NA
    assert row["candidate_prediction_std"] == 0.002
    assert row["candidate_predicted_up_rate"] == 0.958
    assert row["production_predicted_up_rate"] == 0.991
    # 票の日は vote_dates の3日、勝ちは diff > 0 の1日（null の日は勝ちに数えない）
    assert row["vote_wins"] == 1
    assert row["vote_days"] == 3
    csv = format_backtest_promotion_csv(df)
    assert "null" in csv
    assert ",0," in csv


def test_reason_flags_split_by_criterion():
    reason = (
        "基準を引いた的中率が production を上回らない（候補 0.01、production 0.02）; "
        "日ごとの勝ち越しが過半数に届かない（8/19 日）; "
        "候補の予想が全銘柄でほぼ同じ値（標準偏差 0.0001、線 0.001）; なにか別の理由"
    )
    records = [
        dict(_evaluated("a", reason), _file="20261104_014102"),
        dict(_evaluated("b", None, outcome="won"), _file="20261105_220000"),
    ]
    df = build_backtest_promotion_trend(records)

    lost = df.iloc[0]
    assert lost["reason_hit_rate"]
    assert lost["reason_daily_vote"]
    assert lost["reason_constant"]
    assert lost["reason_other"]
    assert not lost["reason_mae"]
    assert not lost["reason_rank_corr"]
    won = df.iloc[1]
    assert not any(won[c] for c in df.columns if c.startswith("reason_"))


# 判定の物差しの列（#2162）を足す前の列の並びと型
_BASE_DTYPES = {
    "file": "object",
    "started_at": "object",
    "version": "object",
    "force": "boolean",
    "dry_run": "boolean",
    "outcome": "object",
    "swapped": "boolean",
    "lock_timeout": "boolean",
    "vote_wins": "Int64",
    "vote_days": "Int64",
    "candidate_hit_rate_minus_majority": "Float64",
    "candidate_rank_corr": "Float64",
    "candidate_predicted_up_rate": "Float64",
    "candidate_prediction_std": "Float64",
    "production_hit_rate_minus_majority": "Float64",
    "production_rank_corr": "Float64",
    "production_predicted_up_rate": "Float64",
    "production_prediction_std": "Float64",
    "n_compared_jp": "Int64",
    "n_compared_us": "Int64",
    "reason_coverage": "boolean",
    "reason_hit_rate": "boolean",
    "reason_mae": "boolean",
    "reason_rank_corr": "boolean",
    "reason_daily_vote": "boolean",
    "reason_constant": "boolean",
    "reason_degraded": "boolean",
    "reason_path_check": "boolean",
    "reason_other": "boolean",
}
_JUDGED = [
    "candidate_mae",
    "production_mae",
    "candidate_daily_rank_corr",
    "production_daily_rank_corr",
]
_MARKET_METRICS = [
    f"{market}_{side}_{key}"
    for market in ("jp", "us")
    for side in ("candidate", "production")
    for key in ("daily_rank_corr", "hit_rate_minus_majority", "predicted_up_rate")
]


def _judged(started_at, reason="x"):
    """全体の mae・daily_rank_corr と、市場ごとの物差しのある回。"""
    record = _evaluated(started_at, reason)
    evaluation = record["evaluation"]
    evaluation["candidate"].update(mae=0.011, daily_rank_corr=-0.037)
    evaluation["production"].update(mae=0.012, daily_rank_corr=0.0)
    evaluation["markets"]["jp"].update(
        candidate=dict(_metrics(0.03, 0.5, 0.6, 0.1), daily_rank_corr=-0.054),
        production=dict(_metrics(0.0, 0.5, 0.0, 0.1), daily_rank_corr=0.034),
    )
    evaluation["markets"]["us"].update(
        candidate=dict(_metrics(-0.02, 0.5, 0.7, 0.1), daily_rank_corr=-0.06),
        production=dict(_metrics(None, 0.5, 1.0, 0.1), daily_rank_corr=0.0),
    )
    return record


def _table(*records):
    return build_backtest_promotion_trend(
        [dict(r, _file=f"2026110{i}_220000") for i, r in enumerate(records, start=1)]
    )


def test_added_columns_are_on_the_right_and_base_columns_are_kept():
    df = _table(_judged("a", "MAE が production を上回らない"), _skipped("b"))

    assert list(df.columns) == list(_BASE_DTYPES) + _JUDGED + _MARKET_METRICS
    # 文字の列の型は pandas の版で object か str になるので、数と印の列だけを比べる
    for column, dtype in _BASE_DTYPES.items():
        if dtype != "object":
            assert str(df[column].dtype) == dtype, column
    assert all(str(df[c].dtype) == "Float64" for c in _JUDGED + _MARKET_METRICS)
    base = df[list(_BASE_DTYPES)].astype(object)
    assert [None if pd.isna(v) else v for v in base.iloc[0]] == (
        ["20261101_220000", "a", "abc", False, False, "lost", False, False, 1, 3]
        + [0.01, 0.05, 0.958, 0.002, 0.02, 0.06, 0.991, None, 0, 50]
        + [False, False, True, False, False, False, False, False, False]
    )
    assert [None if pd.isna(v) else v for v in base.iloc[1]] == (
        ["20261102_220000", "b", None, False, False, "failed", False, False] + [None] * 21
    )


def test_judged_and_market_metrics_are_read_and_zero_is_not_missing():
    row = _table(_judged("a")).iloc[0]

    assert [row[c] for c in _JUDGED] == [0.011, 0.012, -0.037, 0.0]
    assert [row[c] for c in _MARKET_METRICS[:6]] == [-0.054, 0.03, 0.6, 0.034, 0.0, 0.0]
    assert [row[c] for c in _MARKET_METRICS[6:9]] == [-0.06, -0.02, 0.7]
    assert row["us_production_daily_rank_corr"] == 0.0
    assert row["us_production_hit_rate_minus_majority"] is pd.NA
    assert row["us_production_predicted_up_rate"] == 1.0


def test_added_columns_are_missing_when_keys_are_missing():
    no_markets = _judged("no_markets")
    del no_markets["evaluation"]["markets"]
    old = _evaluated("old", "x")  # mae・daily_rank_corr と、市場の candidate・production が無い
    odd = _judged("odd")
    odd["evaluation"]["markets"] = {"jp": None, "us": {"candidate": 1}}
    df = _table(_judged("a"), old, no_markets, _skipped("b"), odd)

    for index in (1, 3):
        for column in _JUDGED + _MARKET_METRICS:
            assert df.iloc[index][column] is pd.NA, (index, column)
    for index in (2, 4):
        assert [df.iloc[index][c] for c in _JUDGED] == [0.011, 0.012, -0.037, 0.0]
        for column in _MARKET_METRICS:
            assert df.iloc[index][column] is pd.NA, (index, column)


def test_judged_columns_remain_without_any_market():
    df = _table(_skipped("b"))
    assert list(df.columns)[-4:] == _JUDGED
    assert all(df.iloc[0][c] is pd.NA for c in _JUDGED)


def test_added_columns_write_null_for_missing_and_zero_for_zero():
    df = _table(_judged("a"), _skipped("b"))
    n_added = len(_JUDGED + _MARKET_METRICS)

    lines = format_backtest_promotion_markdown(df).splitlines()
    cells = [[c.strip() for c in line.strip("|").split("|")][-n_added:] for line in lines]
    assert cells[0] == _JUDGED + _MARKET_METRICS
    assert cells[2][:4] == ["0.011", "0.012", "-0.037", "0"]
    assert cells[2][-2:] == ["null", "1"]
    assert cells[3] == ["null"] * n_added

    csv_lines = format_backtest_promotion_csv(df).splitlines()
    csv_cells = [line.split(",")[-n_added:] for line in csv_lines]
    assert csv_cells[0] == _JUDGED + _MARKET_METRICS
    assert csv_cells[1][:4] == ["0.011", "0.012", "-0.037", "0.0"]
    assert csv_cells[1][-2:] == ["null", "1.0"]
    assert csv_cells[2] == ["null"] * n_added


def test_markdown_writes_null_for_missing():
    records = [dict(_skipped("b"), _file="20261105_220000")]
    text = format_backtest_promotion_markdown(build_backtest_promotion_trend(records))
    lines = text.splitlines()
    assert lines[0].startswith("| file | started_at |")
    assert "null" in lines[2]
