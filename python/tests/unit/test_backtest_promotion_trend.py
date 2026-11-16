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


def test_markdown_writes_null_for_missing():
    records = [dict(_skipped("b"), _file="20261105_220000")]
    text = format_backtest_promotion_markdown(build_backtest_promotion_trend(records))
    lines = text.splitlines()
    assert lines[0].startswith("| file | started_at |")
    assert "null" in lines[2]
