"""日次ドリフトチェックの流れの表の、一覧から数える列のテスト（#2167）。"""

import json

import pytest

from src.reporting.drift_check_trend import (
    TREND_COLUMNS,
    format_drift_check_csv,
    format_drift_check_markdown,
    load_drift_check_trend,
)

_OLD_COLUMNS = [
    "date",
    "n_evaluated",
    "n_triggered",
    "triggered_share",
    "warn_share",
    "warn",
    "mae_threshold",
    "hit_rate_threshold",
    "fresh_skipped",
    "retrain_targets",
    "retrain_success",
]
_NEW_COLUMNS = [
    "n_triggered_jp",
    "n_triggered_us",
    "n_mae_over",
    "n_hit_rate_over",
    "n_repeated",
    "retrain_seconds",
]


def _old_payload(date: str, n_triggered: int, targets: int) -> dict:
    return {
        "date": date,
        "recorded_at": f"{date}T00:00:00Z",
        "n_evaluated": 781,
        "n_triggered": n_triggered,
        "triggered_share": n_triggered / 781,
        "warn_share": 0.15,
        "warn": False,
        "mae_threshold": 0.03,
        "hit_rate_threshold": 0.35,
        "fresh_skipped": n_triggered - targets,
        "retrain_targets": targets,
        "retrain_success": targets,
    }


def _entry(market: str, symbol: str, mae_over: bool, hit_rate_over: bool) -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "mean_abs_error": 0.05,
        "direction_accuracy": 0.5,
        "mae_over": mae_over,
        "hit_rate_over": hit_rate_over,
        "disposition": "retrained",
    }


def _write(tmp_path, date: str, entries, seconds=None) -> None:
    n = len(entries) if isinstance(entries, list) else 0
    payload = _old_payload(date, n, n)
    payload.update(triggered_symbols=entries, retrain_seconds=seconds)
    (tmp_path / f"{date}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_columns_are_appended_to_the_right():
    """今ある列の並びは変わらず、右端に列が足される。"""
    assert TREND_COLUMNS == _OLD_COLUMNS + _NEW_COLUMNS


def test_old_records_leave_only_new_columns_empty(tmp_path):
    """一覧の無い古い記録3件は、足した列だけが空になる。"""
    days = [("2026-11-23", 16, 3), ("2026-11-24", 16, 3), ("2026-11-25", 41, 22)]
    payloads = [_old_payload(*day) for day in days]
    for payload in payloads:
        (tmp_path / f"{payload['date']}.json").write_text(json.dumps(payload), encoding="utf-8")

    df = load_drift_check_trend(str(tmp_path))

    assert df[_OLD_COLUMNS].to_dict("records") == [
        {c: p[c] for c in _OLD_COLUMNS} for p in payloads
    ]
    assert df[_NEW_COLUMNS].isna().all().all()
    assert format_drift_check_csv(df).splitlines()[1].endswith(",null" * 6)
    last = format_drift_check_markdown(df).splitlines()[-1]
    assert last.endswith("| null | null | null | null | null | null |")


def test_detail_columns_are_counted_and_zero_differs_from_null(tmp_path):
    """市場・線ごとの数とくり返しを数え、前の記録に一覧が無ければくり返しは空にする。"""
    old = _old_payload("2026-11-25", 41, 22)
    (tmp_path / "2026-11-25.json").write_text(json.dumps(old), encoding="utf-8")
    _write(
        tmp_path,
        "2026-11-26",
        [
            _entry("us", "A", True, False),
            _entry("us", "B", True, True),
            _entry("jp", "C", False, True),
        ],
        2160.5,
    )
    # 同じ銘柄コードでも市場が違えばくり返しに数えない
    _write(
        tmp_path,
        "2026-11-27",
        [
            _entry("us", "A", True, False),
            _entry("us", "C", True, False),
            _entry("jp", "C", True, False),
        ],
        30.0,
    )
    _write(tmp_path, "2026-11-30", [])
    _write(tmp_path, "2026-12-01", [_entry("us", "A", True, False)], 5.0)

    df = load_drift_check_trend(str(tmp_path))

    csv_rows = [line.split(",")[-6:] for line in format_drift_check_csv(df).splitlines()]
    assert csv_rows == [
        _NEW_COLUMNS,
        ["null"] * 6,
        ["1", "2", "2", "2", "null", "2160.5"],
        ["1", "2", "3", "0", "2", "30.0"],
        ["0", "0", "0", "0", "0", "null"],
        ["0", "1", "1", "0", "0", "5.0"],
    ]
    markdown = format_drift_check_markdown(df).splitlines()
    assert markdown[4].endswith("| 1 | 2 | 3 | 0 | 2 | 30 |")


@pytest.mark.parametrize("broken", [None, "x", {"a": 1}, [1, 2], [None]])
def test_broken_list_is_empty(tmp_path, broken):
    """一覧の形が違う・組めなかった（null）記録は、数えた列を空にする。"""
    _write(tmp_path, "2026-11-26", broken)

    df = load_drift_check_trend(str(tmp_path))

    assert len(df) == 1
    assert df[_NEW_COLUMNS].isna().all().all()
