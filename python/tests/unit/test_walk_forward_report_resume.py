"""週の Walk-Forward レポートを、消えた回の続きから進める（#1931）

W45・W46 は全銘柄の約6時間が起き直しで2回とも消え、補完の回も頭からやり直して週の分が
書かれなかった。銘柄ごとの結果を週の途中のファイルに書き足し、同じ週の補完では済んだ銘柄を飛ばす。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.walk_forward_progress import week_key_at
from src.backtest.walk_forward_report import run_walk_forward_comparison_report
from src.domain.types import SymbolTask

_MOD = "src.backtest.walk_forward_report"
_SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
# 銘柄ごとの fold の total_return。平均は手で数えて AAA=0.02 BBB=0.04 CCC=-0.01 DDD=0.10
_FOLDS = {
    "AAA": [0.01, 0.03],
    "BBB": [0.04, 0.04],
    "CCC": [0.0, -0.02],
    "DDD": [0.10, 0.10],
}


class _Killed(BaseException):
    """デプロイの起き直しでプロセスが消えたことの代わり。except Exception では捕まらない"""


def _tasks() -> list[SymbolTask]:
    return [SymbolTask(market="us", symbol=s) for s in _SYMBOLS]


def _wf_df(symbol: str) -> pd.DataFrame:
    folds = _FOLDS[symbol]
    return pd.DataFrame({"total_return": folds, "sharpe_ratio": [1.0] * len(folds)})


def _run(report_dir: Path, week_key: str, kill_at: str | None = None) -> tuple[list[str], dict]:
    """1回分を回し、run_backtest_walk_forward が呼ばれた銘柄と結果を返す。"""
    called: list[str] = []

    def _fake_wf(**kwargs):
        symbol = kwargs["symbol"]
        if symbol == kill_at:
            raise _Killed()
        called.append(symbol)
        return None, None, _wf_df(symbol)

    result: dict = {}
    with (
        patch(f"{_MOD}.run_backtest_walk_forward", side_effect=_fake_wf),
        patch(f"{_MOD}.get_results_dir", return_value=str(report_dir.parent)),
        patch(f"{_MOD}.ensure_dir", return_value=str(report_dir)),
        patch(f"{_MOD}.save_experiment_run"),
        patch(f"{_MOD}.week_key_at", return_value=week_key),
    ):
        try:
            result = run_walk_forward_comparison_report(_tasks())
        except _Killed:
            pass
    return called, result


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    path = tmp_path / "walk_forward_reports"
    path.mkdir()
    return path


def test_resumed_run_skips_done_symbols_and_writes_all_rows(report_dir):
    called, _ = _run(report_dir, "2026-W46", kill_at="CCC")
    assert called == ["AAA", "BBB"]
    assert list(report_dir.glob("wf_summary_*.csv")) == []
    assert (report_dir / "wf_partial_2026-W46.jsonl").exists()

    called, result = _run(report_dir, "2026-W46")

    # 済んだ AAA・BBB は呼ばず、残りだけを回す
    assert called == ["CCC", "DDD"]
    summaries = list(report_dir.glob("wf_summary_*.csv"))
    assert len(summaries) == 1
    df = pd.read_csv(summaries[0])
    assert df["symbol"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert df["total_return"].tolist() == pytest.approx([0.02, 0.04, -0.01, 0.10])
    assert df["fold_count"].tolist() == [2, 2, 2, 2]
    assert result["success"] == 4
    assert result["failed"] == 0
    assert result["total"] == 4
    # 全部済んだら途中のファイルは消す
    assert not (report_dir / "wf_partial_2026-W46.jsonl").exists()


def test_partial_file_of_another_week_is_not_used(report_dir):
    called, _ = _run(report_dir, "2026-W45", kill_at="CCC")
    assert called == ["AAA", "BBB"]

    called, _ = _run(report_dir, "2026-W46")

    assert called == ["AAA", "BBB", "CCC", "DDD"]
    df = pd.read_csv(next(report_dir.glob("wf_summary_*.csv")))
    assert df["symbol"].tolist() == _SYMBOLS
    assert not (report_dir / "wf_partial_2026-W45.jsonl").exists()


def test_partial_file_with_other_conditions_is_not_used(report_dir):
    _run(report_dir, "2026-W46", kill_at="CCC")
    called: list[str] = []

    def _fake_wf(**kwargs):
        called.append(kwargs["symbol"])
        return None, None, _wf_df(kwargs["symbol"])

    with (
        patch(f"{_MOD}.run_backtest_walk_forward", side_effect=_fake_wf),
        patch(f"{_MOD}.get_results_dir", return_value=str(report_dir.parent)),
        patch(f"{_MOD}.ensure_dir", return_value=str(report_dir)),
        patch(f"{_MOD}.save_experiment_run"),
        patch(f"{_MOD}.week_key_at", return_value="2026-W46"),
    ):
        run_walk_forward_comparison_report(_tasks(), n_splits=3)

    assert called == _SYMBOLS


def test_broken_last_line_is_skipped(report_dir):
    _run(report_dir, "2026-W46", kill_at="CCC")
    path = report_dir / "wf_partial_2026-W46.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"market": "us", "symbol": "CC')

    called, _ = _run(report_dir, "2026-W46")

    assert called == ["CCC", "DDD"]


def test_resumed_report_matches_uninterrupted_report(tmp_path):
    resumed_dir = tmp_path / "resumed" / "walk_forward_reports"
    straight_dir = tmp_path / "straight" / "walk_forward_reports"
    for path in (resumed_dir, straight_dir):
        path.mkdir(parents=True)
        # 前の回: AAA=0.01 BBB=0.05。差は手で数えて AAA=+0.01 BBB=-0.01
        pd.DataFrame(
            {"total_return": [0.01, 0.05], "market": ["us", "us"], "symbol": ["AAA", "BBB"]}
        ).to_csv(path / "wf_summary_20261105_010000.csv", index=False)

    _run(resumed_dir, "2026-W46", kill_at="BBB")
    _, resumed = _run(resumed_dir, "2026-W46")
    _, straight = _run(straight_dir, "2026-W46")

    assert Path(resumed["previous_path"]).name == "wf_summary_20261105_010000.csv"
    resumed_summary = pd.read_csv(resumed["summary_path"])
    pd.testing.assert_frame_equal(resumed_summary, pd.read_csv(straight["summary_path"]))
    assert resumed_summary.columns.tolist() == [
        "total_return",
        "sharpe_ratio",
        "market",
        "symbol",
        "fold_count",
    ]

    comparison = pd.read_csv(resumed["comparison_path"])
    pd.testing.assert_frame_equal(comparison, pd.read_csv(straight["comparison_path"]))
    delta = dict(zip(comparison["symbol"], comparison["delta_total_return"]))
    assert delta["AAA"] == pytest.approx(0.01)
    assert delta["BBB"] == pytest.approx(-0.01)
    assert comparison["has_previous"].tolist() == [True, True, False, False]
    assert (
        Path(resumed["markdown_path"])
        .read_text(encoding="utf-8")
        .startswith("# Walk-Forward 比較レポート")
    )


def test_week_key_follows_scheduler_calendar_in_jst():
    # 2026-11-11T16:00Z は木曜 01:00 JST で W46。UTC ではまだ水曜だが同じ週
    assert week_key_at(datetime(2026, 11, 11, 16, 0, tzinfo=ZoneInfo("UTC"))) == "2026-W46"
    # 2026-11-15T15:30Z は月曜 00:30 JST で次の W47
    assert week_key_at(datetime(2026, 11, 15, 15, 30, tzinfo=ZoneInfo("UTC"))) == "2026-W47"


def test_schedule_allows_more_than_two_interrupted_runs():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["weekly_walk_forward_report"]
    limit = config["max_interrupted_reruns_per_period"]
    assert limit > 2
    # 起き直しが30分に1回でも、定時の回と補完の回で全銘柄の約6時間に届く
    assert timedelta(minutes=30) * (1 + limit) >= timedelta(hours=6)


def test_three_interrupted_runs_are_still_recovered(tmp_path):
    import json
    from unittest.mock import MagicMock

    from run_scheduler import SCHEDULE_CONFIG

    from src.orchestration.scheduler_queue import SchedulerQueueManager

    job_id = "weekly_walk_forward_report"
    config = {**SCHEDULE_CONFIG[job_id], "func": MagicMock()}
    events = [
        {
            "job_id": job_id,
            "reason": "scheduled",
            "status": "started",
            "period_key": "2026-W46",
            "run_id": f"run-{i}",
            "boot_id": f"old-boot-{i}",
            "started_at": f"2026-11-11T{16 + i}:00:00+00:00",
        }
        for i in range(3)
    ]
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    manager = SchedulerQueueManager({job_id: config}, state_file_path=str(state_path))

    # 木曜 05:00 JST。W46 の消えた回3つは数えない枠に入り、補完する
    now = datetime(2026, 11, 12, 5, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert manager.should_recover(job_id, now=now) is True
    assert manager.get_execution_count(job_id, "2026-W46") == 0
