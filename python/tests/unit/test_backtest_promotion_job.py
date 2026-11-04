"""
バックテストによる仮の入れ替えの夜間ジョブ・通知・入口・週次との排他のテスト

モデルと結果の置き場は tmp_path に向ける。学習・評価・Discord は差し替える。
状態のファイル・入れ替え・戻し・見張りの判定は本物を使う。
"""

import json
import os
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from filelock import Timeout

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs import backtest_promotion_job as job
from src.orchestration.scheduler_queue import SchedulerQueueManager
from src.prediction.backtest_promotion import state as st
from src.prediction.types import BacktestPromotionState, HoldoutEvaluation, WinDecision
from src.reporting.discord.notifications_model import (
    send_backtest_promotion_result as _real_send_result,
)
from src.reporting.discord.notifications_model import (
    send_backtest_promotion_rollback as _real_send_rollback,
)

_VERSION = "abc123def456"
_DATES = [pd.Timestamp("2026-10-06") + pd.Timedelta(days=i) for i in range(20)]
_CANDIDATE = {
    "n_baseline": 400,
    "hit_rate_minus_majority": 0.05,
    "mae": 0.010,
    "rank_corr": 0.10,
    "predicted_up_rate": 0.55,
}
_PRODUCTION = {
    "n_baseline": 400,
    "hit_rate_minus_majority": 0.02,
    "mae": 0.012,
    "rank_corr": 0.05,
    "predicted_up_rate": 0.99,
}


def _evaluation(won: bool = True, passed: bool = True) -> HoldoutEvaluation:
    reasons = [] if won else ["MAE が production より小さくない"]
    if not passed:
        reasons.append("道の一致の検査に落ちた")
    return HoldoutEvaluation(
        cutoff="2026-10-06",
        dates=[d.strftime("%Y-%m-%d") for d in _DATES],
        n_symbols=20,
        candidate=dict(_CANDIDATE),
        production=dict(_PRODUCTION),
        daily=[{"date": "2026-10-06", "candidate": 0.05, "production": 0.02, "diff": 0.03}],
        path_check={"compared": 50, "max_abs_diff": 0.0 if passed else 0.1, "passed": passed},
        recorded_reference={"n_matched": 0, "direction_agreement": None, "median_ratio_diff": None},
        decision=WinDecision(won=not reasons, reasons=reasons),
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """置き場を tmp_path に向け、学習・評価・通知を差し替える。"""
    models = tmp_path / "models"
    results = tmp_path / "results"
    unified = models / "unified"
    unified.mkdir(parents=True)
    results.mkdir()
    monkeypatch.setattr("src.utils.data_path_utils.get_models_dir", lambda: str(models))
    # 定義元の差し替えは、この間に初めて import した消費先に残るので環境変数で向ける
    monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(results))
    for prod, cand in zip(st.SWAP_PRODUCTION_NAMES, st.CANDIDATE_NAMES):
        (unified / f"{prod}.joblib").write_bytes(f"old {prod}".encode())
        (unified / f"{cand}.joblib").write_bytes(f"new {prod}".encode())

    mocks = {
        "train": MagicMock(),
        "evaluate": MagicMock(return_value=_evaluation()),
        "result": MagicMock(return_value=True),
        "swap": MagicMock(return_value=True),
        "rollback": MagicMock(return_value=True),
        "accuracy": MagicMock(return_value=pd.DataFrame()),
    }
    monkeypatch.setattr(job, "prediction_code_version", lambda: _VERSION)
    monkeypatch.setattr(job, "latest_holdout_dates", lambda: list(_DATES))
    monkeypatch.setattr(job, "train_unified_model", mocks["train"])
    monkeypatch.setattr(job, "load_unified_model", lambda name: f"model:{name}")
    monkeypatch.setattr(job, "evaluate_holdout", mocks["evaluate"])
    monkeypatch.setattr(job, "load_prediction_accuracy", mocks["accuracy"])
    nm = "src.reporting.discord.notifications_model"
    monkeypatch.setattr(f"{nm}.send_backtest_promotion_result", mocks["result"])
    monkeypatch.setattr(f"{nm}.send_backtest_promotion_swap", mocks["swap"])
    monkeypatch.setattr(f"{nm}.send_backtest_promotion_rollback", mocks["rollback"])
    mocks["unified"] = unified
    mocks["results"] = results / "backtest_promotion"
    # 既定では stock_features は今の版で作り直し済み
    st.save_features_version(_VERSION)
    return mocks


def _result_files(env) -> list:
    folder = env["results"]
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.glob("*.json") if p.name not in ("state.json", "features_version.json")
    )


def _read_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _watch(backups: list[str]) -> dict:
    return {
        "version": "oldver000000",
        "swapped_at": "2026-10-01T22:30:00+09:00",
        "swap_date": "2026-10-01",
        "backups": backups,
        "backtest_candidate": dict(_CANDIDATE),
        "backtest_old_production": {"hit_rate_minus_majority": 0.02, "rank_corr": 0.05},
    }


def _scored_rows(n_days: int, predicted: list[float], actual: list[float]) -> pd.DataFrame:
    """swap_date の後の n_days 日ぶんの採点済みの行（日ごとに同じ predicted/actual）。"""
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp("2026-10-02") + pd.Timedelta(days=i)).strftime("%Y%m%d")
        for j, (p, a) in enumerate(zip(predicted, actual)):
            rows.append(
                {
                    "market": "jp",
                    "symbol": f"{j:04d}",
                    "predicted_at": f"{day}_073000",
                    "predicted_ratio": p,
                    "actual_ratio": a,
                }
            )
    # swap より前の回と、まだ採点されていない回は数えない
    rows.append(
        {
            "market": "jp",
            "symbol": "0000",
            "predicted_at": "20260930_073000",
            "predicted_ratio": 0.01,
            "actual_ratio": 0.01,
        }
    )
    rows.append(
        {
            "market": "jp",
            "symbol": "0000",
            "predicted_at": "20261030_073000",
            "predicted_ratio": 0.01,
            "actual_ratio": None,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 評価するかどうか
# ---------------------------------------------------------------------------


def test_stale_features_skip_without_marking_version(env):
    st.save_features_version("oldcode00000")

    result = job.run_backtest_promotion()

    assert result["outcome"] == "skipped"
    assert result["reason"] == "特徴量がまだ今のコードで作り直されていない"
    assert result["features_version"] == "oldcode00000"
    assert result["result_path"] is None
    assert st.load_state().last_evaluated_version is None
    env["train"].assert_not_called()
    env["result"].assert_not_called()
    assert _result_files(env) == []


def test_missing_features_marker_skips(env):
    (env["results"] / "features_version.json").unlink()

    result = job.run_backtest_promotion()

    assert result["outcome"] == "skipped"
    assert result["reason"] == "特徴量がまだ今のコードで作り直されていない"
    assert st.load_state().last_evaluated_version is None
    env["train"].assert_not_called()


def test_features_marker_equal_to_version_evaluates(env):
    result = job.run_backtest_promotion()

    assert result["outcome"] == "won"
    assert result["features_version"] == _VERSION
    env["train"].assert_called()


def test_force_evaluates_with_stale_features(env):
    st.save_features_version("oldcode00000")

    result = job.run_backtest_promotion(force=True)

    assert result["outcome"] == "won"
    env["train"].assert_called()


def test_result_records_code_version_scope(env):
    result = job.run_backtest_promotion()

    saved = _read_json(result["result_path"])
    assert saved["version_scope"] == ["src/prediction", "src/market_data", "config/settings.py"]


def test_same_version_is_not_reevaluated(env):
    st.save_state(BacktestPromotionState(last_evaluated_version=_VERSION))

    result = job.run_backtest_promotion()

    assert result["outcome"] == "skipped"
    env["train"].assert_not_called()
    env["result"].assert_not_called()
    assert _result_files(env) == []


def test_rolled_back_version_is_not_reevaluated_without_force(env):
    st.save_state(BacktestPromotionState(rolled_back_versions=[_VERSION]))

    result = job.run_backtest_promotion()

    assert result["outcome"] == "skipped"
    env["train"].assert_not_called()


def test_force_on_rolled_back_version_evaluates_but_never_swaps(env):
    st.save_state(BacktestPromotionState(rolled_back_versions=[_VERSION]))

    result = job.run_backtest_promotion(force=True)

    assert result["outcome"] == "won"
    assert result["swapped"] is False
    assert result["swap_blocked_reason"]
    assert (env["unified"] / "UnifiedStockXGBoost.joblib").read_bytes() == (
        b"old UnifiedStockXGBoost"
    )
    assert st.load_state().watch is None
    env["swap"].assert_not_called()


def test_force_reevaluates_same_version(env):
    st.save_state(BacktestPromotionState(last_evaluated_version=_VERSION))

    result = job.run_backtest_promotion(force=True)

    assert result["outcome"] == "won"
    assert env["train"].call_count == 2


def test_active_watch_blocks_evaluation_even_with_force(env):
    st.save_state(BacktestPromotionState(watch=_watch([])))
    env["accuracy"].return_value = _scored_rows(2, [0.01], [0.01])

    result = job.run_backtest_promotion(force=True)

    assert result["outcome"] == "skipped"
    env["train"].assert_not_called()


# ---------------------------------------------------------------------------
# 学習と評価
# ---------------------------------------------------------------------------


def test_candidate_trained_with_cutoff_and_without_saved_selection(env):
    job.run_backtest_promotion()

    calls = env["train"].call_args_list
    assert [c.kwargs["model_type"] for c in calls] == ["XGBoostModel", "LightGBMModel"]
    assert [c.kwargs["model_name"] for c in calls] == st.CANDIDATE_NAMES
    for c in calls:
        assert c.kwargs["train_end_date"] == _DATES[0]
        assert c.kwargs["apply_saved_exclusions"] is False
        assert c.kwargs["save_selection"] is False


def test_evaluate_holdout_gets_candidate_under_production_names(env):
    from src.prediction.predict_unified import predict_with_unified_model

    job.run_backtest_promotion()

    kwargs = env["evaluate"].call_args.kwargs
    assert kwargs["cutoff_dates"] == _DATES
    assert kwargs["candidate_models"] == [
        ("UnifiedStockXGBoost", "model:UnifiedStockXGBoost_candidate"),
        ("UnifiedStockLightGBM", "model:UnifiedStockLightGBM_candidate"),
    ]
    assert kwargs["production_models"] == [
        ("UnifiedStockXGBoost", "model:UnifiedStockXGBoost"),
        ("UnifiedStockLightGBM", "model:UnifiedStockLightGBM"),
    ]
    assert kwargs["live_predict"] is predict_with_unified_model


def test_win_swaps_and_starts_watch(env):
    result = job.run_backtest_promotion()

    assert result["outcome"] == "won"
    assert result["swapped"] is True
    assert (env["unified"] / "UnifiedStockXGBoost.joblib").read_bytes() == (
        b"new UnifiedStockXGBoost"
    )
    assert (env["unified"] / "UnifiedStockLightGBM.joblib").read_bytes() == (
        b"new UnifiedStockLightGBM"
    )
    state = st.load_state()
    assert state.last_evaluated_version == _VERSION
    watch = state.watch
    assert watch["version"] == _VERSION
    assert watch["backups"] == result["backups"]
    assert len(watch["backups"]) == 2
    assert all(os.path.exists(p) for p in watch["backups"])
    assert watch["backtest_candidate"] == _CANDIDATE
    assert watch["backtest_old_production"] == _PRODUCTION
    assert watch["swap_date"] == watch["swapped_at"][:10]

    files = _result_files(env)
    assert len(files) == 1
    assert files[0].name == os.path.basename(result["result_path"])
    data = _read_json(files[0])
    assert data["outcome"] == "won"
    assert data["swapped"] is True
    assert data["cutoff"] == "2026-10-06"
    assert data["evaluation"]["decision"]["won"] is True
    assert data["evaluation"]["path_check"]["passed"] is True
    assert data["evaluation"]["cutoff"] == "2026-10-06"
    # 評価の中身はジョブのキーを上書きしない
    assert data["outcome"] == "won" and data["dates"] == result["dates"]
    assert data["backups"] == result["backups"]
    env["result"].assert_called_once()
    env["swap"].assert_called_once()
    env["rollback"].assert_not_called()


def test_result_file_is_named_by_jst_timestamp(env):
    result = job.run_backtest_promotion()

    name = os.path.basename(result["result_path"])
    assert len(name) == len("YYYYMMDD_HHMMSS.json")
    assert name[8] == "_" and name.endswith(".json")
    assert result["started_at"].endswith("+09:00")


def test_dry_run_never_swaps(env):
    result = job.run_backtest_promotion(dry_run=True)

    assert result["outcome"] == "won"
    assert result["swapped"] is False
    assert (env["unified"] / "UnifiedStockXGBoost.joblib").read_bytes() == (
        b"old UnifiedStockXGBoost"
    )
    state = st.load_state()
    assert state.watch is None
    assert state.last_evaluated_version is None
    data = _read_json(_result_files(env)[0])
    assert data["dry_run"] is True
    env["swap"].assert_not_called()


def test_loss_marks_version_and_notifies(env):
    env["evaluate"].return_value = _evaluation(won=False)

    result = job.run_backtest_promotion()

    assert result["outcome"] == "lost"
    assert result["swapped"] is False
    assert st.load_state().last_evaluated_version == _VERSION
    assert env["result"].call_args[0][0]["outcome"] == "lost"
    env["swap"].assert_not_called()


def test_path_check_failure_is_not_comparable(env):
    env["evaluate"].return_value = _evaluation(won=True, passed=False)

    result = job.run_backtest_promotion()

    assert result["outcome"] == "not_comparable"
    assert result["swapped"] is False
    assert st.load_state().last_evaluated_version == _VERSION
    assert env["result"].call_args[0][0]["outcome"] == "not_comparable"


def test_path_check_that_compared_nothing_is_failure_and_version_not_marked(env):
    st.save_state(BacktestPromotionState(last_evaluated_version="previous0000"))
    evaluation = _evaluation(won=False, passed=False)
    evaluation.path_check = {"compared": 0, "max_abs_diff": None, "passed": False}
    env["evaluate"].return_value = evaluation

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert result["swapped"] is False
    assert st.load_state().last_evaluated_version == "previous0000"
    assert env["result"].call_args[0][0]["outcome"] == "failed"
    assert _read_json(result["result_path"])["evaluation"]["path_check"]["compared"] == 0


def test_failure_is_not_recorded_as_loss_and_version_not_marked(env):
    st.save_state(BacktestPromotionState(last_evaluated_version="previous0000"))
    env["train"].side_effect = RuntimeError("stock_features が空")

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert "stock_features が空" in result["error"]
    assert st.load_state().last_evaluated_version == "previous0000"
    data = _read_json(_result_files(env)[0])
    assert data["outcome"] == "failed"
    assert "stock_features が空" in data["error"]
    sent = env["result"].call_args[0][0]
    assert sent["outcome"] == "failed"
    env["evaluate"].assert_not_called()


def test_evaluation_error_is_failure(env):
    env["evaluate"].side_effect = ValueError("候補と production の予想がそろった行が無い")

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert st.load_state().last_evaluated_version is None


def test_swap_error_is_failure_and_version_not_marked(env, monkeypatch):
    monkeypatch.setattr(job, "swap_in_candidate", MagicMock(side_effect=OSError("disk full")))

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert result["swapped"] is False
    state = st.load_state()
    assert state.last_evaluated_version is None
    assert state.watch is None
    env["swap"].assert_not_called()


def test_state_save_failure_after_swap_undoes_swap(env, monkeypatch):
    monkeypatch.setattr(job, "save_state", MagicMock(side_effect=OSError("state を書けない")))

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert result["swapped"] is False
    assert result["swap_undone"] is True
    assert "state を書けない" in result["error"]
    for prod in st.SWAP_PRODUCTION_NAMES:
        assert (env["unified"] / f"{prod}.joblib").read_bytes() == f"old {prod}".encode()
    state = st.load_state()
    assert state.last_evaluated_version is None
    assert state.watch is None
    env["swap"].assert_not_called()
    assert env["result"].call_args[0][0]["outcome"] == "failed"
    assert _read_json(_result_files(env)[0])["swap_undone"] is True


def test_state_save_and_restore_failure_reports_both(env, monkeypatch):
    monkeypatch.setattr(job, "save_state", MagicMock(side_effect=OSError("state を書けない")))
    monkeypatch.setattr(job, "restore_backups", MagicMock(side_effect=OSError("戻せない")))

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert result["swapped"] is True
    assert result["swap_undone"] is False
    assert "state を書けない" in result["error"] and "戻せない" in result["error"]
    assert "候補のまま" in result["reason"]
    assert len(result["backups"]) == 2
    assert st.load_state().last_evaluated_version is None
    env["swap"].assert_not_called()
    sent = env["result"].call_args[0][0]
    assert sent["restore_error"] and sent["backups"] == result["backups"]


def test_result_notification_warns_when_restore_failed():
    from src.reporting.discord import notifications_model as nm

    result = {
        "outcome": "failed",
        "reason": "入れ替えた後に状態を保存できず、戻しにも失敗した。production が候補のままの恐れ",
        "error": "OSError: state を書けない\n戻しの失敗: OSError: 戻せない",
        "restore_error": "OSError: 戻せない",
        "backups": ["/m/UnifiedStockXGBoost_20261103_220000.joblib"],
        "result_path": "/r/x.json",
    }
    spec, fields = _sent(MagicMock(return_value=True), nm.send_backtest_promotion_result, result)
    assert "失敗" in spec.title
    warn = [f for f in fields if "候補のまま" in f["name"]]
    assert warn and "UnifiedStockXGBoost_20261103_220000" in warn[0]["value"]
    assert any("戻せない" in f["value"] for f in fields)


def test_unreadable_state_is_failure_and_notified(env, monkeypatch):
    monkeypatch.setattr(job, "load_state", MagicMock(side_effect=ValueError("壊れた JSON")))

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert "壊れた JSON" in result["error"]
    assert st.unified_model_lock().is_locked is False
    env["result"].assert_called_once()
    env["train"].assert_not_called()


def test_lock_timeout_is_skipped_and_notified_once(env, monkeypatch):
    lock = MagicMock()
    lock.acquire.side_effect = Timeout("lock")
    monkeypatch.setattr(job, "unified_model_lock", lambda timeout_s: lock)

    result = job.run_backtest_promotion()

    assert result["outcome"] == "skipped"
    assert result["lock_timeout"] is True
    env["train"].assert_not_called()
    env["result"].assert_called_once()
    assert len(_result_files(env)) == 1
    lock.release.assert_not_called()


def test_runs_inside_unified_model_lock(env):
    seen = []
    env["train"].side_effect = lambda **kw: seen.append(st.unified_model_lock().is_locked)

    job.run_backtest_promotion()

    assert seen == [True, True]
    assert st.unified_model_lock().is_locked is False


def test_notification_failure_does_not_break_the_job(env):
    env["result"].side_effect = RuntimeError("webhook down")

    result = job.run_backtest_promotion()

    assert result["outcome"] == "won"
    assert st.load_state().last_evaluated_version == _VERSION


# ---------------------------------------------------------------------------
# 見張り
# ---------------------------------------------------------------------------


def _swapped_env(env):
    """入れ替えた後の状態を作る（本番は新しい中身、写しは古い中身）。"""
    backups = st.swap_in_candidate()
    st.save_state(BacktestPromotionState(last_evaluated_version=_VERSION, watch=_watch(backups)))
    return backups


def test_watch_waits_until_five_scored_days(env):
    _swapped_env(env)
    env["accuracy"].return_value = _scored_rows(4, [0.01, 0.02], [-0.01, 0.02])

    result = job.run_backtest_promotion()

    assert result["outcome"] == "skipped"
    assert result["watch"]["judged"] is False
    assert result["watch"]["scored_days"] == 4
    assert st.load_state().watch is not None
    env["rollback"].assert_not_called()
    env["train"].assert_not_called()
    assert _result_files(env) == []


def test_watch_rolls_back_and_records_version(env):
    _swapped_env(env)
    # 全部上がる予想で、実際とは逆の順位 → 見張りに当たり、旧 production のほうが良い
    predicted = [0.01, 0.02, 0.03, 0.04]
    actual = [0.02, 0.01, -0.01, -0.02]
    env["accuracy"].return_value = _scored_rows(6, predicted, actual)

    result = job.run_backtest_promotion()

    watch = result["watch"]
    assert watch["judged"] is True
    assert watch["scored_days"] == 6
    assert watch["rolled_back"] is True
    assert watch["triggered"]
    assert (env["unified"] / "UnifiedStockXGBoost.joblib").read_bytes() == (
        b"old UnifiedStockXGBoost"
    )
    state = st.load_state()
    assert state.watch is None
    assert "oldver000000" in state.rolled_back_versions
    env["rollback"].assert_called_once()
    assert env["rollback"].call_args[0][0]["watch"]["rolled_back"] is True
    data = _read_json(_result_files(env)[0])
    assert data["watch"]["rolled_back"] is True
    # 見張りに使うのは swap の後の採点済みの日の、最初の5日
    assert watch["live"]["n_baseline"] == 5 * len(predicted)
    env["train"].assert_not_called()


def test_watch_triggered_but_old_worse_keeps_model_and_notifies(env):
    backups = _swapped_env(env)
    state = st.load_state()
    state.watch["backtest_old_production"] = {"hit_rate_minus_majority": -0.5, "rank_corr": -0.9}
    st.save_state(state)
    env["accuracy"].return_value = _scored_rows(
        5, [0.01, 0.02, 0.03, 0.04], [0.02, 0.01, -0.01, -0.02]
    )

    result = job.run_backtest_promotion()

    watch = result["watch"]
    assert watch["judged"] is True
    assert watch["triggered"]
    assert watch["old_is_better"] is False
    assert watch["rolled_back"] is False
    assert (env["unified"] / "UnifiedStockXGBoost.joblib").read_bytes() == (
        b"new UnifiedStockXGBoost"
    )
    state = st.load_state()
    assert state.watch is None
    assert state.rolled_back_versions == []
    assert all(os.path.exists(p) for p in backups)
    env["rollback"].assert_called_once()
    assert len(_result_files(env)) == 1


def test_watch_rollback_then_state_save_failure_keeps_rollback_in_result(env, monkeypatch):
    backups = _swapped_env(env)
    env["accuracy"].return_value = _scored_rows(
        5, [0.01, 0.02, 0.03, 0.04], [0.02, 0.01, -0.01, -0.02]
    )
    monkeypatch.setattr(job, "save_state", MagicMock(side_effect=OSError("state を書けない")))

    result = job.run_backtest_promotion()

    # 本番は旧モデルに戻っている
    for prod in st.SWAP_PRODUCTION_NAMES:
        assert (env["unified"] / f"{prod}.joblib").read_bytes() == f"old {prod}".encode()
    watch = result["watch"]
    assert watch["judged"] is True
    assert watch["rolled_back"] is True
    assert watch["backups"] == backups
    assert "state を書けない" in watch["error"]
    assert result["outcome"] == "failed"
    assert "旧モデルに戻した" in result["reason"]
    data = _read_json(_result_files(env)[0])
    assert data["watch"]["rolled_back"] is True
    assert "state を書けない" in data["watch"]["error"]
    # 状態は書けなかったので見張りは残る（次の夜にもう一度判定する）
    assert st.load_state().watch is not None
    env["rollback"].assert_called_once()
    env["result"].assert_called_once()
    env["train"].assert_not_called()

    # env は通知を差し替えているので、本物の通知の関数で中身を確かめる
    _, fields = _sent(MagicMock(return_value=True), _real_send_result, result)
    restored = [f for f in fields if "旧モデルに戻した" in f["name"]]
    assert restored and os.path.basename(backups[0]) in restored[0]["value"]
    spec, fields = _sent(MagicMock(return_value=True), _real_send_rollback, result)
    assert "戻した" in spec.title
    assert any("状態を保存できなかった" in f["name"] for f in fields)
    assert any(os.path.basename(backups[0]) in f["value"] for f in fields)


def test_watch_error_keeps_watch_and_fails(env):
    _swapped_env(env)
    env["accuracy"].side_effect = RuntimeError("DB が読めない")

    result = job.run_backtest_promotion()

    assert result["outcome"] == "failed"
    assert "DB が読めない" in result["error"]
    assert st.load_state().watch is not None
    env["result"].assert_called_once()
    env["train"].assert_not_called()


def test_dry_run_does_not_judge_watch(env):
    _swapped_env(env)
    env["accuracy"].return_value = _scored_rows(6, [0.01, 0.02], [0.02, -0.02])

    result = job.run_backtest_promotion(dry_run=True)

    assert result["outcome"] == "skipped"
    assert st.load_state().watch is not None
    env["accuracy"].assert_not_called()


# ---------------------------------------------------------------------------
# 週次の学習との排他
# ---------------------------------------------------------------------------

_WINS = {
    "challenger_wins": True,
    "production_hit_rate": 0.5,
    "production_sharpe": 0.5,
    "challenger_hit_rate": 0.6,
    "challenger_sharpe": 0.8,
}


def _weekly_patches(stack: ExitStack, mock_train, promote_result, mock_send):
    gate = MagicMock(eligible=True, reason="ok", criteria={})
    stack.enter_context(patch("config.settings.AUTO_PROMOTE_MODEL", True))
    stack.enter_context(
        patch("src.prediction.promotion_gate.evaluate_promotion", MagicMock(return_value=gate))
    )
    stack.enter_context(patch("src.prediction.promotion_gate.save_promotion_result"))
    stack.enter_context(
        patch(
            "src.prediction.shadow_evaluation.evaluate_shadow_models",
            MagicMock(return_value=_WINS),
        )
    )
    stack.enter_context(
        patch(
            "src.prediction.shadow_evaluation.promote_unified_challenger",
            MagicMock(return_value=promote_result),
        )
    )
    stack.enter_context(
        patch("src.orchestration.jobs.weekly_training._challengers_trained_this_week", lambda n: [])
    )
    stack.enter_context(patch("src.prediction.shadow_record.save_weekly_shadow_record"))
    stack.enter_context(
        patch("src.prediction.unified_model_pipeline.train_unified_model", mock_train)
    )
    stack.enter_context(patch("src.prediction.prediction_pipeline.run_accuracy_check"))
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_drift_alert"))
    stack.enter_context(
        patch("src.reporting.discord.discord_utils.send_weekly_training_completion")
    )
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_promotion_result"))
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_webhook_text", mock_send))


def test_weekly_training_holds_unified_model_lock(env):
    from src.orchestration.jobs.weekly import run_weekly_training

    seen = []
    mock_train = MagicMock(side_effect=lambda **kw: seen.append(st.unified_model_lock().is_locked))
    with ExitStack() as stack:
        _weekly_patches(stack, mock_train, {"promoted": [], "skipped": []}, MagicMock())
        run_weekly_training()

    assert seen == [True, True]
    assert st.unified_model_lock().is_locked is False


def test_weekly_training_lock_timeout_notifies_and_does_not_train(env, monkeypatch):
    from src.orchestration.jobs.weekly import run_weekly_training

    lock = MagicMock()
    lock.acquire.side_effect = Timeout("lock")
    monkeypatch.setattr(
        "src.prediction.backtest_promotion.state.unified_model_lock", lambda *a, **k: lock
    )
    mock_train = MagicMock()
    mock_send = MagicMock()
    with ExitStack() as stack:
        _weekly_patches(stack, mock_train, {"promoted": [], "skipped": []}, mock_send)
        with pytest.raises(NonRetryableJobError, match="ロック"):
            run_weekly_training()

    mock_train.assert_not_called()
    assert mock_send.call_count == 1
    assert "ロック" in mock_send.call_args[0][0]


def test_weekly_promotion_clears_watch(env):
    from src.orchestration.jobs.weekly import run_weekly_training

    st.save_state(BacktestPromotionState(last_evaluated_version=_VERSION, watch=_watch([])))
    with ExitStack() as stack:
        _weekly_patches(stack, MagicMock(), {"promoted": ["x"], "skipped": []}, MagicMock())
        run_weekly_training()

    state = st.load_state()
    assert state.watch is None
    assert state.last_evaluated_version == _VERSION


def test_weekly_without_promotion_keeps_watch(env):
    from src.orchestration.jobs.weekly import run_weekly_training

    st.save_state(BacktestPromotionState(watch=_watch([])))
    with ExitStack() as stack:
        _weekly_patches(stack, MagicMock(), {"promoted": [], "skipped": ["x"]}, MagicMock())
        run_weekly_training()

    assert st.load_state().watch is not None


# ---------------------------------------------------------------------------
# スケジュール・入口
# ---------------------------------------------------------------------------


def test_schedule_config_has_nightly_backtest_promotion():
    from run_scheduler import SCHEDULE_CONFIG

    c = SCHEDULE_CONFIG["nightly_backtest_promotion"]
    assert (c["day_of_week"], c["hour"], c["minute"]) == ("mon-sun", 22, 0)
    assert c["trigger"] == "cron"
    assert c["period"] == "daily"
    assert c["recovery_delay_minutes"] == 60
    assert c["max_executions_per_period"] == 1
    assert (
        c["description"]
        == "毎日 22:00 - 予想のコードが変わったらバックテストで候補を評価し暫定昇格"
    )


def test_schedule_job_calls_run_backtest_promotion():
    import run_scheduler

    result = {"outcome": "lost", "lock_timeout": False}
    with patch("src.orchestration.scheduler.run_backtest_promotion", return_value=result) as run:
        run_scheduler.SCHEDULE_CONFIG["nightly_backtest_promotion"]["func"]()
    run.assert_called_once_with()


def _queue_manager(job_id: str, tmp_path, notifier=None) -> SchedulerQueueManager:
    """本番の SCHEDULE_CONFIG の job_id の設定のままのキューを作る。"""
    from run_scheduler import SCHEDULE_CONFIG

    return SchedulerQueueManager(
        {job_id: SCHEDULE_CONFIG[job_id]},
        state_file_path=str(tmp_path / "queue_state.json"),
        error_notifier=notifier,
    )


def _saved_statuses(tmp_path) -> list:
    saved = json.loads((tmp_path / "queue_state.json").read_text(encoding="utf-8"))
    return [e["status"] for e in saved["events"]]


@pytest.mark.parametrize(
    "result",
    [
        {"outcome": "failed", "lock_timeout": False, "reason": "見張りの判定に失敗"},
        {"outcome": "skipped", "lock_timeout": True, "reason": "統合モデルのロックが取れない"},
    ],
)
@patch("src.orchestration.scheduler_queue.time.sleep")
def test_nightly_job_failure_is_recorded_as_error_once(mock_sleep, result, tmp_path):
    """failed とロック切れは error で記録し、heartbeat を失敗にして、呼び直さない（#1564）"""
    notifier = MagicMock()
    manager = _queue_manager("nightly_backtest_promotion", tmp_path, notifier)

    with (
        patch("src.orchestration.scheduler.run_backtest_promotion", return_value=result) as run,
        patch("src.orchestration.scheduler_queue.ping_heartbeat") as ping,
        pytest.raises(NonRetryableJobError, match=result["reason"]),
    ):
        manager.run_job("nightly_backtest_promotion", reason="scheduled", force=True)

    run.assert_called_once_with()
    mock_sleep.assert_not_called()
    ping.assert_called_once_with("nightly_backtest_promotion", success=False)
    notifier.assert_called_once()
    statuses = _saved_statuses(tmp_path)
    assert "retry" not in statuses
    assert statuses[-1] == "error"


@pytest.mark.parametrize(
    "outcome,reason",
    [
        ("won", None),
        ("lost", "MAE が production より小さくない"),
        ("not_comparable", "道の一致の検査に落ちた"),
        ("skipped", "この版はもう比べた"),
    ],
)
def test_nightly_job_without_failure_is_recorded_as_success(outcome, reason, tmp_path):
    result = {"outcome": outcome, "lock_timeout": False, "reason": reason}
    notifier = MagicMock()
    manager = _queue_manager("nightly_backtest_promotion", tmp_path, notifier)

    with (
        patch("src.orchestration.scheduler.run_backtest_promotion", return_value=result),
        patch("src.orchestration.scheduler_queue.ping_heartbeat") as ping,
    ):
        manager.run_job("nightly_backtest_promotion", reason="scheduled", force=True)

    ping.assert_called_once_with("nightly_backtest_promotion", success=True)
    notifier.assert_not_called()
    assert _saved_statuses(tmp_path)[-1] == "success"


def test_nightly_job_records_error_after_writing_result_and_notifying(env, tmp_path):
    """本物の夜間ジョブで評価が落ちても、JSON と知らせを出してから error にし、学習し直さない"""
    env["evaluate"].side_effect = RuntimeError("評価が落ちた")
    manager = _queue_manager("nightly_backtest_promotion", tmp_path)

    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        pytest.raises(NonRetryableJobError, match="評価が落ちた"),
    ):
        manager.run_job("nightly_backtest_promotion", reason="scheduled", force=True)

    assert env["train"].call_count == 2
    env["result"].assert_called_once()
    assert len(_result_files(env)) == 1
    assert _saved_statuses(tmp_path)[-1] == "error"


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_weekly_training_lock_timeout_is_error_and_recovers_same_day(
    mock_sleep, env, monkeypatch, tmp_path
):
    """週次学習のロック切れは error で記録し、呼び直さず、同じ日の補完に乗る（#1564）"""
    lock = MagicMock()
    lock.acquire.side_effect = Timeout("lock")
    monkeypatch.setattr(
        "src.prediction.backtest_promotion.state.unified_model_lock", lambda *a, **k: lock
    )
    job_id = "weekly_model_training"
    notifier = MagicMock()
    manager = _queue_manager(job_id, tmp_path, notifier)
    # 2026-11-03 は定時の曜日の火曜。定時は 01:00、補完の待ちは 15 分
    now = datetime(2026, 11, 3, 1, 20, tzinfo=manager.tz)
    period_key = manager._build_period_key(manager.schedule_config[job_id], now)

    mock_train = MagicMock()
    with (
        ExitStack() as stack,
        patch("src.orchestration.scheduler_queue.ping_heartbeat") as ping,
    ):
        _weekly_patches(stack, mock_train, {"promoted": [], "skipped": []}, MagicMock())
        with pytest.raises(NonRetryableJobError):
            manager.run_job(job_id, reason="scheduled", period_key=period_key)

    lock.acquire.assert_called_once()
    mock_train.assert_not_called()
    mock_sleep.assert_not_called()
    ping.assert_called_once_with(job_id, success=False)
    notifier.assert_called_once()
    statuses = _saved_statuses(tmp_path)
    assert "retry" not in statuses
    assert statuses[-1] == "error"
    assert manager.should_recover(job_id, now=now) is True


def test_cli_passes_flags_and_prints_result_path(capsys):
    import run_backtest_promotion as cli

    fake = MagicMock(return_value={"outcome": "won", "result_path": "/x/20261103_220000.json"})
    with (
        patch("src.orchestration.jobs.backtest_promotion_job.run_backtest_promotion", fake),
        patch("src.orchestration.port_wiring.wire_ports"),
    ):
        cli.main(["--force", "--dry-run"])

    fake.assert_called_once_with(force=True, dry_run=True)
    assert "/x/20261103_220000.json" in capsys.readouterr().out


def test_cli_without_result_file_prints_reason(capsys):
    import run_backtest_promotion as cli

    fake = MagicMock(
        return_value={"outcome": "skipped", "reason": "同じ版はもう比べた", "result_path": None}
    )
    with (
        patch("src.orchestration.jobs.backtest_promotion_job.run_backtest_promotion", fake),
        patch("src.orchestration.port_wiring.wire_ports"),
    ):
        cli.main([])

    fake.assert_called_once_with(force=False, dry_run=False)
    assert "同じ版はもう比べた" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 通知
# ---------------------------------------------------------------------------


def _sent(send, func, result) -> tuple:
    with patch("src.reporting.discord.notifications_model.send_status_fields", send):
        assert func(result) is True
    spec, fields = send.call_args[0][:2]
    return spec, fields


@pytest.mark.parametrize(
    "outcome,word",
    [
        ("won", "勝ち"),
        ("lost", "負け"),
        ("not_comparable", "比べが成り立たない"),
        ("failed", "失敗"),
    ],
)
def test_result_notification_title_follows_outcome(outcome, word):
    from src.reporting.discord import notifications_model as nm

    result = {
        "outcome": outcome,
        "version": _VERSION,
        "cutoff": "2026-10-06",
        "dates": ["2026-10-06", "2026-11-02"],
        "evaluation": {
            "n_symbols": 20,
            "candidate": _CANDIDATE,
            "production": _PRODUCTION,
            "daily": [{"diff": 0.01}, {"diff": -0.01}, {"diff": None}],
            "path_check": {"compared": 50, "max_abs_diff": 0.0, "passed": True},
            "decision": {"won": outcome == "won", "reasons": ["MAE が小さくない"]},
        },
        "error": "RuntimeError: 空" if outcome == "failed" else None,
        "result_path": "/r/20261103_220000.json",
        "dry_run": False,
    }
    spec, fields = _sent(MagicMock(return_value=True), nm.send_backtest_promotion_result, result)
    assert word in spec.title
    text = "\n".join(f["value"] for f in fields)
    assert _VERSION in text
    assert "20 銘柄" in text
    assert "1/3 日" in text
    if outcome != "failed":
        assert "MAE が小さくない" in text
    assert all(len(f["value"]) <= 1024 for f in fields)


def test_result_notification_for_skip_has_reason():
    from src.reporting.discord import notifications_model as nm

    result = {"outcome": "skipped", "reason": "ロックが取れない", "result_path": None}
    spec, fields = _sent(MagicMock(return_value=True), nm.send_backtest_promotion_result, result)
    assert "見送り" in spec.title
    assert any("ロックが取れない" in f["value"] for f in fields)


def test_swap_notification_lists_backups():
    from src.reporting.discord import notifications_model as nm

    result = {"version": _VERSION, "backups": ["/m/UnifiedStockXGBoost_20261103_220000.joblib"]}
    spec, fields = _sent(MagicMock(return_value=True), nm.send_backtest_promotion_swap, result)
    assert "入れ替え" in spec.title
    assert any("UnifiedStockXGBoost_20261103_220000" in f["value"] for f in fields)


@pytest.mark.parametrize("rolled_back,word", [(True, "戻した"), (False, "戻さない")])
def test_rollback_notification_title_follows_verdict(rolled_back, word):
    from src.reporting.discord import notifications_model as nm

    result = {
        "watch": {
            "version": _VERSION,
            "scored_days": 5,
            "live": {"hit_rate_minus_majority": -0.01, "rank_corr": -0.2, "predicted_up_rate": 1},
            "triggered": ["実測の順位相関が0以下（-0.2）"],
            "old_is_better": rolled_back,
            "rolled_back": rolled_back,
        },
        "result_path": "/r/x.json",
    }
    spec, fields = _sent(MagicMock(return_value=True), nm.send_backtest_promotion_rollback, result)
    assert word in spec.title
    assert any("順位相関" in f["value"] for f in fields)
