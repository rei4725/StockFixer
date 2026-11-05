"""全銘柄が同じ値のモデルを記録して知らせ、比べで勝ちにしないことのテスト（#1602）。"""

from __future__ import annotations

import logging
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.domain.ports import AlertLevel
from src.prediction.backtest_promotion.judge import judge_win
from src.prediction.baseline_metrics import baseline_metrics

_N_ROWS = 400
_CONSTANT = 0.00056774527  # 11-05 の朝の challenger の値
_NOISE = 6e-17  # 同じ回の challenger の標準偏差の桁


def _stock_features_like() -> pd.DataFrame:
    n_days = 400
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                "date": dates,
                "market": "us",
                "symbol": f"S{code}",
                "close_lag1": rng.uniform(99, 101, n_days),
                "y": rng.uniform(-0.05, 0.05, n_days),
            }
        )
        for code in range(4)
    ]
    return pd.concat(frames, ignore_index=True)


class _FakeBooster:
    def __init__(self, n_trees: int) -> None:
        self._n_trees = n_trees

    def num_boosted_rounds(self) -> int:
        return self._n_trees


class _FakeModel:
    """XGBoostModel と同じ口の、値を決めて返すモデル。"""

    model_name = "T"

    def __init__(self, predictions, best_iteration: int = 0, n_trees: int = 51) -> None:
        self._predictions = predictions
        self.model = SimpleNamespace(
            best_iteration=best_iteration,
            get_booster=lambda: _FakeBooster(n_trees),
        )
        self.saved = False

    def train(self, X, y, eval_set=None) -> None:
        return None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(self._predictions, Exception):
            raise self._predictions
        return pd.Series(self._predictions(len(X)), index=X.index)

    def save_model(self, path: str) -> None:
        self.saved = True


def _train(model, notifier) -> None:
    from src.prediction.unified_model_pipeline import train_unified_model

    with (
        patch("src.prediction.unified_model_pipeline.load_all_stock_data") as mock_load,
        patch("src.prediction.manager.ModelManager") as mock_mm,
        patch("src.prediction.unified_model_pipeline.load_excluded_features", return_value=[]),
        patch("src.prediction.unified_model_pipeline._compute_and_save_unified_feature_selection"),
        patch("src.prediction.ports._notification_port", notifier, create=True),
        tempfile.TemporaryDirectory() as tmp,
    ):
        mock_load.return_value = _stock_features_like()
        mock_mm.return_value.create_model.return_value = model
        train_unified_model(model_type="XGBoostModel", model_name="T", save_dir=tmp)


def test_constant_model_logs_best_iteration_and_std_and_notifies_once(caplog):
    """定数の予想のモデルは、学習のログに数字が出て、知らせが1回出て、保存はされる。"""
    notifier = MagicMock()
    model = _FakeModel(lambda n: np.full(n, _CONSTANT))

    with caplog.at_level(logging.INFO):
        _train(model, notifier)

    text = caplog.text
    assert "best_iteration=0" in text
    assert "trees=51" in text
    assert "val_pred_std=0" in text
    notifier.send_alert.assert_called_once()
    args, kwargs = notifier.send_alert.call_args
    assert "T" in args[0]
    assert kwargs["level"] is AlertLevel.WARNING
    assert model.saved is True


def test_varied_model_logs_numbers_without_notice(caplog):
    """production の桁のばらつきがあるモデルは、ログだけで知らせない。"""
    notifier = MagicMock()
    rng = np.random.default_rng(1)
    model = _FakeModel(lambda n: rng.normal(0.06, 0.02, n), best_iteration=120, n_trees=171)

    with caplog.at_level(logging.INFO):
        _train(model, notifier)

    assert "best_iteration=120" in caplog.text
    notifier.send_alert.assert_not_called()


def test_model_without_best_iteration_does_not_fail(caplog):
    """best_iteration が取れない型（MLP）でも学習は終わり、値は None と出る。"""
    notifier = MagicMock()
    rng = np.random.default_rng(2)
    model = _FakeModel(lambda n: rng.normal(0.0, 0.02, n))
    model.model = SimpleNamespace()

    with caplog.at_level(logging.INFO):
        _train(model, notifier)

    assert "best_iteration=None" in caplog.text
    assert "trees=None" in caplog.text
    assert model.saved is True


def test_check_failure_does_not_stop_training():
    """検証の行の予想で例外が出ても、保存は済み、学習は失敗にならない。"""
    notifier = MagicMock()
    model = _FakeModel(RuntimeError("予想に失敗"))

    _train(model, notifier)

    assert model.saved is True
    notifier.send_alert.assert_not_called()


def test_notice_failure_does_not_stop_training():
    """知らせの失敗で学習を止めない。"""
    notifier = MagicMock()
    notifier.send_alert.side_effect = RuntimeError("webhook")
    model = _FakeModel(lambda n: np.full(n, _CONSTANT))

    _train(model, notifier)

    assert model.saved is True


def test_wire_ports_connects_the_training_notice():
    """エントリポイントの wire_ports() で、学習の知らせが Discord へつながる。"""
    from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter
    from src.orchestration import port_wiring
    from src.prediction import ports

    saved_wired = port_wiring._wired
    with patch("src.prediction.ports._notification_port", None):
        try:
            port_wiring.wire_ports(force=True)
            assert isinstance(ports.get_notification_port(), DiscordNotificationAdapter)
        finally:
            port_wiring._wired = saved_wired


def _holdout_rows(candidate_noise: float) -> tuple[dict, dict]:
    """候補が的中・順位相関・MAE のどれでも production を上回る形の物差しを返す。

    候補の上下は candidate_noise だけで、向きはいつも「上がる」である。
    """
    actual = np.r_[np.full(240, 0.01), np.full(160, -0.01)] + np.linspace(-1e-3, 1e-3, _N_ROWS)
    days = np.tile([f"2026-10-{d:02d}" for d in range(1, 21)], _N_ROWS // 20)
    ranks = pd.Series(actual).groupby(days).rank().to_numpy()
    candidate = pd.DataFrame(
        {
            "date": days,
            "predicted_ratio": _CONSTANT + candidate_noise * ranks,
            "actual_ratio": actual,
        }
    )
    wrong = np.arange(_N_ROWS) % 3 == 0
    production = candidate.assign(predicted_ratio=np.where(wrong, -0.06, 0.06))
    return baseline_metrics(candidate), baseline_metrics(production)


def test_constant_candidate_cannot_win_the_nightly_swap():
    """的中率・順位相関・MAE で production を上回っても、定数の候補は勝ちにならない。"""
    cand, prod = _holdout_rows(_NOISE)
    assert cand["hit_rate_minus_majority"] > prod["hit_rate_minus_majority"]
    assert cand["rank_corr"] > max(prod["rank_corr"], 0)
    assert cand["mae"] < prod["mae"]

    decision = judge_win(cand, prod, [0.01] * 20, _N_ROWS)

    assert decision.won is False
    assert len(decision.reasons) == 1
    assert "ほぼ同じ値" in decision.reasons[0]


def test_varied_candidate_judged_as_before():
    """ばらつきのある候補は、同じ形の入力で今までどおり勝つ。"""
    cand, prod = _holdout_rows(1e-3)

    decision = judge_win(cand, prod, [0.01] * 20, _N_ROWS)

    assert decision.won is True and decision.reasons == []


def test_prediction_std_is_within_a_day_not_across_days():
    """日ごとに値が変わっても、同じ日の銘柄が同じ値なら標準偏差は 0 である。"""
    rows = pd.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2"],
            "predicted_ratio": [0.01, 0.01, 0.03, 0.03],
            "actual_ratio": [0.01, -0.01, 0.02, -0.02],
        }
    )

    assert baseline_metrics(rows)["prediction_std"] == 0.0


def _accuracy_rows(model_name: str, predicted: np.ndarray, hit: np.ndarray) -> pd.DataFrame:
    n = len(predicted)
    actual = np.where(hit == (predicted >= 0), 0.005, -0.005)
    return pd.DataFrame(
        {
            "model_name": model_name,
            "predicted_at": [f"202610{d % 20 + 1:02d}_224500" for d in range(n)],
            "direction_match": hit,
            "predicted_ratio": predicted,
            "actual_ratio": actual + np.linspace(-1e-4, 1e-4, n),
        }
    )


def _shadow_tables(chal_predicted: np.ndarray) -> dict[str, pd.DataFrame]:
    n = len(chal_predicted)
    return {
        "production": _accuracy_rows(
            "production", np.tile([0.06, -0.06, 0.05, 0.07], n // 4), np.arange(n) % 2 == 0
        ),
        "chal": _accuracy_rows("chal", chal_predicted, np.arange(n) % 4 != 3),
    }


def _evaluate_shadow(chal_predicted: np.ndarray) -> tuple[dict, MagicMock]:
    from src.prediction.shadow_evaluation import evaluate_shadow_models

    tables = _shadow_tables(chal_predicted)
    with (
        patch(
            "src.prediction.shadow_evaluation.load_prediction_accuracy",
            side_effect=lambda **kw: tables[kw["model_name"]],
        ),
        patch("src.prediction.shadow_evaluation._record_ab_result") as record,
    ):
        return evaluate_shadow_models(challenger_version="chal"), record


def test_constant_challenger_does_not_win_the_weekly_shadow():
    """的中率と Sharpe が上回っても定数の challenger は勝たず、定数と記録する。"""
    result, record = _evaluate_shadow(np.full(400, _CONSTANT))

    assert result["challenger_hit_rate"] > result["production_hit_rate"]
    assert result["challenger_sharpe"] > result["production_sharpe"]
    assert result["challenger_wins"] is False
    assert result["challenger_constant"] is True
    assert result["challenger_prediction_std"] < 1e-4
    assert record.call_args_list[1].kwargs["challenger_constant"] is True


def test_varied_challenger_wins_the_weekly_shadow_as_before():
    """ばらつきのある challenger の勝ち負けは今までどおりである。"""
    rng = np.random.default_rng(3)
    result, _ = _evaluate_shadow(np.abs(rng.normal(0.01, 0.01, 400)))

    assert result["challenger_wins"] is True
    assert result["challenger_constant"] is False


def _evaluate_gate(shadow_predicted: np.ndarray):
    from src.prediction.promotion_gate import evaluate_promotion

    n = len(shadow_predicted)
    tables = {
        "shadow": _accuracy_rows("shadow", shadow_predicted, np.arange(n) % 4 != 3),
        "current": _accuracy_rows("current", np.full(n, 0.06), np.arange(n) % 2 == 0),
    }
    wf = pd.DataFrame(
        {
            "model_name": ["shadow", "current"],
            "total_return": [0.1, 0.05],
            "max_drawdown": [-0.05, -0.1],
            "sharpe_ratio": [1.0, 0.5],
        }
    )
    with (
        patch("src.prediction.promotion_gate._load_latest_wf_summary", return_value=(wf, "wf.csv")),
        patch(
            "src.prediction.promotion_gate.load_prediction_accuracy",
            side_effect=lambda **kw: tables[kw["model_name"]],
        ),
        patch("src.prediction.promotion_gate._collect_run_ids", return_value=[]),
    ):
        return evaluate_promotion("shadow", "current")


def test_constant_shadow_is_not_eligible_at_the_promotion_gate():
    """他の基準を全部満たしても、定数のモデルは昇格ゲートを通らない。"""
    result = _evaluate_gate(np.full(400, _CONSTANT))

    assert result.eligible is False
    assert result.criteria["prediction_spread"]["passed"] is False
    assert "prediction_spread" in result.reason
    assert all(
        c["passed"] is True for name, c in result.criteria.items() if name != "prediction_spread"
    )


def test_varied_shadow_stays_eligible_at_the_promotion_gate():
    """ばらつきのあるモデルの判定は今までどおりである。"""
    rng = np.random.default_rng(4)
    result = _evaluate_gate(rng.normal(0.003, 0.002, 400))

    assert result.eligible is True
    assert result.criteria["prediction_spread"]["passed"] is True
