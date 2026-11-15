"""reporting/drift_thresholds.py の判定と設定の読み順のテスト（#1820）。"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest

from src.reporting import drift_thresholds as dt
from src.reporting.drift_thresholds import (
    DriftSelection,
    DriftThresholds,
    load_drift_thresholds,
    select_drifted_symbols,
)

_ENV_KEYS = (
    "DRIFT_MAE_THRESHOLD",
    "DRIFT_HIT_RATE_THRESHOLD",
    "DRIFT_MIN_SAMPLES",
    "DRIFT_MAE_RELATIVE_RATIO",
    "DRIFT_HIT_RATE_RELATIVE_MARGIN",
    "DRIFT_WARN_SHARE",
)


def _summary(rows: list[tuple[str, float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["symbol", "mean_abs_error", "direction_accuracy", "n_samples"]
    )


def _symbols(selection: DriftSelection) -> set[str]:
    return set(selection.triggered["symbol"])


class TestSelectDriftedSymbols:
    def test_fixed_thresholds_apply_when_median_is_low(self) -> None:
        # 中央値 MAE 0.01 × 1.5 = 0.015 < 0.02、中央値 Hit 0.6 - 0.2 = 0.4 < 0.45
        summary = _summary(
            [
                ("A", 0.010, 0.60, 20),
                ("B", 0.010, 0.60, 20),
                ("C", 0.025, 0.60, 20),
                ("D", 0.010, 0.44, 20),
                ("E", 0.019, 0.46, 20),
            ]
        )
        sel = select_drifted_symbols(summary, DriftThresholds())
        assert sel.mae_threshold == pytest.approx(0.02)
        assert sel.hit_rate_threshold == pytest.approx(0.4)
        # Hit の線は 0.4 に下がるので D (0.44) は超えない
        assert _symbols(sel) == {"C"}
        assert sel.n_evaluated == 5

    def test_mae_threshold_raised_to_median_ratio(self) -> None:
        # 全銘柄の MAE が高い: 中央値 0.04 × 1.5 = 0.06
        summary = _summary(
            [
                ("A", 0.04, 0.60, 20),
                ("B", 0.04, 0.60, 20),
                ("C", 0.059, 0.60, 20),
                ("D", 0.07, 0.60, 20),
                ("E", 0.03, 0.60, 20),
            ]
        )
        sel = select_drifted_symbols(summary, DriftThresholds())
        assert sel.mae_threshold == pytest.approx(0.06)
        assert _symbols(sel) == {"D"}

    def test_hit_rate_threshold_lowered_to_median_minus_margin(self) -> None:
        # 中央値 Hit 0.9 - 0.2 = 0.7 > 0.45 なので線は 0.45 のまま
        high = _summary(
            [
                ("A", 0.01, 0.90, 20),
                ("B", 0.01, 0.90, 20),
                ("C", 0.01, 0.45, 20),
                ("D", 0.01, 0.46, 20),
            ]
        )
        sel_high = select_drifted_symbols(high, DriftThresholds())
        assert sel_high.hit_rate_threshold == pytest.approx(0.45)
        assert _symbols(sel_high) == {"C"}

        # 中央値 Hit 0.5 - 0.2 = 0.3 < 0.45 なので線は 0.3 に下がる
        low = _summary(
            [
                ("A", 0.01, 0.50, 20),
                ("B", 0.01, 0.50, 20),
                ("C", 0.01, 0.29, 20),
                ("D", 0.01, 0.31, 20),
                ("E", 0.01, 0.50, 20),
            ]
        )
        sel_low = select_drifted_symbols(low, DriftThresholds())
        assert sel_low.hit_rate_threshold == pytest.approx(0.3)
        assert _symbols(sel_low) == {"C"}

    def test_min_samples_excludes_rows_from_median_and_trigger(self) -> None:
        summary = _summary(
            [
                ("A", 0.01, 0.60, 10),
                ("B", 0.01, 0.60, 10),
                ("C", 0.03, 0.60, 10),
                # サンプル不足: 線を超えていても選ばれず、中央値にも入らない
                ("X", 0.50, 0.00, 9),
                ("Y", 0.50, 0.00, 9),
                ("Z", 0.50, 0.00, 9),
                ("W", 0.50, 0.00, 9),
            ]
        )
        sel = select_drifted_symbols(summary, DriftThresholds(min_samples=10))
        assert sel.n_evaluated == 3
        assert sel.mae_threshold == pytest.approx(0.02)
        assert _symbols(sel) == {"C"}

    def test_triggered_share(self) -> None:
        summary = _summary(
            [
                ("A", 0.01, 0.60, 20),
                ("B", 0.01, 0.60, 20),
                ("C", 0.01, 0.60, 20),
                ("D", 0.03, 0.60, 20),
            ]
        )
        sel = select_drifted_symbols(summary, DriftThresholds())
        assert sel.triggered_share == pytest.approx(0.25)

    def test_no_eligible_rows_gives_zero_share_and_base_thresholds(self) -> None:
        summary = _summary([("A", 0.5, 0.0, 1), ("B", 0.5, 0.0, 2)])
        sel = select_drifted_symbols(summary, DriftThresholds(min_samples=10))
        assert sel.n_evaluated == 0
        assert sel.triggered.empty
        assert sel.triggered_share == 0.0
        assert sel.mae_threshold == pytest.approx(0.02)
        assert sel.hit_rate_threshold == pytest.approx(0.45)

    def test_share_zero_when_n_evaluated_zero_directly(self) -> None:
        sel = DriftSelection(
            triggered=_summary([]), mae_threshold=0.02, hit_rate_threshold=0.45, n_evaluated=0
        )
        assert sel.triggered_share == 0.0


class TestLoadDriftThresholds:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in _ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    @staticmethod
    def _patch_config(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> list[str]:
        calls: list[str] = []

        def fake(key: str, default: Optional[str] = None) -> Optional[str]:
            calls.append(key)
            return values.get(key, default)

        monkeypatch.setattr("src.utils.db.system_config.get_config_value", fake)
        return calls

    def test_defaults_when_no_config_and_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._patch_config(monkeypatch, {})
        th = load_drift_thresholds()
        assert th == DriftThresholds()
        assert th.mae_threshold == pytest.approx(0.02)
        assert th.hit_rate_threshold == pytest.approx(0.45)
        assert th.min_samples == 10
        assert th.mae_relative_ratio == pytest.approx(float(dt.DRIFT_MAE_RELATIVE_RATIO_DEFAULT))
        assert th.hit_rate_relative_margin == pytest.approx(
            float(dt.DRIFT_HIT_RATE_RELATIVE_MARGIN_DEFAULT)
        )
        assert th.warn_share == pytest.approx(float(dt.DRIFT_WARN_SHARE_DEFAULT))
        assert set(calls) == {
            "drift.mae_threshold",
            "drift.hit_rate_threshold",
            "drift.min_samples",
            "drift.mae_relative_ratio",
            "drift.hit_rate_relative_margin",
            "drift.warn_share",
        }

    def test_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_config(monkeypatch, {})
        monkeypatch.setenv("DRIFT_MAE_THRESHOLD", "0.05")
        monkeypatch.setenv("DRIFT_MIN_SAMPLES", "7")
        monkeypatch.setenv("DRIFT_WARN_SHARE", "0.3")
        th = load_drift_thresholds()
        assert th.mae_threshold == pytest.approx(0.05)
        assert th.min_samples == 7
        assert th.warn_share == pytest.approx(0.3)
        assert th.hit_rate_threshold == pytest.approx(0.45)

    def test_config_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_config(
            monkeypatch,
            {
                "drift.mae_threshold": "0.08",
                "drift.hit_rate_threshold": "0.40",
                "drift.min_samples": "15",
                "drift.mae_relative_ratio": "2.0",
                "drift.hit_rate_relative_margin": "0.1",
                "drift.warn_share": "0.25",
            },
        )
        for key in _ENV_KEYS:
            monkeypatch.setenv(key, "999")
        th = load_drift_thresholds()
        assert th == DriftThresholds(
            mae_threshold=0.08,
            hit_rate_threshold=0.40,
            min_samples=15,
            mae_relative_ratio=2.0,
            hit_rate_relative_margin=0.1,
            warn_share=0.25,
        )
