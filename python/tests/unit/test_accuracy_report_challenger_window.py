"""採点の JSON の challenger の欄を、今の challenger の版の境より後の行に絞るテスト（#1341）。"""

import json
from datetime import date, datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

_TODAY_AT = "20261019_223000"
_MID_AT = "20261016_223000"
_OLD_AT = "20261010_223000"
_SYMBOLS = [("jp", "7203"), ("jp", "6758")]


def _acc_row(model, market, symbol, predicted_at, pred, act, match):
    return {
        "market": market,
        "symbol": symbol,
        "model_name": model,
        "predicted_at": predicted_at,
        "horizon": 1,
        "predicted_price": 100.0 * (1 + pred),
        "actual_price": 100.0 * (1 + act),
        "predicted_ratio": pred,
        "actual_ratio": act,
        "direction_match": match,
    }


def _accuracy_rows() -> pd.DataFrame:
    rows = []
    for model in ("production", "challenger"):
        rows += [
            # 今日の採点
            _acc_row(model, "jp", "7203", _TODAY_AT, 0.02, 0.01, True),
            _acc_row(model, "jp", "6758", _TODAY_AT, 0.01, -0.03, False),
            # 境（10/15）より後の前の日
            _acc_row(model, "jp", "7203", _MID_AT, 0.01, 0.02, True),
            # 境より前（学び直す前の版の challenger）
            _acc_row(model, "jp", "7203", _OLD_AT, 0.03, -0.01, False),
            _acc_row(model, "jp", "6758", _OLD_AT, 0.01, -0.01, False),
        ]
    return pd.DataFrame(rows)


def _snapshot(model_version):
    return pd.DataFrame(
        [
            {"market": m, "symbol": s, "predicted_at": _TODAY_AT, "model_version": model_version}
            for m, s in _SYMBOLS
        ]
    )


def _write(tmp_path, trained_at):
    from src.prediction.accuracy_report import write_daily_accuracy_report

    with (
        patch("src.prediction.db.load_prediction_accuracy", return_value=_accuracy_rows()),
        patch(
            "src.prediction.db.load_prediction_results",
            side_effect=lambda **kw: _snapshot(kw.get("model_version")),
        ),
        patch(
            "src.prediction.challenger_window.current_challenger_trained_at",
            return_value=trained_at,
        ),
    ):
        path = write_daily_accuracy_report(
            report_date=date(2026, 10, 20),
            output_dir=str(tmp_path),
            sector_getter=lambda market, symbol: "Technology",
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _assert_production_unchanged(report):
    prod = report["models"]["production"]["1"]
    assert prod["n_scored"] == 2
    assert prod["four_week"]["n_scored"] == 5
    assert prod["four_week"]["n_days"] == 3
    assert "version_since" not in prod


def test_challenger_counts_only_rows_after_version_boundary(tmp_path):
    """challenger の4週と今日は境より後の行だけ、production は28日の行で数えること"""
    trained_at = datetime(2026, 10, 15, tzinfo=timezone.utc)
    report = _write(tmp_path, trained_at)

    challenger = report["models"]["challenger"]["1"]
    assert challenger["version_since"] == trained_at.isoformat()
    assert challenger["n_scored"] == 2
    assert challenger["four_week"]["n_scored"] == 3
    assert challenger["four_week"]["n_days"] == 2
    assert challenger["four_week"]["direction_accuracy"] == pytest.approx(2 / 3)
    _assert_production_unchanged(report)


def test_challenger_today_excludes_predictions_of_previous_version(tmp_path):
    """今日採点した予想が学び直す前の版のものなら、challenger の今日の欄に入れないこと"""
    trained_at = datetime(2026, 10, 20, tzinfo=timezone.utc)
    report = _write(tmp_path, trained_at)

    challenger = report["models"]["challenger"]["1"]
    assert challenger["version_since"] == trained_at.isoformat()
    assert challenger["n_scored"] == 0
    assert challenger["four_week"]["n_scored"] == 0
    assert challenger["top_misses"] == []
    _assert_production_unchanged(report)


def test_no_challenger_model_writes_zero_counts(tmp_path):
    """challenger の joblib が無いときは0件で書き、JSON の書き出しが落ちないこと"""
    report = _write(tmp_path, None)

    challenger = report["models"]["challenger"]["1"]
    assert challenger["version_since"] is None
    assert challenger["n_scored"] == 0
    assert challenger["four_week"]["n_scored"] == 0
    assert challenger["by_sector"] == {}
    _assert_production_unchanged(report)
