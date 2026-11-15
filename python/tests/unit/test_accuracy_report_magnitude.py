"""日次の的中の報告の by_magnitude と four_week.by_market のテスト（#1813）。"""

import pandas as pd

from src.prediction.accuracy_report import build_horizon_report


def _rows(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        pred = (i + 1) * 0.001 * (1 if i % 2 == 0 else -1)
        act = pred if i % 3 else -pred
        rows.append(
            {
                "market": "us" if i % 2 == 0 else "jp",
                "symbol": f"S{i}",
                "model_name": "production",
                "predicted_at": "20261113_120000",
                "horizon": 1,
                "predicted_ratio": pred,
                "actual_ratio": act,
                "direction_match": (pred > 0) == (act > 0),
            }
        )
    return pd.DataFrame(rows)


def test_by_magnitude_counts_sum_to_n_scored():
    rows = _rows(23)
    report = build_horizon_report(rows, rows, None)
    by_mag = report["by_magnitude"]
    assert list(by_mag) == ["q1", "q2", "q3", "q4", "q5"]
    assert sum(v["n_scored"] for v in by_mag.values()) == report["n_scored"] == 23
    for v in by_mag.values():
        assert "direction_accuracy" in v and "mean_abs_error" in v
    # 五分位は |predicted_ratio| の小さい順
    assert by_mag["q1"]["mean_abs_error"] <= by_mag["q5"]["mean_abs_error"]


def test_by_magnitude_empty_when_fewer_than_five():
    rows = _rows(4)
    report = build_horizon_report(rows, rows, None)
    assert report["by_magnitude"] == {}
    assert report["n_scored"] == 4


def test_four_week_has_by_market():
    rows = _rows(10)
    report = build_horizon_report(rows.head(2), rows, None)
    by_market = report["four_week"]["by_market"]
    assert set(by_market) == {"jp", "us"}
    assert sum(v["n_scored"] for v in by_market.values()) == 10


def test_existing_keys_kept():
    rows = _rows(6)
    report = build_horizon_report(rows, rows, None)
    for key in ("n_scored", "direction_accuracy", "mean_abs_error", "four_week"):
        assert key in report
    for key in ("top_misses", "by_market", "by_sector"):
        assert key in report
    for key in ("n_days", "daily_direction_accuracy_mean", "n_scored"):
        assert key in report["four_week"]
