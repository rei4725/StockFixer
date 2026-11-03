"""production と challenger を、predicted_at の UTC の日付でなく取引日でそろえるテスト（#1493）。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

# JST 10-27 09:10 のやり直しの回（取引日 10-26）と、JST 10-28 朝の2回（取引日 10-27）
_RERUN = "20261027_001027"
_MORNING = ["20261027_224340", "20261027_224629"]
_TRAINED_AT = datetime(2026, 10, 27, 4, 1, tzinfo=timezone.utc)


def _rows(model_name: str, predicted_ats: list[str], with_keys: bool = True) -> pd.DataFrame:
    frames = []
    for pa in predicted_ats:
        data = {"model_name": model_name, "predicted_at": pa, "symbol": ["7203", "6758"]}
        if with_keys:
            data["market"] = "jp"
        frame = pd.DataFrame(data)
        if not with_keys:
            frame = frame.drop(columns=["symbol"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestFilterToDaysTradingDay(unittest.TestCase):
    def setUp(self):
        trained = patch(
            "src.prediction.challenger_window.current_challenger_trained_at",
            return_value=_TRAINED_AT,
        )
        trained.start()
        self.addCleanup(trained.stop)

    def test_drops_production_rerun_row_of_other_trading_day(self):
        from src.prediction.challenger_window import filter_current_challenger, filter_to_days

        _, window = filter_current_challenger(_rows("challenger", _MORNING))
        prod = _rows("production", [_RERUN, *_MORNING])

        kept = filter_to_days(prod, window)

        self.assertEqual(set(kept["predicted_at"]), set(_MORNING))
        self.assertEqual(window.n_production, window.n_challenger)
        self.assertEqual(window.days, ["20261027"])
        self.assertEqual(window.as_dict()["window_days"], 1)

    def test_day_only_alignment_uses_jst_date(self):
        from src.prediction.challenger_window import filter_current_challenger, filter_to_days

        _, window = filter_current_challenger(_rows("challenger", _MORNING, with_keys=False))
        prod = _rows("production", [_RERUN, *_MORNING], with_keys=False)

        kept = filter_to_days(prod, window)

        self.assertIsNone(window.keys)
        self.assertEqual(set(kept["predicted_at"]), set(_MORNING))
        self.assertEqual(window.as_dict()["window_days"], 1)


if __name__ == "__main__":
    unittest.main()
