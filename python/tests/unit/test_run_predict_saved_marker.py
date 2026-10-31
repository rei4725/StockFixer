"""手で流す予想の CLI run_predict.py が日次の [3/6] と同じ保存の終わりの印の門を通す（#1449）。"""

import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import run_predict

from src.orchestration.jobs import prediction_saved_marker as marker

_PIPE = "run_predict"


def _rows(market: str, n: int) -> list:
    return [SimpleNamespace(market=market, symbol=f"{market}{i}") for i in range(n)]


class TestRunPredictSavedMarker(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "marker.txt")
        p = patch.object(marker, "_marker_path", return_value=path)
        p.start()
        self.addCleanup(p.stop)
        marker.mark_prediction_saved()
        self.today = marker.read_prediction_saved_day()
        self.assertNotEqual(self.today, "")

    def _main(self, argv: list[str], rows: list | None = None, n_jp: int = 0):
        with ExitStack() as stack:

            def p(*args, **kwargs):
                return stack.enter_context(patch(*args, **kwargs))

            p("src.orchestration.port_wiring.wire_ports")
            p(f"{_PIPE}.run_predict_single")
            p(f"{_PIPE}.run_predict_watchlist")
            p(f"{_PIPE}.predict_all_individual", return_value=rows or [])
            p(f"{_PIPE}.predict_all_unified", return_value=rows or [])
            p(f"{_PIPE}.output_top_worst_results")
            p(
                "src.watchlist.batch_runner.load_target_symbols",
                return_value=[SimpleNamespace(market="jp", symbol=str(i)) for i in range(n_jp)],
            )
            p("src.orchestration.jobs.unavailable_gates._notify_stage_failure")
            stack.enter_context(patch.object(sys, "argv", ["run_predict.py", *argv]))
            run_predict.main()

    def test_single_clears_marker(self):
        """--mode single で流すと印が今日でなくなり、8:50 は新規を止める"""
        self._main(["--mode", "single", "--market", "jp", "--symbol", "7203"])
        self.assertNotEqual(marker.read_prediction_saved_day(), self.today)

    def test_watchlist_clears_marker(self):
        self._main(["--mode", "watchlist"])
        self.assertNotEqual(marker.read_prediction_saved_day(), self.today)

    def test_individual_clears_marker(self):
        self._main(["--individual"], rows=_rows("jp", 10), n_jp=10)
        self.assertNotEqual(marker.read_prediction_saved_day(), self.today)

    def test_unified_mostly_missing_does_not_mark(self):
        """既定のモードで jp の予想が線を超えて欠けたら印を立てない"""
        self._main([], rows=_rows("jp", 1), n_jp=10)
        self.assertEqual(marker.read_prediction_saved_day(), "")

    def test_unified_within_line_marks_today(self):
        """既定のモードで jp の欠けが線の内なら印を今日にする"""
        marker.clear_prediction_saved_marker()
        self._main([], rows=_rows("jp", 10), n_jp=10)
        self.assertEqual(marker.read_prediction_saved_day(), self.today)

    def test_check_accuracy_keeps_marker(self):
        """--check-accuracy は保存しないので印に触れない"""
        with patch(f"{_PIPE}.run_check_accuracy"):
            self._main(["--check-accuracy"])
        self.assertEqual(marker.read_prediction_saved_day(), self.today)


if __name__ == "__main__":
    unittest.main()
