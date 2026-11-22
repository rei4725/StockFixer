"""prediction/unified_run_summary.py の完了のログの文面と数え方を確かめる（#2099）。

期待する文面は、本番のログ（stockfixer.log.1 の 40604 行・40826 行、2026-11-18）の行に合わせる。
"""

import logging
import unittest

from src.prediction.unified_run_summary import log_unified_prediction_summary

LOGGER_NAME = "src.prediction.prediction_pipeline"


class TestLogUnifiedPredictionSummary(unittest.TestCase):
    def test_production_line_without_errors(self) -> None:
        with self.assertLogs(LOGGER_NAME, level="DEBUG") as cm:
            log_unified_prediction_summary(
                n_rows=781,
                n_total=783,
                stale_keys=[("jp", "1111"), ("jp", "2222")],
                error_keys=[],
            )
        self.assertEqual(len(cm.records), 1)
        record = cm.records[0]
        self.assertEqual(record.levelno, logging.INFO)
        self.assertEqual(
            record.getMessage(),
            "並列予測完了（統合モデル）: 781/783銘柄, 特徴量が古いため予想なし: 2銘柄, "
            "例外で予想なし: 0銘柄, そのほかの理由で予想なし: 0銘柄",
        )
        self.assertFalse([r for r in cm.records if r.levelno >= logging.WARNING])

    def test_errors_are_counted_and_listed_sorted_in_warning(self) -> None:
        with self.assertLogs(LOGGER_NAME, level="DEBUG") as cm:
            log_unified_prediction_summary(
                n_rows=779,
                n_total=783,
                stale_keys=[("jp", "1111"), ("jp", "2222")],
                error_keys=[("jp", "7203"), ("jp", "6758")],
            )
        self.assertEqual(len(cm.records), 2)
        info, warning = cm.records
        self.assertEqual(info.levelno, logging.INFO)
        self.assertEqual(
            info.getMessage(),
            "並列予測完了（統合モデル）: 779/783銘柄, 特徴量が古いため予想なし: 2銘柄, "
            "例外で予想なし: 2銘柄, そのほかの理由で予想なし: 0銘柄",
        )
        self.assertEqual(warning.levelno, logging.WARNING)
        self.assertEqual(warning.getMessage(), "例外で予想を出せなかった銘柄: jp_6758, jp_7203")

    def test_label_changes_parenthesis(self) -> None:
        with self.assertLogs(LOGGER_NAME, level="DEBUG") as cm:
            log_unified_prediction_summary(
                n_rows=781,
                n_total=783,
                stale_keys=[("jp", "1111"), ("jp", "2222")],
                error_keys=[],
                label="challenger 統合モデル",
            )
        self.assertEqual(
            cm.records[0].getMessage(),
            "並列予測完了（challenger 統合モデル）: 781/783銘柄, "
            "特徴量が古いため予想なし: 2銘柄, 例外で予想なし: 0銘柄, "
            "そのほかの理由で予想なし: 0銘柄",
        )

    def test_other_count_is_total_minus_rows_stale_errors(self) -> None:
        with self.assertLogs(LOGGER_NAME, level="DEBUG") as cm:
            log_unified_prediction_summary(
                n_rows=770,
                n_total=783,
                stale_keys=[("jp", "1111"), ("jp", "2222"), ("us", "AAPL")],
                error_keys=[("us", "MSFT")],
            )
        # 783 - 770 - 3 - 1 = 9
        self.assertEqual(
            cm.records[0].getMessage(),
            "並列予測完了（統合モデル）: 770/783銘柄, 特徴量が古いため予想なし: 3銘柄, "
            "例外で予想なし: 1銘柄, そのほかの理由で予想なし: 9銘柄",
        )
        self.assertEqual(cm.records[1].getMessage(), "例外で予想を出せなかった銘柄: us_MSFT")


if __name__ == "__main__":
    unittest.main()
