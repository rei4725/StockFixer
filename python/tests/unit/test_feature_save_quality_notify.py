"""品質の知らせの選び分けと、記録・知らせの失敗を握る枝のテスト（#2165）"""

import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd

from src.domain.ports import AlertLevel
from src.market_data.feature_save import _notify_quality_issues, save_features_to_db
from src.market_data.quality_check import QualityCheckResult, QualityIssue

_M = "src.market_data.feature_save"
_LOGGER = "src.market_data.feature_save"

_NAN = QualityIssue(check="nan_rate", level="warning", detail="欠損率が高い")
_STEP = QualityIssue(check="price_step", level="warning", detail="段差")
_ZERO = QualityIssue(check="zero_volume", level="warning", detail="出来高ゼロ")
_MISSING = QualityIssue(check="consecutive_missing", level="error", detail="終値の連続欠け")

_TITLE = "[データ品質] jp/7203（社名）"


def _result(*issues: QualityIssue) -> QualityCheckResult:
    return QualityCheckResult(
        market="jp",
        symbol="7203",
        passed=all(i.level != "error" for i in issues),
        issues=list(issues),
    )


def _as_dicts(*issues: QualityIssue) -> list[dict]:
    return [{"check": i.check, "level": i.level, "detail": i.detail} for i in issues]


class _QualityNotifyTestBase(unittest.TestCase):
    """DB と Discord に触れる口をすべて差し替え、呼ばれた順を calls に残す"""

    def setUp(self):
        self.calls = MagicMock()
        targets = {
            "insert": "src.utils.db.quality_log.insert_quality_log",
            "suffix": f"{_M}.symbol_name_suffix",
            "upsert": f"{_M}.upsert_stock_features",
        }
        for name, target in targets.items():
            patcher = patch(target, getattr(self.calls, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls.suffix.return_value = "（社名）"
        self.port = self.calls.port
        self.insert = self.calls.insert


class TestNotifyQualityIssues(_QualityNotifyTestBase):
    def test_other_warnings_are_recorded_but_not_sent(self):
        """nan_rate・price_step の warning だけなら、全件を記録し、知らせは送らない"""
        _notify_quality_issues("jp", "7203", _result(_NAN, _STEP), self.port)

        self.insert.assert_called_once_with("jp", "7203", _as_dicts(_NAN, _STEP))
        self.port.send_alert.assert_not_called()

    def test_zero_volume_is_sent_as_warning_without_other_warnings(self):
        """zero_volume は WARNING で1通。nan_rate は本文に入らず、記録には全件入る"""
        _notify_quality_issues("jp", "7203", _result(_NAN, _ZERO), self.port)

        self.insert.assert_called_once_with("jp", "7203", _as_dicts(_NAN, _ZERO))
        self.port.send_alert.assert_called_once_with(
            title=_TITLE, message="・出来高ゼロ", level=AlertLevel.WARNING
        )

    def test_error_with_zero_volume_is_sent_as_one_error(self):
        """error と zero_volume がそろうと ERROR で1通、本文に2行。題に社名の後ろ付けが入る"""
        for issues, message in (
            ((_MISSING, _ZERO), "・終値の連続欠け\n・出来高ゼロ"),
            ((_ZERO, _MISSING), "・出来高ゼロ\n・終値の連続欠け"),
        ):
            with self.subTest(first=issues[0].check):
                self.calls.reset_mock()
                _notify_quality_issues("jp", "7203", _result(*issues), self.port)

                self.insert.assert_called_once_with("jp", "7203", _as_dicts(*issues))
                self.calls.suffix.assert_called_once_with("jp", "7203")
                self.port.send_alert.assert_called_once_with(
                    title=_TITLE, message=message, level=AlertLevel.ERROR
                )

    def test_without_port_error_is_only_recorded(self):
        """知らせの口が None なら、error があっても記録だけで、知らせの失敗のログも出ない"""
        with self.assertNoLogs(_LOGGER, level="ERROR"):
            _notify_quality_issues("jp", "7203", _result(_MISSING), None)

        self.insert.assert_called_once_with("jp", "7203", _as_dicts(_MISSING))
        self.calls.suffix.assert_not_called()

    def test_record_failure_does_not_stop_alert(self):
        """記録が例外でも、ログに残して知らせは送られる"""
        self.insert.side_effect = RuntimeError("db down")

        with self.assertLogs(_LOGGER, level="ERROR") as logs:
            _notify_quality_issues("jp", "7203", _result(_MISSING), self.port)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("品質ログ記録エラー: db down", logs.output[0])
        self.port.send_alert.assert_called_once_with(
            title=_TITLE, message="・終値の連続欠け", level=AlertLevel.ERROR
        )

    def test_alert_failure_is_not_raised_to_caller(self):
        """知らせが例外でも、ログに残して呼び手に投げない"""
        self.port.send_alert.side_effect = RuntimeError("discord down")

        with self.assertLogs(_LOGGER, level="ERROR") as logs:
            _notify_quality_issues("jp", "7203", _result(_MISSING), self.port)

        self.port.send_alert.assert_called_once()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("品質チェック Discord通知エラー: discord down", logs.output[0])
        self.insert.assert_called_once_with("jp", "7203", _as_dicts(_MISSING))


class TestSaveFeaturesToDb(_QualityNotifyTestBase):
    def test_no_result_or_no_issues_only_saves(self):
        """品質の結果が None か issue が空なら、保存だけで記録も知らせもしない"""
        data = pd.DataFrame({"Close": [1.0]})
        for label, quality in (("none", None), ("empty", _result())):
            with self.subTest(label):
                self.calls.reset_mock()
                save_features_to_db("jp", "7203", data, quality, self.port)

                self.assertEqual(self.calls.mock_calls, [call.upsert("jp", "7203", data)])

    def test_issues_are_recorded_and_sent_after_save(self):
        """issue があれば、保存の後に記録し、渡された知らせの口まで送る"""
        data = pd.DataFrame({"Close": [1.0]})

        save_features_to_db("jp", "7203", data, _result(_ZERO), self.port)

        self.assertEqual(
            self.calls.mock_calls,
            [
                call.upsert("jp", "7203", data),
                call.insert("jp", "7203", _as_dicts(_ZERO)),
                call.suffix("jp", "7203"),
                call.port.send_alert(
                    title=_TITLE, message="・出来高ゼロ", level=AlertLevel.WARNING
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
