"""ユニットテスト: src.domain.exceptions"""

import unittest

from src.domain.exceptions import (
    BrokerError,
    ConfigError,
    DataFetchError,
    ModelError,
    PipelineError,
    RiskError,
    StockFixerError,
)


class TestExceptionHierarchy(unittest.TestCase):
    def test_all_subclasses_inherit_stock_fixer_error(self):
        for cls in (DataFetchError, ModelError, BrokerError, PipelineError, RiskError, ConfigError):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, StockFixerError))

    def test_stock_fixer_error_inherits_exception(self):
        self.assertTrue(issubclass(StockFixerError, Exception))

    def test_catch_by_base_class(self):
        for cls in (DataFetchError, ModelError, BrokerError, PipelineError, RiskError, ConfigError):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(StockFixerError):
                    raise cls("test")

    def test_risk_error_message_preserved(self):
        err = RiskError("日次損失上限超過")
        self.assertEqual(str(err), "日次損失上限超過")

    def test_config_error_message_preserved(self):
        err = ConfigError("optimal_params.json バリデーション失敗")
        self.assertEqual(str(err), "optimal_params.json バリデーション失敗")


if __name__ == "__main__":
    unittest.main()
