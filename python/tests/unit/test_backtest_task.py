"""
BacktestTask / ReturnRegressionTask のユニットテスト
"""

import unittest

import pandas as pd

from src.backtest.task import BacktestTask, ReturnRegressionTask, _build_threshold_series


def _make_price_df(close_prices: list, col="Close") -> pd.DataFrame:
    """テスト用の価格 DataFrame を生成する"""
    idx = pd.date_range("2024-01-01", periods=len(close_prices), freq="D")
    return pd.DataFrame({col: close_prices}, index=idx)


class TestReturnRegressionTaskLabels(unittest.TestCase):
    """make_labels のテスト"""

    def setUp(self):
        self.task = ReturnRegressionTask()

    def test_label_col_is_y(self):
        self.assertEqual(self.task.label_col, "y")

    def test_make_labels_returns_series(self):
        df = _make_price_df([100, 110, 105, 120, 115])
        labels = self.task.make_labels(df)
        self.assertIsInstance(labels, pd.Series)

    def test_make_labels_length_matches_df(self):
        df = _make_price_df([100, 110, 105, 120, 115])
        labels = self.task.make_labels(df)
        self.assertEqual(len(labels), len(df))

    def test_make_labels_last_row_is_nan(self):
        """翌日変化率なので最終行は NaN になる"""
        df = _make_price_df([100, 110, 105])
        labels = self.task.make_labels(df)
        self.assertTrue(pd.isna(labels.iloc[-1]))

    def test_make_labels_correct_values(self):
        """(Close_t+1 - Close_t) / Close_t が正しく計算される"""
        df = _make_price_df([100.0, 110.0, 99.0])
        labels = self.task.make_labels(df)
        # 1日目: (110-100)/100 = 0.1
        self.assertAlmostEqual(labels.iloc[0], 0.1, places=10)
        # 2日目: (99-110)/110 ≈ -0.1
        self.assertAlmostEqual(labels.iloc[1], (99 - 110) / 110, places=10)

    def test_make_labels_works_with_lowercase_close(self):
        """列名が 'close'（小文字）でも動作する"""
        df = _make_price_df([100.0, 120.0, 110.0], col="close")
        labels = self.task.make_labels(df)
        self.assertAlmostEqual(labels.iloc[0], 0.2, places=10)

    def test_make_labels_name_is_label_col(self):
        """返却 Series の name が label_col と一致する"""
        df = _make_price_df([100, 110])
        labels = self.task.make_labels(df)
        self.assertEqual(labels.name, self.task.label_col)


class TestReturnRegressionTaskSignal(unittest.TestCase):
    """make_signal のテスト"""

    def _pred(self, values: list) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
        return pd.Series(values, index=idx)

    def test_signal_buy_when_pred_positive(self):
        task = ReturnRegressionTask(threshold=0.0)
        pred = self._pred([0.01, 0.02, 0.005])
        signal = task.make_signal(pred)
        self.assertTrue((signal == 1).all())

    def test_signal_sell_when_pred_negative(self):
        task = ReturnRegressionTask(threshold=0.0)
        pred = self._pred([-0.01, -0.02, -0.001])
        signal = task.make_signal(pred)
        self.assertTrue((signal == -1).all())

    def test_signal_hold_when_pred_zero(self):
        task = ReturnRegressionTask(threshold=0.0)
        pred = self._pred([0.0, 0.0])
        signal = task.make_signal(pred)
        self.assertTrue((signal == 0).all())

    def test_signal_threshold_suppresses_weak_signal(self):
        """閾値以下の予測は hold になる"""
        task = ReturnRegressionTask(threshold=0.005)
        pred = self._pred([0.003, -0.003, 0.01, -0.01])
        signal = task.make_signal(pred)
        self.assertEqual(signal.iloc[0], 0)  # 0.003 < 0.005 → hold
        self.assertEqual(signal.iloc[1], 0)  # -0.003 > -0.005 → hold
        self.assertEqual(signal.iloc[2], 1)  # 0.01 > 0.005 → buy
        self.assertEqual(signal.iloc[3], -1)  # -0.01 < -0.005 → sell

    def test_signal_returns_series_with_same_index(self):
        task = ReturnRegressionTask()
        pred = self._pred([0.01, -0.01, 0.0])
        signal = task.make_signal(pred)
        self.assertIsInstance(signal, pd.Series)
        pd.testing.assert_index_equal(signal.index, pred.index)

    def test_signal_values_are_integers(self):
        """シグナルは整数 (-1, 0, 1) のみ"""
        task = ReturnRegressionTask(threshold=0.003)
        pred = self._pred([0.01, -0.01, 0.001, -0.001])
        signal = task.make_signal(pred)
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))


class TestAdaptiveThresholdSeries(unittest.TestCase):
    def _pred(self, values: list[float]) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
        return pd.Series(values, index=idx)

    def test_short_series_keeps_base_threshold(self):
        pred = self._pred([0.001, 0.002, -0.001, 0.003])

        threshold = _build_threshold_series(pred, base_threshold=0.005, window=20)

        self.assertTrue((threshold == 0.005).all())

    def test_high_volatility_expands_threshold(self):
        pred = self._pred(
            [0.001] * 20
            + [0.020, -0.018, 0.022, -0.021, 0.019, -0.020, 0.018, -0.019, 0.017, 0.006]
        )

        threshold = _build_threshold_series(pred, base_threshold=0.005, window=20)

        self.assertLess(threshold.iloc[19], 0.005)
        self.assertGreater(threshold.iloc[-1], 0.005)

    def test_make_signal_suppresses_borderline_buy_in_high_volatility(self):
        pred = self._pred(
            [0.001] * 20
            + [0.020, -0.018, 0.022, -0.021, 0.019, -0.020, 0.018, -0.019, 0.017, 0.006]
        )
        task = ReturnRegressionTask(threshold=0.005, adaptive_window=20)

        signal = task.make_signal(pred)

        self.assertEqual(signal.iloc[-1], 0)


class TestBacktestTaskProtocol(unittest.TestCase):
    """BacktestTask Protocol チェック"""

    def test_return_regression_task_is_backtest_task(self):
        task = ReturnRegressionTask()
        self.assertIsInstance(task, BacktestTask)


if __name__ == "__main__":
    unittest.main()
