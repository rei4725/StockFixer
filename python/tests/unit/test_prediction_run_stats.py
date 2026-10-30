"""
予想の回ごとに ATR のクリップと特徴量の0埋めを数えてログに出すことのテスト（#1318）
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

_MOD = "src.prediction.predict_unified"
_STATS = "src.prediction.prediction_run_stats"


def _features(atr: float, with_feat_b: bool = True) -> pd.DataFrame:
    data = {"feat_a": [1.0, 2.0], "atr_lag1": [atr, atr], "y": [0.0, 0.0]}
    if with_feat_b:
        data["feat_b"] = [3.0, 4.0]
    return pd.DataFrame(data)


def _model(pred: float, expected: list) -> MagicMock:
    model = MagicMock()
    model.model = model
    model.feature_names_in_ = expected
    model.predict.return_value = [pred]
    return model


class _PatchedCase(unittest.TestCase):
    def _start(self, features: pd.DataFrame, pred: float, expected: list) -> None:
        model = _model(pred, expected)
        patches = [
            patch(f"{_MOD}.load_feature_data", return_value=features),
            patch(f"{_MOD}._fetch_current_price", return_value=1000.0),
            patch(f"{_MOD}.get_service_url", return_value=None),
            patch(
                f"{_MOD}.get_cached_model",
                side_effect=lambda name: model if name == "UnifiedStockXGBoost" else None,
            ),
            patch(f"{_MOD}.load_model_weights", return_value=[1.0]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class TestClipCounting(_PatchedCase):
    def _predict(self, atr: float, pred: float):
        from src.prediction.predict_unified import predict_with_unified_model
        from src.prediction.prediction_run_stats import collect_prediction_run_stats

        self._start(_features(atr), pred, ["feat_a", "feat_b", "atr_lag1"])
        with patch(f"{_STATS}.logger"):
            with collect_prediction_run_stats("test") as stats:
                result = predict_with_unified_model("jp", "7203")
        return result, stats.snapshot()[("jp", 1)]

    def test_upper_clip_counted(self):
        """上限に当たった行は上限として数え、予想の値はクリップ後のまま"""
        # 上限は 3 × 10 / 1000 = 0.03
        result, snap = self._predict(atr=10.0, pred=0.10)
        self.assertAlmostEqual(result.diff_ratio, 0.03)
        self.assertEqual((snap.total, snap.clipped_upper, snap.clipped_lower), (1, 1, 0))

    def test_lower_clip_counted(self):
        """下限に当たった行は下限として数える"""
        result, snap = self._predict(atr=10.0, pred=-0.10)
        self.assertAlmostEqual(result.diff_ratio, -0.03)
        self.assertEqual((snap.total, snap.clipped_upper, snap.clipped_lower), (1, 0, 1))

    def test_within_range_not_counted(self):
        """レンジの内の行はクリップに数えない"""
        result, snap = self._predict(atr=10.0, pred=0.01)
        self.assertAlmostEqual(result.diff_ratio, 0.01)
        self.assertEqual((snap.total, snap.clipped_upper, snap.clipped_lower), (1, 0, 0))

    def test_no_atr_counted_separately(self):
        """ATR が無い行はクリップの判定ができない行として数える"""
        _, snap = self._predict(atr=float("nan"), pred=0.10)
        self.assertEqual((snap.total, snap.clipped_upper, snap.no_atr), (1, 0, 1))


class TestZeroFillCounting(_PatchedCase):
    def _predict(self, with_feat_b: bool):
        from src.prediction.predict_unified import predict_with_unified_model
        from src.prediction.prediction_run_stats import collect_prediction_run_stats

        self._start(_features(10.0, with_feat_b), 0.01, ["feat_a", "feat_b", "atr_lag1"])
        with patch(f"{_STATS}.logger"):
            with collect_prediction_run_stats("test") as stats:
                result = predict_with_unified_model("jp", "7203")
        return result, stats.snapshot()[("jp", 1)]

    def test_missing_column_counted(self):
        """欠けた列は0埋めの特徴量として名前ごとに数える"""
        result, snap = self._predict(with_feat_b=False)
        self.assertAlmostEqual(result.diff_ratio, 0.01)
        self.assertEqual(snap.filled_symbols, 1)
        self.assertEqual(dict(snap.filled_features), {"feat_b": 1})

    def test_no_missing_column(self):
        """欠けた列が無ければ0埋めは0件"""
        _, snap = self._predict(with_feat_b=True)
        self.assertEqual(snap.filled_symbols, 0)
        self.assertEqual(dict(snap.filled_features), {})


class TestLogSummary(unittest.TestCase):
    def test_logs_one_line_each_per_market_horizon(self):
        """市場・ホライズンごとに、クリップと0埋めを1行ずつ INFO で出す"""
        from src.prediction.prediction_run_stats import (
            collect_prediction_run_stats,
            record_clip,
            record_zero_fill,
        )

        with patch(f"{_STATS}.logger") as mock_logger:
            with collect_prediction_run_stats("daily"):
                record_clip("jp", 1, 0.10, 0.03)
                record_clip("jp", 1, -0.10, 0.03)
                record_clip("jp", 1, 0.01, 0.01)
                record_clip("jp", 1, 0.01, None)
                record_zero_fill("jp", 1, [f"f{i:02d}" for i in range(12)])
                record_zero_fill("jp", 1, [])

        lines = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]
        self.assertEqual(len(lines), 2)
        clip_line, fill_line = lines
        self.assertIn("market=jp horizon=1", clip_line)
        self.assertIn("銘柄数=4", clip_line)
        self.assertIn("上限=1（25.0%）", clip_line)
        self.assertIn("下限=1（25.0%）", clip_line)
        self.assertIn("ATRなし=1", clip_line)
        self.assertIn("0埋めの銘柄数=1/2", fill_line)
        self.assertIn("特徴量の数=12", fill_line)
        self.assertIn("f09", fill_line)
        self.assertNotIn("f10", fill_line)
        self.assertIn("ほか2件", fill_line)

    def test_not_recorded_outside_collection(self):
        """集める回の外で記録しても何も起きない"""
        from src.prediction.prediction_run_stats import record_clip

        record_clip("jp", 1, 0.10, 0.03)


class TestPredictAllUnifiedLogs(unittest.TestCase):
    @patch("src.prediction.prediction_pipeline.log_unified_prediction_summary")
    @patch("src.prediction.prediction_pipeline.get_active_symbols")
    @patch(f"{_MOD}.preload_models")
    def test_predict_all_unified_logs_stats(self, _preload, mock_symbols, _summary):
        """predict_all_unified の回の終わりにクリップの行が出る"""
        from src.prediction.prediction_pipeline import predict_all_unified

        mock_symbols.return_value = [("jp", "7203")]
        model = _model(0.10, ["feat_a", "feat_b", "atr_lag1"])
        with (
            patch(f"{_MOD}.load_feature_data", return_value=_features(10.0, False)),
            patch(f"{_MOD}._fetch_current_price", return_value=1000.0),
            patch(f"{_MOD}.get_service_url", return_value=None),
            patch(f"{_MOD}.get_cached_model", return_value=model),
            patch(f"{_MOD}.load_model_weights", return_value=[0.5, 0.5]),
            patch(f"{_STATS}.logger") as mock_logger,
        ):
            rows = predict_all_unified(max_workers=1)

        self.assertEqual(len(rows), 1)
        lines = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]
        self.assertTrue(any("上限=1（100.0%）" in line for line in lines), lines)
        self.assertTrue(any("名前=feat_b" in line for line in lines), lines)


class TestChallengerShadowLogs(unittest.TestCase):
    @patch("src.prediction.unified_run_summary.log_unified_prediction_summary")
    @patch("src.utils.db.get_active_symbols")
    @patch("src.prediction.shadow_evaluation.available_unified_challengers")
    def test_challenger_shadow_logs_stats(self, mock_available, mock_symbols, _summary):
        """challenger の影の予想の回の終わりに run=challenger の行が出る（#1431）"""
        from src.prediction.shadow_evaluation import predict_with_challenger_unified

        challenger = "UnifiedStockXGBoost_challenger"
        mock_available.return_value = [challenger]
        mock_symbols.return_value = [("jp", "7203"), ("jp", "6758")]
        model = _model(0.10, ["feat_a", "feat_b", "atr_lag1"])
        with (
            patch(f"{_MOD}.load_feature_data", return_value=_features(10.0, False)),
            patch(f"{_MOD}._fetch_current_price", return_value=1000.0),
            patch(f"{_MOD}.get_service_url", return_value=None),
            patch(
                f"{_MOD}.get_cached_model",
                side_effect=lambda name: model if name == challenger else None,
            ),
            patch(f"{_MOD}.load_model_weights", return_value=[1.0]),
            patch(f"{_STATS}.logger") as mock_logger,
        ):
            rows = predict_with_challenger_unified()

        self.assertEqual(len(rows), 2)
        lines = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]
        clip_lines = [line for line in lines if line.startswith("予想のATRクリップ")]
        fill_lines = [line for line in lines if line.startswith("予想の特徴量の0埋め")]
        self.assertEqual(len(clip_lines), 1, lines)
        self.assertEqual(len(fill_lines), 1, lines)
        self.assertIn("run=challenger market=jp horizon=1", clip_lines[0])
        self.assertIn("上限=2（100.0%）", clip_lines[0])
        self.assertIn("run=challenger market=jp horizon=1", fill_lines[0])
        self.assertIn("0埋めの銘柄数=2/2", fill_lines[0])
        self.assertIn("名前=feat_b", fill_lines[0])


if __name__ == "__main__":
    unittest.main()
