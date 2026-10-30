"""段差の足で特徴量を作らなかった銘柄の stock_features の前の行を消すテスト（#1399）"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.domain.types import BatchResult
from src.market_data.batch_db_write import write_batch_to_db
from src.market_data.quality_check import QualityCheckResult, QualityIssue

_P = "src.market_data.pipeline"


def _make_ohlcv(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, periods),
            "High": np.linspace(101, 121, periods),
            "Low": np.linspace(99, 119, periods),
            "Close": np.linspace(100, 120, periods),
            "Volume": np.full(periods, 3000),
        },
        index=dates,
    )


def _step_quality(symbol: str) -> QualityCheckResult:
    return QualityCheckResult(
        market="jp",
        symbol=symbol,
        passed=True,
        issues=[QualityIssue(check="price_step", level="warning", detail="段差")],
    )


def _skip(symbol: str, quality=None, raw=True) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "status": "skip",
        "raw_data": _make_ohlcv() if raw else None,
        "quality_result": quality,
    }


class TestWriteBatchDropsStepFeatures(unittest.TestCase):
    def test_price_step_skip_deletes_stock_features(self):
        """price_step の品質つきで skip になった銘柄は stock_features の行を消す"""
        batch = BatchResult(succeeded=[], failed=[], skipped=[_skip("5706", _step_quality("5706"))])
        delete = MagicMock()
        result = write_batch_to_db(
            batch,
            MagicMock(),
            MagicMock(),
            None,
            record_quality_issues=MagicMock(),
            delete_features=delete,
        )
        delete.assert_called_once_with("jp", "5706")
        self.assertEqual([r["symbol"] for r in result.skipped], ["5706"])
        self.assertEqual(result.failed, [])

    def test_skip_without_price_step_keeps_stock_features(self):
        """止まった足・行の不足など price_step の無い skip では消さない"""
        other = QualityCheckResult(
            market="jp",
            symbol="7203",
            passed=True,
            issues=[QualityIssue(check="zero_volume", level="warning", detail="出来高0")],
        )
        batch = BatchResult(
            succeeded=[],
            failed=[],
            skipped=[_skip("1301", raw=False), _skip("1332"), _skip("7203", other)],
        )
        delete = MagicMock()
        result = write_batch_to_db(
            batch,
            MagicMock(),
            MagicMock(),
            None,
            record_quality_issues=MagicMock(),
            delete_features=delete,
        )
        delete.assert_not_called()
        self.assertEqual(len(result.skipped), 3)

    def test_delete_failure_counts_failed_and_others_continue(self):
        """消すのが落ちた銘柄は failed に数え、ほかの銘柄の保存は続ける"""
        data = pd.DataFrame({"y": [1.0]})
        batch = BatchResult(
            succeeded=[
                {"market": "jp", "symbol": "9984", "data": ("jp", "9984", data, None, None)}
            ],
            failed=[],
            skipped=[_skip("5706", _step_quality("5706")), _skip("8035", _step_quality("8035"))],
        )

        def _delete(market, symbol):
            if symbol == "5706":
                raise RuntimeError("DB 断")

        delete = MagicMock(side_effect=_delete)
        save_raw = MagicMock()
        save_features = MagicMock()
        result = write_batch_to_db(
            batch,
            save_raw,
            save_features,
            None,
            record_quality_issues=MagicMock(),
            delete_features=delete,
        )
        self.assertEqual(delete.call_count, 2)
        self.assertEqual([f.symbol for f in result.failed], ["5706"])
        self.assertIn("stock_features", result.failed[0].error)
        self.assertEqual([r["symbol"] for r in result.skipped], ["8035"])
        self.assertEqual([r["symbol"] for r in result.succeeded], ["9984"])
        # 足の保存は消すのが落ちた銘柄も含めて続ける
        self.assertEqual(save_raw.call_count, 2)
        save_features.assert_called_once()


class TestRunBatchPipelinePassesDelete(unittest.TestCase):
    def test_run_batch_pipeline_passes_delete_stock_features(self):
        from src.market_data import pipeline

        skip = ("jp", "5706", None, _make_ohlcv(), _step_quality("5706"))
        with (
            patch(f"{_P}.fetch_stock_data_with_features", return_value=skip),
            patch(f"{_P}.write_batch_to_db") as mock_write,
        ):
            pipeline.run_batch_pipeline([{"market": "jp", "symbol": "5706"}])
        self.assertIs(
            mock_write.call_args.kwargs["delete_features"], pipeline.delete_stock_features
        )


class TestSequentialDropsStepFeatures(unittest.TestCase):
    def test_price_step_skip_deletes_stock_features(self):
        from src.market_data.pipeline import save_stock_data_with_features

        result = ("jp", "5706", None, None, _step_quality("5706"))
        with (
            patch(f"{_P}.fetch_stock_data_with_features", return_value=result),
            patch(f"{_P}._notify_quality_issues"),
            patch(f"{_P}.delete_stock_features") as mock_delete,
            patch(f"{_P}.upsert_stock_features") as mock_upsert,
        ):
            save_stock_data_with_features("jp", "5706")
        mock_delete.assert_called_once_with("jp", "5706")
        mock_upsert.assert_not_called()

    def test_skip_without_price_step_keeps_stock_features(self):
        from src.market_data.pipeline import save_stock_data_with_features

        result = ("jp", "1332", None, _make_ohlcv(), None)
        with (
            patch(f"{_P}.fetch_stock_data_with_features", return_value=result),
            patch(f"{_P}.delete_stock_features") as mock_delete,
        ):
            save_stock_data_with_features("jp", "1332")
        mock_delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
