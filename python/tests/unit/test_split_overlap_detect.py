"""差分の取得で DB の直近の足と突き合わせて分割を見つけ、段差の足を見張るテスト（#1386）"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.domain.types import BatchResult
from src.market_data.batch_db_write import write_batch_to_db
from src.market_data.pipeline import fetch_stock_data_with_features

_P = "src.market_data.pipeline"
_PRICE_COLS = ("Open", "High", "Low", "Close")


def _make_ohlcv(periods: int = 66) -> pd.DataFrame:
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


def _with_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Dividends"] = 0.0
    out["Stock Splits"] = 0.0
    return out


def _scale_before(df: pd.DataFrame, pos: int, factor: float) -> pd.DataFrame:
    """pos 本目より前の値段を factor 倍にする（分割前の値段の足を作る）"""
    out = df.copy()
    for col in _PRICE_COLS:
        out.iloc[:pos, out.columns.get_loc(col)] *= factor
    return out


def _yf(bars: pd.DataFrame):
    """yfinance の差し替え。start〜end の足を返す"""

    def _fetch(_market, _ticker, start, end):
        return bars.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    return _fetch


def _fetch(symbol: str = "8035"):
    return fetch_stock_data_with_features(
        market="jp",
        symbol=symbol,
        start_date="2023-01-01",
        end_date="2023-03-07",
        defer_raw_save=True,
    )


@patch(f"{_P}.fetch_pageview_features", return_value=None)
@patch(f"{_P}.fetch_news_sentiment_with_llm", return_value=None)
@patch(f"{_P}.fetch_additional_macro_features", return_value=None)
@patch(f"{_P}.fetch_cross_asset_features", return_value=None)
@patch(f"{_P}.should_fetch_fresh_data", return_value=True)
class TestOverlapDetect(unittest.TestCase):
    def test_adjusted_overlap_without_split_row_refetches_full_history(self, *_mocks):
        """取得が止まっていた間の 1:5 分割で、差分の窓に Stock Splits の行が無くても全期間を取り直す"""
        adjusted = _with_actions(_make_ohlcv())
        # DB の足は 60 本目まで。分割の権利落ちは 62 本目で、DB の足は分割前の値段（5 倍）
        db_bars = _scale_before(_make_ohlcv(), 62, 5.0).iloc[:60]
        with (
            patch(f"{_P}.get_raw_ohlcv_from_db", return_value=db_bars),
            patch(f"{_P}.get_stock_data", side_effect=_yf(adjusted)) as mock_get,
        ):
            result = _fetch()

        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_get.call_args_list[1].args[2], "2023-01-01")
        self.assertIsNotNone(result)
        _m, _s, data, raw, _q = result
        self.assertEqual(len(raw), len(adjusted))
        self.assertAlmostEqual(raw["Close"].iloc[0], adjusted["Close"].iloc[0])
        self.assertLess(data["Close_lag1"].pct_change().abs().max(), 0.1)

    def test_overlap_within_tolerance_fetches_only_diff(self, *_mocks):
        """重なる期間の比が閾値の中（配当の調整の 1% ずれ）なら、差分だけを取って差分だけを保存に回す"""
        ohlcv = _make_ohlcv()
        yf_bars = _with_actions(ohlcv)
        for col in _PRICE_COLS:
            yf_bars[col] = yf_bars[col] * 0.99
        with (
            patch(f"{_P}.get_raw_ohlcv_from_db", return_value=ohlcv.iloc[:60]),
            patch(f"{_P}.get_stock_data", side_effect=_yf(yf_bars)) as mock_get,
        ):
            result = _fetch("7203")

        mock_get.assert_called_once()
        # DB の直近 5 本（02-25〜03-01）と重ねて取る
        self.assertEqual(mock_get.call_args.args[2], "2023-02-25")
        self.assertIsNotNone(result)
        self.assertIsNotNone(result[2])
        raw = result[3]
        self.assertEqual(len(raw), 6)
        self.assertEqual(raw.index.min(), pd.Timestamp("2023-03-02"))

    def test_step_in_db_bars_is_healed_by_full_refetch(self, *_mocks):
        """DB に PR 892 より前の分割の段差が残っていれば、全期間を取り直して段差を消す"""
        adjusted = _with_actions(_make_ohlcv())
        db_bars = _scale_before(_make_ohlcv(), 30, 3.0).iloc[:60]
        with (
            patch(f"{_P}.get_raw_ohlcv_from_db", return_value=db_bars),
            patch(f"{_P}.get_stock_data", side_effect=_yf(adjusted)) as mock_get,
        ):
            result = _fetch("285A")

        self.assertEqual(mock_get.call_count, 2)
        self.assertIsNotNone(result)
        _m, _s, data, raw, quality = result
        self.assertIsNotNone(data)
        self.assertEqual(len(raw), len(adjusted))
        self.assertLess(data["Close_lag1"].pct_change().abs().max(), 0.1)
        self.assertFalse([i for i in quality.issues if i.check == "price_step"])

    def test_step_left_after_refetch_is_warned_and_excluded(self, *_mocks):
        """取り直しても前日比が極端な足が残る銘柄は、WARNING にして特徴量を作らない"""
        stepped = _with_actions(_scale_before(_make_ohlcv(), 30, 3.0))
        db_bars = stepped.drop(columns=["Dividends", "Stock Splits"]).iloc[:60]
        with (
            patch(f"{_P}.get_raw_ohlcv_from_db", return_value=db_bars),
            patch(f"{_P}.get_stock_data", side_effect=_yf(stepped)) as mock_get,
            self.assertLogs("src.market_data", level="WARNING") as logs,
        ):
            result = _fetch("5706")

        self.assertEqual(mock_get.call_count, 2)
        self.assertIsNotNone(result)
        _m, _s, data, raw, quality = result
        # 特徴量を作らない（stock_features を更新しないため、予想の上位とシグナルに出ない）
        self.assertIsNone(data)
        # 取り直した全期間の足は保存に回す
        self.assertEqual(len(raw), len(stepped))
        steps = [i for i in quality.issues if i.check == "price_step"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].level, "warning")
        self.assertIn("2023-01-31", steps[0].detail)
        self.assertTrue([m for m in logs.output if "WARNING" in m and "jp/5706" in m])


class TestStepQualityLogInBatch(unittest.TestCase):
    def test_skipped_symbol_with_step_is_recorded_in_quality_log(self):
        """特徴量をスキップした段差の銘柄も、フェーズ2で data_quality_log に記録し、特徴量は保存しない"""
        from src.market_data.pipeline import _notify_quality_issues
        from src.market_data.quality_check import QualityCheckResult, QualityIssue

        quality = QualityCheckResult(
            market="jp",
            symbol="5706",
            passed=True,
            issues=[QualityIssue(check="price_step", level="warning", detail="段差")],
        )
        batch = BatchResult(
            succeeded=[],
            failed=[],
            skipped=[
                {
                    "market": "jp",
                    "symbol": "5706",
                    "status": "skip",
                    "raw_data": _make_ohlcv(),
                    "quality_result": quality,
                }
            ],
        )
        with (
            patch("src.utils.db.quality_log.insert_quality_log") as mock_log,
            patch(f"{_P}.save_features_to_db") as mock_save_features,
        ):
            result = write_batch_to_db(
                batch,
                lambda *_a: None,
                mock_save_features,
                None,
                record_quality_issues=_notify_quality_issues,
            )

        mock_save_features.assert_not_called()
        mock_log.assert_called_once()
        market, symbol, issues = mock_log.call_args.args
        self.assertEqual((market, symbol), ("jp", "5706"))
        self.assertEqual(issues[0]["check"], "price_step")
        self.assertEqual(len(result.skipped), 1)


if __name__ == "__main__":
    unittest.main()
