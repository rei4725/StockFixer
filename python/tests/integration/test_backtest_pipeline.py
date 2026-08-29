"""
Integration Test: バックテストパイプライン End-to-End

合成 OHLCV を Postgres の stock_features へ投入し、
run_backtest_single / run_backtest_walk_forward が実データパスで
完走することを検証する。
"""

import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_MARKET = "jp"
_SYMBOL = "TESTBT"
_N_DAYS = 200
_XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None


def _make_synthetic_ohlcv(n_days: int = _N_DAYS) -> pd.DataFrame:
    """n_days営業日分の固定OHLCV DataFrame（yfinance戻り値と同形式）を生成する。"""
    rng = np.random.default_rng(42)
    last_bday = pd.Timestamp.today().normalize()
    while last_bday.weekday() >= 5:
        last_bday -= pd.Timedelta(days=1)
    dates = pd.bdate_range(end=last_bday, periods=n_days)

    close = np.cumsum(rng.normal(0, 0.5, n_days)) + 100.0
    close = np.clip(close, 50.0, 300.0)

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.uniform(-0.005, 0.005, n_days)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "Close": close,
            "Volume": rng.integers(500_000, 2_000_000, n_days).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _seed_stock_features(market: str, symbol: str) -> None:
    """合成OHLCVを特徴量生成してstock_featuresへ保存する
    （tests/e2e/conftest.py の _generate_and_save_features と同等の手順）。
    """
    from src.market_data.saver import save_raw_ohlcv
    from src.market_data.technical import add_technical_indicators, create_basic_lag_features
    from src.utils.data_path_utils import normalize_col
    from src.utils.db import upsert_stock_features

    df = _make_synthetic_ohlcv()
    save_raw_ohlcv(market, symbol, df)

    work = df.copy()
    work = add_technical_indicators(work)
    if int(work.isnull().sum().sum()) > 0:
        work = work.ffill().bfill()

    X, y = create_basic_lag_features(work, n_lags=10)
    if X is None or X.empty:
        raise RuntimeError("テスト用特徴量の生成に失敗しました")

    X.columns = [normalize_col(c) for c in X.columns]
    data = X.copy()
    data["market"] = market
    data["symbol"] = symbol
    data["market_encoded"] = 1 if market == "jp" else 0
    data["y"] = y
    upsert_stock_features(market, symbol, data)


@unittest.skipUnless(_XGBOOST_AVAILABLE, "XGBoost not available")
class TestBacktestPipelineIntegration(unittest.TestCase):
    """バックテストパイプラインの Postgres 実データパス End-to-End テスト"""

    def setUp(self):
        from src.backtest.ports import set_model_manager_factory
        from src.prediction.manager import ModelManager

        set_model_manager_factory(ModelManager)
        _seed_stock_features(_MARKET, _SYMBOL)

    def test_backtest_single_runs_without_error(self):
        """単一期間バックテストが実行可能なことを確認"""
        from src.backtest.pipeline import run_backtest_single

        result_df, metrics, price_series = run_backtest_single(
            market=_MARKET,
            symbol=_SYMBOL,
            model_type="XGBoostModel",
            model_name="TestBacktestModel",
            task_name="return_regression",
            threshold=0.0,
            source="file",
            initial_cash=1_000_000,
            fee_rate=0.001,
            slippage=0.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            position_sizing="full",
            position_fraction=0.5,
            ensemble=False,
            start_date=None,
            end_date=None,
            train_ratio=0.8,
        )

        self.assertIsNotNone(metrics, "メトリクスが None でないこと")
        self.assertIn("final_cash", metrics, "final_cash メトリクスが存在すること")
        self.assertIn("total_return", metrics, "total_return メトリクスが存在すること")
        self.assertIn("sharpe_ratio", metrics, "sharpe_ratio メトリクスが存在すること")

    def test_backtest_walk_forward_runs_without_error(self):
        """Walk-Forward バックテストが実行可能なことを確認"""
        from src.backtest.pipeline import run_backtest_walk_forward

        _, _, wf_df = run_backtest_walk_forward(
            market=_MARKET,
            symbol=_SYMBOL,
            model_type="XGBoostModel",
            model_name="TestWalkForwardModel",
            task_name="return_regression",
            threshold=0.0,
            source="file",
            n_splits=3,
            initial_cash=1_000_000,
            fee_rate=0.001,
            slippage=0.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            position_sizing="full",
            position_fraction=0.5,
            ensemble=False,
        )

        self.assertIsNotNone(wf_df, "Walk-Forward 結果が None でないこと")
        self.assertGreater(len(wf_df), 0, "Walk-Forward 結果に最低1行以上のデータがあること")

        expected_cols = ["fold", "val_start", "val_end", "total_return", "sharpe_ratio"]
        for col in expected_cols:
            self.assertIn(col, wf_df.columns, f"{col} 列が存在すること")


if __name__ == "__main__":
    unittest.main()
