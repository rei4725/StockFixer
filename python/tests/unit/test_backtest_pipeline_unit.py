"""
Unit Test: バックテストパイプライン ロジック

load_features, _build_task, メトリクス計算など、
パイプラインの計算ロジックのみをテスト（外部依存なし）。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    _extract_trade_pnl,
    _max_drawdown,
    _sharpe_per_trade,
    compute_metrics,
)
from src.backtest.task import ReturnRegressionTask


class TestBacktestTaskLocalLogic:
    """BacktestTask のローカルロジック"""

    def test_return_regression_task_label_generation(self):
        """ReturnRegressionTask のラベル生成"""
        task = ReturnRegressionTask()

        # サンプルデータ
        df = pd.DataFrame(
            {"Close": [100, 105, 110, 95, 100]},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )

        labels = task.make_labels(df)

        # ラベルが Series として返される
        assert isinstance(labels, pd.Series)
        # 翌日変化率のため最終行は NaN だが、インデックス長は入力と同じ
        assert len(labels) == len(df)

    def test_signal_generation_from_prediction(self):
        """予測値からシグナルを生成"""
        task = ReturnRegressionTask()

        # サンプル予測値
        pred = pd.Series([0.02, -0.01, 0.015, 0.005, -0.02])

        # デフォルト閾値（0.0）
        signal = task.make_signal(pred)

        # Buy/Sell/Hold シグナルが生成される
        assert all(signal.isin([1, -1, 0]))
        # 正の予測値 → Buy (1)
        assert signal.iloc[0] == 1
        # 負の予測値 → Sell (-1)
        assert signal.iloc[1] == -1


class TestComputeMetrics:
    """メトリクス計算（dtype エラー修正の検証）"""

    def test_metrics_from_empty_trade_log(self):
        """取引なし時の メトリクス"""
        metrics = compute_metrics(pd.DataFrame(), initial_cash=1_000_000)

        assert metrics["num_trades"] == 0
        assert metrics["total_return"] == 0.0
        assert metrics["final_cash"] == 1_000_000

    def test_metrics_from_single_win_trade(self):
        """単一勝利トレード"""
        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "action": ["buy", "sell"],
                "price": [100, 110],
                "qty": [10, 10],
                "cash": [990_000, 1_090_000],
            }
        )

        metrics = compute_metrics(trade_log, initial_cash=1_000_000)

        assert metrics["num_trades"] == 1
        assert metrics["win_rate"] == 1.0
        assert metrics["total_return"] > 0.0

    def test_metrics_from_multiple_mixed_trades(self):
        """複数のWin/Loss混在トレード"""
        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "action": ["buy", "sell", "buy", "sell", "final_sell"],
                "price": [100, 110, 105, 100, 100],
                "qty": [10, 10, 10, 10, 10],
                "cash": [990_000, 1_090_000, 1_000_000, 1_090_000, 1_090_000],
            }
        )

        metrics = compute_metrics(trade_log, initial_cash=1_000_000)

        assert metrics["num_trades"] == 2
        assert 0.0 < metrics["win_rate"] <= 1.0

    def test_profit_factor_calculation_with_none(self):
        """profit_factor の None 値を含む計算"""
        # profit_factor が None の場合がある（取引がない場合など）
        metrics1 = compute_metrics(pd.DataFrame(), initial_cash=1_000_000)
        # None が返される
        assert metrics1["profit_factor"] is None

        # 複数メトリクスを混合
        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "action": ["buy", "sell"],
                "price": [100, 110],
                "qty": [10, 10],
                "cash": [990_000, 1_090_000],
            }
        )
        metrics2 = compute_metrics(trade_log, initial_cash=1_000_000)
        # 勝ちトレードのみの場合は損失ゼロのため None が返る
        assert metrics2["profit_factor"] is None


class TestMetricsHelpers:
    """メトリクスヘルパー関数"""

    def test_extract_trade_pnl(self):
        """Buy-Sell ペアから損益を抽出"""
        trade_log = pd.DataFrame(
            {
                "action": ["buy", "sell", "buy", "sell"],
                "price": [100, 110, 105, 100],
                "qty": [10, 10, 10, 10],
            }
        )

        wins, losses = _extract_trade_pnl(trade_log)[:2]

        # Trade 1: 100→110 (勝ち)
        # Trade 2: 105→100 (負け)
        assert len(wins) == 1
        assert len(losses) == 1
        assert wins[0] > 0
        assert losses[0] < 0

    def test_max_drawdown_calculation(self):
        """最大ドローダウン計算"""
        equity = pd.Series([1_000_000, 900_000, 950_000, 850_000, 900_000])

        dd = _max_drawdown(equity)

        # ピークの 1_000_000 からボトムの 850_000 へ
        # ドローダウン率 = (850_000 - 1_000_000) / 1_000_000 = -0.15
        assert dd < 0
        assert -0.2 < dd < 0

    def test_sharpe_ratio_calculation(self):
        """シャープレシオ計算"""
        # サンプルリターン列（±5%）
        returns = [0.05, -0.03, 0.04, -0.02, 0.06]

        sharpe = _sharpe_per_trade(returns)

        # シャープレシオは計算可能（数値）
        assert isinstance(sharpe, (int, float))


class TestProbitFactorNoneHandling:
    """profit_factor None 値処理（dtype エラー修正確認）"""

    def test_metrics_dataframe_none_to_nan_conversion(self):
        """メトリクス DataFrame での None → NaN 変換"""
        # profit_factor が None を含む DataFrame
        metrics_list = [
            {"total_return": 0.01, "sharpe_ratio": 0.5, "profit_factor": None},
            {"total_return": -0.02, "sharpe_ratio": -0.3, "profit_factor": 2.5},
            {"total_return": 0.005, "sharpe_ratio": 0.2, "profit_factor": None},
        ]
        metrics_df = pd.DataFrame(metrics_list)

        # None → NaN 変換
        for col in ["total_return", "sharpe_ratio", "profit_factor"]:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        # 平均計算が成功する（dtype エラーなし）
        means = metrics_df.mean()

        assert not means.isna().all()
        # profit_factor の NaN は平均から除外されている
        assert np.isnan(means["profit_factor"]) or isinstance(means["profit_factor"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# 追加テスト: load_features / run_backtest_single / print_backtest_metrics
# ---------------------------------------------------------------------------


class TestLoadFeaturesRawSource:
    """load_features(source='raw') のテスト"""

    @patch("src.market_data.loader.get_raw_ohlcv_from_db")
    @patch("src.market_data.technical.add_technical_indicators")
    @patch("src.market_data.technical.create_basic_lag_features")
    def test_load_features_raw_returns_dataframe(self, mock_lag, mock_ti, mock_raw):
        """source='raw' で DataFrame が返ること"""
        n = 35
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        raw_df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [105.0] * n,
                "Low": [95.0] * n,
                "Close": [102.0] * n,
                "Volume": [1000.0] * n,
            },
            index=dates,
        )
        mock_raw.return_value = raw_df
        mock_ti.return_value = raw_df

        X_df = pd.DataFrame({"feat1": [100.0] * 25}, index=dates[:25])
        y_series = pd.Series([0.01] * 25, index=dates[:25])
        mock_lag.return_value = (X_df, y_series)

        from src.backtest.pipeline import load_features

        result = load_features("jp", "7203", "raw")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_raw.assert_called_once_with("jp", "7203", None, None)

    @patch("src.market_data.loader.get_raw_ohlcv_from_db")
    def test_load_features_raw_exits_on_empty(self, mock_raw):
        """空データの場合は SystemExit が発生すること"""
        mock_raw.return_value = pd.DataFrame()

        from src.backtest.pipeline import load_features

        with pytest.raises(SystemExit):
            load_features("jp", "7203", "raw")


class TestLoadFeaturesFileSource:
    """load_features(source='file') のテスト"""

    @patch("src.utils.db.load_stock_features")
    def test_load_features_file_with_date_column(self, mock_load):
        """date 列がある場合は DatetimeIndex に変換されること"""
        n = 20
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="B").astype(str),
                "Close_lag1": [100.0 + i for i in range(n)],
                "feat1": [1.0] * n,
                "y": [0.01] * n,
            }
        )
        mock_load.return_value = df

        from src.backtest.pipeline import load_features

        result = load_features("jp", "7203", "file")

        assert isinstance(result.index, pd.DatetimeIndex)

    @patch("src.utils.db.load_stock_features")
    def test_load_features_file_close_from_lag1(self, mock_load):
        """Close 列がなく Close_lag1 がある場合は Close 列が追加されること"""
        n = 20
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="B").astype(str),
                "Close_lag1": [100.0 + i for i in range(n)],
                "y": [0.01] * n,
            }
        )
        mock_load.return_value = df

        from src.backtest.pipeline import load_features

        result = load_features("jp", "7203", "file")

        assert "Close" in result.columns

    @patch("src.utils.db.load_stock_features")
    def test_load_features_file_exits_on_empty(self, mock_load):
        """空データの場合は SystemExit が発生すること"""
        mock_load.return_value = pd.DataFrame()

        from src.backtest.pipeline import load_features

        with pytest.raises(SystemExit):
            load_features("jp", "7203", "file")


class TestRunBacktestSingle:
    """run_backtest_single() のテスト"""

    @patch("src.backtest.pipeline.load_features")
    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.backtest.backtester.Backtester")
    @patch("src.trading.signal_generator.SignalGenerator")
    def test_run_backtest_single_returns_metrics(self, mock_sg, mock_bt, mock_get_mm, mock_lf):
        """基本フローでメトリクス辞書が返ること"""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        mock_df = pd.DataFrame(
            {
                "feat1": np.random.uniform(0, 1, n),
                "y": np.random.uniform(-0.05, 0.05, n),
                "Close": [100.0 + i for i in range(n)],
            },
            index=dates,
        )
        mock_lf.return_value = mock_df

        # ModelManager mock
        mock_mm = mock_get_mm.return_value
        n_test = n - int(n * 0.8)  # 4
        mock_mm.predict_with_model.return_value = [0.01] * n_test

        # Backtester mock
        mock_bt_instance = mock_bt.return_value
        mock_bt_instance.simulate_trading.return_value = (
            pd.DataFrame(),
            {"total_return": 0.01, "sharpe_ratio": 0.5, "num_trades": 2},
        )

        from src.backtest.pipeline import run_backtest_single

        result_df, metrics, price = run_backtest_single("jp", "7203")

        assert isinstance(metrics, dict)
        mock_mm.create_model.assert_called_once()
        mock_mm.train_model.assert_called_once()
        mock_bt_instance.simulate_trading.assert_called_once()

    @patch("src.backtest.pipeline.load_features")
    @patch("src.backtest.pipeline._ensemble_predict")
    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.backtest.backtester.Backtester")
    @patch("src.trading.signal_generator.SignalGenerator")
    def test_run_backtest_single_ensemble_mode(
        self, mock_sg, mock_bt, mock_get_mm, mock_ep, mock_lf
    ):
        """ensemble=True の場合は _ensemble_predict が呼ばれること"""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        mock_df = pd.DataFrame(
            {
                "feat1": np.random.uniform(0, 1, n),
                "y": np.random.uniform(-0.05, 0.05, n),
                "Close": [100.0 + i for i in range(n)],
            },
            index=dates,
        )
        mock_lf.return_value = mock_df

        n_test = n - int(n * 0.8)  # 4
        test_index = dates[int(n * 0.8) :]
        mock_ep.return_value = pd.Series([0.01] * n_test, index=test_index)

        mock_bt_instance = mock_bt.return_value
        mock_bt_instance.simulate_trading.return_value = (
            pd.DataFrame(),
            {"total_return": 0.02},
        )

        from src.backtest.pipeline import run_backtest_single

        run_backtest_single("jp", "7203", ensemble=True)

        mock_ep.assert_called_once()

    @patch("src.backtest.pipeline.load_features")
    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.backtest.backtester.Backtester")
    @patch("src.trading.signal_generator.SignalGenerator")
    def test_exclude_sentiment_removes_sentiment_columns(
        self, mock_sg, mock_bt, mock_get_mm, mock_lf
    ):
        """exclude_sentiment=True のとき sentiment_* / news_count 列が学習から除外される"""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        mock_df = pd.DataFrame(
            {
                "feat1": np.random.uniform(0, 1, n),
                "sentiment_score": np.random.uniform(-1, 1, n),
                "sentiment_ma5": np.random.uniform(-1, 1, n),
                "sentiment_momentum": np.random.uniform(-0.1, 0.1, n),
                "news_count": np.random.randint(0, 10, n).astype(float),
                "y": np.random.uniform(-0.05, 0.05, n),
                "Close": [100.0 + i for i in range(n)],
            },
            index=dates,
        )
        mock_lf.return_value = mock_df

        mock_mm = mock_get_mm.return_value
        n_test = n - int(n * 0.8)
        mock_mm.predict_with_model.return_value = [0.01] * n_test

        mock_bt_instance = mock_bt.return_value
        mock_bt_instance.simulate_trading.return_value = (pd.DataFrame(), {"total_return": 0.01})

        from src.backtest.pipeline import run_backtest_single

        run_backtest_single("jp", "7203", exclude_sentiment=True)

        train_call = mock_mm.train_model.call_args
        X_train_used = train_call[0][1]  # 第2引数が X_train
        sentiment_cols = [
            c for c in X_train_used.columns if c.startswith("sentiment_") or c == "news_count"
        ]
        assert sentiment_cols == [], f"センチメント列が除外されていない: {sentiment_cols}"

    @patch("src.backtest.pipeline.load_features")
    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.backtest.backtester.Backtester")
    @patch("src.trading.signal_generator.SignalGenerator")
    def test_include_sentiment_keeps_sentiment_columns(
        self, mock_sg, mock_bt, mock_get_mm, mock_lf
    ):
        """exclude_sentiment=False（デフォルト）では sentiment_* 列が学習に含まれる"""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        mock_df = pd.DataFrame(
            {
                "feat1": np.random.uniform(0, 1, n),
                "sentiment_score": np.random.uniform(-1, 1, n),
                "news_count": np.random.randint(0, 10, n).astype(float),
                "y": np.random.uniform(-0.05, 0.05, n),
                "Close": [100.0 + i for i in range(n)],
            },
            index=dates,
        )
        mock_lf.return_value = mock_df

        mock_mm = mock_get_mm.return_value
        n_test = n - int(n * 0.8)
        mock_mm.predict_with_model.return_value = [0.01] * n_test

        mock_bt_instance = mock_bt.return_value
        mock_bt_instance.simulate_trading.return_value = (pd.DataFrame(), {"total_return": 0.01})

        from src.backtest.pipeline import run_backtest_single

        run_backtest_single("jp", "7203", exclude_sentiment=False)

        train_call = mock_mm.train_model.call_args
        X_train_used = train_call[0][1]
        assert "sentiment_score" in X_train_used.columns
        assert "news_count" in X_train_used.columns


class TestPrintBacktestMetrics:
    """print_backtest_metrics() のテスト"""

    def test_print_backtest_metrics_no_error(self):
        """メトリクス辞書を渡しても例外が発生しないこと"""
        from src.backtest.pipeline import print_backtest_metrics

        metrics = {
            "total_return": 0.05,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "num_trades": 10,
            "win_rate": 0.6,
            "profit_factor": 2.0,
            "final_cash": 1_050_000,
        }
        # 例外が発生しないことを確認（stdout に出力されるだけ）
        print_backtest_metrics(metrics, label="jp/7203")

    def test_print_backtest_metrics_empty_does_nothing(self):
        """空辞書を渡しても例外が発生しないこと"""
        from src.backtest.pipeline import print_backtest_metrics

        print_backtest_metrics({})

    def test_print_backtest_metrics_with_gross(self, capsys):
        """gross フィールドがある場合も正常に出力されること"""
        from src.backtest.pipeline import print_backtest_metrics

        metrics = {
            "total_return": 0.05,
            "final_cash": 1_050_000,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "num_trades": 5,
            "win_rate": 0.6,
            "profit_factor": 2.0,
            "gross_total_return": 0.06,
            "gross_final_cash": 1_060_000,
            "gross_sharpe_ratio": 1.3,
            "gross_max_drawdown": -0.07,
            "cost_impact_cash": -10_000,
            "cost_impact_return": -0.01,
        }
        print_backtest_metrics(metrics, label="test")
        captured = capsys.readouterr()
        assert "GROSS" in captured.out


# ---------------------------------------------------------------------------
# 追加テスト: _ensemble_predict / save_backtest_results
# ---------------------------------------------------------------------------


import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402


class TestEnsemblePredict(unittest.TestCase):
    """_ensemble_predict のテスト"""

    def test_returns_averaged_predictions(self):
        """XGBoost と LightGBM の平均予測値が返ること"""
        from src.backtest.pipeline import _ensemble_predict

        mock_mm = MagicMock()
        mock_mm.predict_with_model.side_effect = [
            np.array([0.01, 0.02, 0.03]),  # XGBoost
            np.array([0.02, 0.01, 0.04]),  # LightGBM
        ]

        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        y_train = pd.Series([0.01, 0.02, 0.03])
        X_test = pd.DataFrame({"f1": [4.0, 5.0, 6.0]})

        result = _ensemble_predict(mock_mm, X_train, y_train, X_test, "BacktestModel")

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        np.testing.assert_allclose(result.values, [0.015, 0.015, 0.035])

    def test_uses_test_index(self):
        """返り値の index が X_test の index と一致すること"""
        from src.backtest.pipeline import _ensemble_predict

        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_mm = MagicMock()
        mock_mm.predict_with_model.side_effect = [
            np.array([0.01, 0.02, 0.03]),
            np.array([0.03, 0.02, 0.01]),
        ]

        X_train = pd.DataFrame({"f1": [1.0, 2.0]})
        y_train = pd.Series([0.01, 0.02])
        X_test = pd.DataFrame({"f1": [3.0, 4.0, 5.0]}, index=dates)

        result = _ensemble_predict(mock_mm, X_train, y_train, X_test, "Model")

        pd.testing.assert_index_equal(result.index, dates)

    def test_calls_create_model_for_both_types(self):
        """XGBoost と LightGBM の create_model が呼ばれること"""
        from src.backtest.pipeline import _ensemble_predict

        mock_mm = MagicMock()
        mock_mm.predict_with_model.side_effect = [
            np.array([0.01]),
            np.array([0.02]),
        ]

        X_train = pd.DataFrame({"f1": [1.0]})
        y_train = pd.Series([0.01])
        X_test = pd.DataFrame({"f1": [2.0]})

        _ensemble_predict(mock_mm, X_train, y_train, X_test, "Base")

        calls = [str(c) for c in mock_mm.create_model.call_args_list]
        assert any("XGBoostModel" in c for c in calls)
        assert any("LightGBMModel" in c for c in calls)


class TestSaveBacktestResults(unittest.TestCase):
    """save_backtest_results のテスト"""

    def test_saves_trade_log_csv(self):
        """単一期間バックテストの取引ログが CSV として保存されること"""
        from src.backtest.pipeline import save_backtest_results

        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3),
                "action": ["buy", "sell", "buy"],
                "price": [100.0, 110.0, 105.0],
                "qty": [10, 10, 10],
                "cash": [990000.0, 1090000.0, 980000.0],
            }
        )
        metrics = {"total_return": 0.05, "sharpe_ratio": 1.2}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.utils.data_path_utils.get_results_dir", return_value=tmp_dir):
                save_backtest_results(trade_log, metrics, None, "jp", "7203", "return_regression")

            out_dir = Path(tmp_dir) / "backtest" / "jp_7203"
            csv_files = list(out_dir.glob("*.csv"))
            self.assertGreaterEqual(len(csv_files), 1)

    def test_saves_walk_forward_csv(self):
        """Walk-Forward 結果が CSV として保存されること"""
        from src.backtest.pipeline import save_backtest_results

        wf_df = pd.DataFrame(
            {
                "split": [0, 1, 2],
                "total_return": [0.01, 0.02, -0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.utils.data_path_utils.get_results_dir", return_value=tmp_dir):
                save_backtest_results(None, None, wf_df, "us", "AAPL", "return_regression")

            out_dir = Path(tmp_dir) / "backtest" / "us_AAPL"
            csv_files = list(out_dir.glob("wf_*.csv"))
            self.assertGreaterEqual(len(csv_files), 1)

    def test_does_nothing_when_all_none(self):
        """result_df, wf_df ともに None の場合は CSV が作成されないこと"""
        from src.backtest.pipeline import save_backtest_results

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.utils.data_path_utils.get_results_dir", return_value=tmp_dir):
                save_backtest_results(None, None, None, "jp", "7203", "return_regression")

            out_dir = Path(tmp_dir) / "backtest" / "jp_7203"
            csv_files = list(out_dir.glob("*.csv")) if out_dir.exists() else []
            self.assertEqual(len(csv_files), 0)


# ---------------------------------------------------------------------------
# 追加テスト: load_features(source='api')
# ---------------------------------------------------------------------------


class TestLoadFeaturesApiSource(unittest.TestCase):
    """load_features(source='api') のテスト"""

    @patch("src.utils.data_path_utils.get_ticker")
    @patch("src.market_data.technical.create_basic_lag_features")
    @patch("src.market_data.technical.add_technical_indicators")
    @patch("src.market_data.loader.get_stock_data")
    def test_api_source_calls_get_stock_data(self, mock_yf, mock_ti, mock_lag, mock_ticker):
        """source='api' では get_stock_data が呼ばれること"""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Open": np.random.rand(n) * 100 + 100,
                "High": np.random.rand(n) * 100 + 110,
                "Low": np.random.rand(n) * 100 + 90,
                "Close": np.random.rand(n) * 100 + 100,
                "Volume": np.random.randint(1000, 9999, n).astype(float),
            },
            index=dates,
        )
        mock_ticker.return_value = "7203.T"
        mock_yf.return_value = df
        mock_ti.return_value = df

        feat_df = pd.DataFrame(
            {"close_lag1": np.random.randn(n - 5)},
            index=dates[: n - 5],
        )
        y = pd.Series(np.random.randn(n - 5), index=dates[: n - 5])
        mock_lag.return_value = (feat_df, y)

        from src.backtest.pipeline import load_features

        result = load_features("jp", "7203", "api")

        mock_yf.assert_called()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    @patch("src.utils.data_path_utils.get_ticker")
    @patch("src.market_data.loader.get_stock_data")
    def test_api_source_exits_on_empty_data(self, mock_yf, mock_ticker):
        """空データの場合 SystemExit が発生すること"""
        mock_ticker.return_value = "7203.T"
        mock_yf.return_value = pd.DataFrame()

        from src.backtest.pipeline import load_features

        with pytest.raises(SystemExit):
            load_features("jp", "7203", "api")

    @patch("src.utils.data_path_utils.get_ticker")
    @patch("src.market_data.technical.create_basic_lag_features")
    @patch("src.market_data.technical.add_technical_indicators")
    @patch("src.market_data.loader.get_stock_data")
    def test_api_source_result_has_close_column(self, mock_yf, mock_ti, mock_lag, mock_ticker):
        """source='api' の結果に Close 列が含まれること"""
        n = 40
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [105.0] * n,
                "Low": [95.0] * n,
                "Close": [102.0] * n,
                "Volume": [1000.0] * n,
            },
            index=dates,
        )
        mock_ticker.return_value = "7203.T"
        mock_yf.return_value = df
        mock_ti.return_value = df

        feat_df = pd.DataFrame(
            {"close_lag1": np.random.randn(n - 5)},
            index=dates[: n - 5],
        )
        y = pd.Series(np.random.randn(n - 5), index=dates[: n - 5])
        mock_lag.return_value = (feat_df, y)

        from src.backtest.pipeline import load_features

        result = load_features("jp", "7203", "api")

        self.assertIn("Close", result.columns)


# ---------------------------------------------------------------------------
# 追加テスト: print_backtest_metrics エッジケース / save_backtest_results 追加
# / run_backtest_walk_forward
# ---------------------------------------------------------------------------


class TestPrintBacktestMetricsAdditional:
    """print_backtest_metrics() のエッジケース（未カバー行 481-484, 498-506, 518）"""

    def test_none_win_rate_and_drawdown_no_error(self):
        """win_rate, max_drawdown が None でも例外が発生しないこと"""
        from src.backtest.pipeline import print_backtest_metrics

        metrics = {
            "total_return": 0.01,
            "final_cash": 1_010_000,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "num_trades": 5,
            "win_rate": None,
            "profit_factor": None,
        }
        print_backtest_metrics(metrics, label="test_none_values")

    def test_avg_position_fraction_is_printed(self, capsys):
        """avg_position_fraction フィールドがある場合に出力されること"""
        from src.backtest.pipeline import print_backtest_metrics

        metrics = {
            "total_return": 0.05,
            "final_cash": 1_050_000,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "num_trades": 10,
            "win_rate": 0.6,
            "profit_factor": 2.0,
            "avg_position_fraction": 0.3,
            "max_position_fraction": 0.8,
            "avg_position_value": 300_000.0,
            "atr_fallback_trades": 2,
        }
        print_backtest_metrics(metrics, label="atr_test")
        captured = capsys.readouterr()
        assert "avg_position_fraction" in captured.out

    def test_benchmark_alpha_is_printed(self, capsys):
        """benchmark データがある場合にアルファ情報が出力されること"""
        from src.backtest.pipeline import print_backtest_metrics

        metrics = {
            "total_return": 0.10,
            "final_cash": 1_100_000,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "num_trades": 5,
            "win_rate": 0.6,
            "profit_factor": 2.0,
        }
        benchmark = {
            "total_return": 0.07,
            "ticker": "N225",
            "start": "2024-01-01",
            "end": "2024-12-31",
        }
        print_backtest_metrics(metrics, label="benchmark_test", benchmark=benchmark)
        captured = capsys.readouterr()
        assert "N225" in captured.out


class TestSaveBacktestResultsAdditional(unittest.TestCase):
    """save_backtest_results の追加テスト"""

    def test_auto_creates_output_directory(self):
        """保存先ディレクトリが存在しなくても自動作成されること"""
        from src.backtest.pipeline import save_backtest_results

        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=2),
                "action": ["buy", "sell"],
                "price": [100.0, 110.0],
                "qty": [10, 10],
                "cash": [990000.0, 1090000.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.utils.data_path_utils.get_results_dir", return_value=tmp_dir):
                save_backtest_results(trade_log, {}, None, "us", "TSLA", "return_regression")

            out_dir = Path(tmp_dir) / "backtest" / "us_TSLA"
            self.assertTrue(out_dir.exists())

    def test_empty_result_df_does_not_create_csv(self):
        """空 DataFrame を渡した場合は CSV が作成されないこと"""
        from src.backtest.pipeline import save_backtest_results

        empty_df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.utils.data_path_utils.get_results_dir", return_value=tmp_dir):
                save_backtest_results(empty_df, {}, None, "jp", "7203", "return_regression")

            out_dir = Path(tmp_dir) / "backtest" / "jp_7203"
            csv_files = list(out_dir.glob("trades_*.csv")) if out_dir.exists() else []
            self.assertEqual(len(csv_files), 0)


class TestRunBacktestWalkForward(unittest.TestCase):
    """run_backtest_walk_forward のテスト（未カバー行 382-415）"""

    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.trading.signal_generator.SignalGenerator")
    @patch("src.backtest.walk_forward.WalkForwardValidator")
    def test_returns_walk_forward_results(self, mock_wfv_cls, mock_sg, mock_get_mm):
        """WalkForwardValidator の実行結果が (None, None, wf_df) として返ること"""
        from src.backtest.pipeline import run_backtest_walk_forward

        expected_df = pd.DataFrame(
            {
                "split": [0, 1, 2],
                "total_return": [0.01, 0.02, -0.01],
            }
        )
        mock_wfv_cls.return_value.run.return_value = expected_df

        result_df, metrics, wf_df = run_backtest_walk_forward("jp", "7203", n_splits=3)

        self.assertIsNone(result_df)
        self.assertIsNone(metrics)
        pd.testing.assert_frame_equal(wf_df, expected_df)
        mock_wfv_cls.return_value.run.assert_called_once()

    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.trading.signal_generator.SignalGenerator")
    @patch("src.backtest.walk_forward.WalkForwardValidator")
    def test_ensemble_skips_create_model(self, mock_wfv_cls, mock_sg, mock_get_mm):
        """ensemble=True の場合は create_model が呼ばれないこと"""
        from src.backtest.pipeline import run_backtest_walk_forward

        mock_wfv_cls.return_value.run.return_value = pd.DataFrame({"split": [0]})

        run_backtest_walk_forward("jp", "7203", ensemble=True)

        mock_get_mm.return_value.create_model.assert_not_called()

    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.trading.signal_generator.SignalGenerator")
    @patch("src.backtest.walk_forward.WalkForwardValidator")
    def test_non_ensemble_calls_create_model(self, mock_wfv_cls, mock_sg, mock_get_mm):
        """ensemble=False の場合は create_model が呼ばれること"""
        from src.backtest.pipeline import run_backtest_walk_forward

        mock_wfv_cls.return_value.run.return_value = pd.DataFrame({"split": [0]})

        run_backtest_walk_forward("jp", "7203", ensemble=False)

        mock_get_mm.return_value.create_model.assert_called_once()

    @patch("src.backtest.pipeline.get_model_manager")
    @patch("src.trading.signal_generator.SignalGenerator")
    @patch("src.backtest.walk_forward.WalkForwardValidator")
    def test_lightgbm_model_type(self, mock_wfv_cls, mock_sg, mock_get_mm):
        """LightGBMModel 指定でも正常動作すること"""
        from src.backtest.pipeline import run_backtest_walk_forward

        mock_wfv_cls.return_value.run.return_value = pd.DataFrame({"split": [0]})

        result_df, metrics, wf_df = run_backtest_walk_forward(
            "us", "AAPL", model_type="LightGBMModel", n_splits=2
        )

        self.assertIsNone(result_df)
        self.assertIsNone(metrics)
        mock_get_mm.return_value.create_model.assert_called_once_with(
            "LightGBMModel", "BacktestLightGBMModel"
        )
