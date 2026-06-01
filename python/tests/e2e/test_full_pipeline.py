"""
Cat3: 1銘柄フル結合テスト

合成 OHLCV → market_data_raw → stock_features → model → prediction_results
の全パイプラインが「処理落ちなしに完走すること」と
「各 DB テーブル / ファイルに期待どおりデータが書き込まれること」を検証する。

外部依存:
    - yfinance: モック（predict_single_stock の get_stock_data を差し替え）
    - Discord: conftest.e2e_db_env 内でモック済み

実行方法:
    python -m pytest tests/e2e/test_full_pipeline.py -v --timeout=300
"""

import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.timeout(300)


class TestRawOHLCV:
    """Step 1: market_data_raw への書き込み確認"""

    def test_raw_ohlcv_rows_inserted(self, e2e_db_env):
        """合成 OHLCV が market_data_raw へ保存されていること。"""
        from src.utils.db import load_raw_ohlcv

        market = e2e_db_env["market"]
        symbol = e2e_db_env["symbol"]
        df = load_raw_ohlcv(market, symbol)

        assert df is not None, "load_raw_ohlcv が None を返した"
        assert not df.empty, "market_data_raw に行が存在しない"
        assert len(df) >= 100, f"想定より行数が少ない: {len(df)}"

    def test_raw_ohlcv_no_null_close(self, e2e_db_env):
        """Close 列に NaN が存在しないこと。"""
        from src.utils.db import load_raw_ohlcv

        df = load_raw_ohlcv(e2e_db_env["market"], e2e_db_env["symbol"])
        close_null = (
            df["Close"].isnull().sum() if "Close" in df.columns else df["close"].isnull().sum()
        )
        assert close_null == 0, f"Close に NaN が {close_null} 件存在する"

    def test_raw_ohlcv_columns(self, e2e_db_env):
        """OHLCV の必須カラムが揃っていること。"""
        from src.utils.db import load_raw_ohlcv

        df = load_raw_ohlcv(e2e_db_env["market"], e2e_db_env["symbol"])
        # load_raw_ohlcv は先頭大文字で返す
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"必須カラム '{col}' が存在しない"


class TestStockFeatures:
    """Step 2: stock_features への書き込み確認"""

    def test_features_rows_inserted(self, e2e_db_env):
        """stock_features に特徴量行が存在すること。"""
        from src.utils.db import load_stock_features

        df = load_stock_features(e2e_db_env["market"], e2e_db_env["symbol"])

        assert df is not None, "load_stock_features が None を返した"
        assert not df.empty, "stock_features に行が存在しない"
        assert len(df) >= 50, f"想定より行数が少ない: {len(df)}"

    def test_features_has_y_column(self, e2e_db_env):
        """ターゲット列 y が存在すること。"""
        from src.utils.db import load_stock_features

        df = load_stock_features(e2e_db_env["market"], e2e_db_env["symbol"])
        assert "y" in df.columns, "y 列が存在しない"

    def test_features_has_lag_columns(self, e2e_db_env):
        """ラグ特徴量（Close_lag1 等）が生成されていること。"""
        from src.utils.db import load_stock_features

        df = load_stock_features(e2e_db_env["market"], e2e_db_env["symbol"])
        lag_cols = [c for c in df.columns if "_lag" in c]
        assert len(lag_cols) > 0, "ラグ特徴量が存在しない"

    def test_features_has_technical_columns(self, e2e_db_env):
        """テクニカル指標（SMA / RSI 等）が生成されていること。"""
        from src.utils.db import load_stock_features

        df = load_stock_features(e2e_db_env["market"], e2e_db_env["symbol"])
        ta_cols = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ("sma", "rsi", "macd", "bb", "atr"))
        ]
        assert len(ta_cols) > 0, "テクニカル指標が存在しない"


class TestModelTraining:
    """Step 3: モデル学習・保存確認"""

    def test_xgboost_model_file_exists(self, e2e_db_env):
        """XGBoost モデルファイルが保存されていること。"""
        models_dir = e2e_db_env["models_dir"]
        market = e2e_db_env["market"]
        symbol = e2e_db_env["symbol"]
        model_path = os.path.join(models_dir, f"{market}_{symbol}", "StockXGBoostModel.joblib")
        assert os.path.exists(model_path), f"XGBoost モデルが存在しない: {model_path}"

    def test_lightgbm_model_file_exists(self, e2e_db_env):
        """LightGBM モデルファイルが保存されていること。"""
        models_dir = e2e_db_env["models_dir"]
        market = e2e_db_env["market"]
        symbol = e2e_db_env["symbol"]
        model_path = os.path.join(models_dir, f"{market}_{symbol}", "StockLightGBMModel.joblib")
        assert os.path.exists(model_path), f"LightGBM モデルが存在しない: {model_path}"

    def test_model_metrics_saved_to_db(self, e2e_db_env):
        """model_metrics テーブルに学習指標が保存されていること。"""
        import src.utils.db._connection as _conn

        with _conn._db_connection() as con:
            rows = con.execute(
                "SELECT * FROM model_metrics WHERE market = ? AND symbol = ?",
                [e2e_db_env["market"], e2e_db_env["symbol"]],
            ).fetchall()

        assert len(rows) > 0, "model_metrics に行が存在しない"

    def test_experiment_run_saved_to_db(self, e2e_db_env):
        """experiment_runs テーブルに実験ランが記録されていること。"""
        import src.utils.db._connection as _conn

        with _conn._db_connection() as con:
            rows = con.execute(
                "SELECT * FROM experiment_runs WHERE market = ? AND symbol = ?",
                [e2e_db_env["market"], e2e_db_env["symbol"]],
            ).fetchall()

        assert len(rows) > 0, "experiment_runs に行が存在しない"


class TestPrediction:
    """Step 4: 予測実行確認（yfinance はモック）"""

    def test_predict_single_stock_returns_result(self, e2e_db_env):
        """predict_single_stock が None 以外の PredictionResult を返すこと。"""
        from src.prediction.predict_single import predict_single_stock

        with patch(
            "src.market_data.prediction_adapter.PredictionMarketDataAdapter.get_stock_data",
            return_value=e2e_db_env["ohlcv"],
        ):
            result = predict_single_stock(
                e2e_db_env["market"],
                e2e_db_env["symbol"],
            )

        assert result is not None, "predict_single_stock が None を返した"

    def test_prediction_result_has_valid_price(self, e2e_db_env):
        """予測終値が正の値であること。"""
        from src.prediction.predict_single import predict_single_stock

        with patch(
            "src.market_data.prediction_adapter.PredictionMarketDataAdapter.get_stock_data",
            return_value=e2e_db_env["ohlcv"],
        ):
            result = predict_single_stock(
                e2e_db_env["market"],
                e2e_db_env["symbol"],
            )

        assert result is not None
        assert result.avg_pred_price > 0, f"予測終値が正でない: {result.avg_pred_price}"

    def test_prediction_result_diff_ratio_reasonable(self, e2e_db_env):
        """予測変化率が ±50% 以内の合理的な範囲であること。"""
        from src.prediction.predict_single import predict_single_stock

        with patch(
            "src.market_data.prediction_adapter.PredictionMarketDataAdapter.get_stock_data",
            return_value=e2e_db_env["ohlcv"],
        ):
            result = predict_single_stock(
                e2e_db_env["market"],
                e2e_db_env["symbol"],
            )

        assert result is not None
        assert abs(result.diff_ratio) < 0.5, f"予測変化率が非現実的: {result.diff_ratio:.2%}"

    def test_prediction_result_saved_to_db(self, e2e_db_env):
        """予測結果を DB へ保存できること。"""
        import src.utils.db._connection as _conn
        from src.prediction.db import save_prediction_results
        from src.prediction.predict_single import predict_single_stock

        with patch(
            "src.market_data.prediction_adapter.PredictionMarketDataAdapter.get_stock_data",
            return_value=e2e_db_env["ohlcv"],
        ):
            result = predict_single_stock(
                e2e_db_env["market"],
                e2e_db_env["symbol"],
            )

        assert result is not None
        save_prediction_results("20260412_e2etest", [result])

        with _conn._db_connection() as con:
            rows = con.execute(
                "SELECT * FROM prediction_results WHERE market = ? AND symbol = ?",
                [e2e_db_env["market"], e2e_db_env["symbol"]],
            ).fetchall()
        assert len(rows) > 0, "prediction_results に行が保存されていない"
