"""
Cat2: データ品質アサーション

e2e_db_env で生成した合成データに対して「書き込まれたデータが品質基準を満たすこと」を検証する。
また production DB が利用可能な場合はそちらにも同じ品質チェックを適用する。

チェック対象:
    - NaN / Inf 混入
    - (market, symbol, date) の重複行
    - Close 値の正値チェック
    - y 列（ターゲット変化率）の分布が合理的であること
    - prediction_results の予測値が合理的範囲内であること

実行方法:
    python -m pytest tests/e2e/test_data_quality.py -v --timeout=120
"""

import numpy as np
import pytest

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------------------
# stock_features テーブルの品質チェック
# ---------------------------------------------------------------------------
class TestStockFeaturesQuality:
    """e2e 環境の stock_features テーブルに対する品質アサーション。"""

    def _load(self, e2e_db_env):
        from src.utils.db import load_stock_features

        df = load_stock_features(e2e_db_env["market"], e2e_db_env["symbol"])
        assert df is not None and not df.empty, "stock_features が空"
        return df

    def test_no_nan_in_numeric_features(self, e2e_db_env):
        """数値特徴量列に NaN が存在しないこと。"""
        df = self._load(e2e_db_env)
        exclude = {"market", "symbol", "y", "date"}
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
        nan_counts = df[numeric_cols].isnull().sum()
        bad_cols = nan_counts[nan_counts > 0]
        assert bad_cols.empty, f"NaN を含む列が存在する:\n{bad_cols.to_dict()}"

    def test_no_inf_in_numeric_features(self, e2e_db_env):
        """数値特徴量列に Inf が存在しないこと。"""
        df = self._load(e2e_db_env)
        exclude = {"market", "symbol", "y", "date"}
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
        inf_counts = np.isinf(df[numeric_cols].values).sum(axis=0)
        bad = {
            numeric_cols[i]: int(inf_counts[i])
            for i in range(len(numeric_cols))
            if inf_counts[i] > 0
        }
        assert not bad, f"Inf を含む列が存在する:\n{bad}"

    def test_no_duplicate_rows(self, e2e_db_env):
        """(market, symbol, date) の重複行が存在しないこと。
        \n        load_stock_features は market / symbol 列を drop して返すため、date 列のみで重複确認する。
        """
        df = self._load(e2e_db_env)
        if "date" not in df.columns:
            pytest.skip("date 列が存在しないためスキップ")
        dup_count = df.duplicated(subset=["date"]).sum()
        assert dup_count == 0, f"重複行が {dup_count} 件存在する"

    def test_close_lag1_positive(self, e2e_db_env):
        """Close_lag1 が正値であること（ゼロや負値はデータ異常）。"""
        df = self._load(e2e_db_env)
        if "Close_lag1" not in df.columns:
            pytest.skip("Close_lag1 列が存在しないためスキップ")
        bad = (df["Close_lag1"] <= 0).sum()
        assert bad == 0, f"Close_lag1 に非正値が {bad} 件存在する"

    def test_y_column_finite_and_small(self, e2e_db_env):
        """
        ターゲット列 y（翌日変化率）が有限かつ ±100% 以内であること。
        100% を超える日次変化率は通常発生しないためデータ異常を示す。
        """
        df = self._load(e2e_db_env)
        assert "y" in df.columns, "y 列が存在しない"
        y = df["y"].dropna()
        assert not np.isinf(y).any(), "y 列に Inf が存在する"
        extreme = (y.abs() > 1.0).sum()
        assert extreme == 0, f"y が ±100% を超える行が {extreme} 件存在する"

    def test_row_count_sufficient(self, e2e_db_env):
        """モデル学習に十分な行数（>= 50）が存在すること。"""
        df = self._load(e2e_db_env)
        assert len(df) >= 50, f"行数が不足: {len(df)}"

    def test_feature_column_count_sufficient(self, e2e_db_env):
        """特徴量列数が最低 10 以上あること（SHAP / 特徴量選択に必要）。"""
        df = self._load(e2e_db_env)
        exclude = {"market", "symbol", "y", "date", "market_encoded", "row_num"}
        feature_cols = [c for c in df.columns if c not in exclude]
        assert len(feature_cols) >= 10, f"特徴量列数が少なすぎる: {len(feature_cols)}"


# ---------------------------------------------------------------------------
# market_data_raw テーブルの品質チェック
# ---------------------------------------------------------------------------
class TestMarketDataRawQuality:
    """e2e 環境の market_data_raw テーブルに対する品質アサーション。"""

    def _load(self, e2e_db_env):
        from src.utils.db import load_raw_ohlcv

        df = load_raw_ohlcv(e2e_db_env["market"], e2e_db_env["symbol"])
        assert df is not None and not df.empty, "market_data_raw が空"
        return df

    def test_ohlcv_price_positive(self, e2e_db_env):
        """OHLCV の価格列（Open/High/Low/Close）がすべて正値であること。"""
        df = self._load(e2e_db_env)
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                continue
            bad = (df[col] <= 0).sum()
            assert bad == 0, f"{col} に非正値が {bad} 件存在する"

    def test_high_gte_low(self, e2e_db_env):
        """High >= Low が常に成立すること。"""
        df = self._load(e2e_db_env)
        if "High" not in df.columns or "Low" not in df.columns:
            pytest.skip("High / Low 列が存在しないためスキップ")
        bad = (df["High"] < df["Low"]).sum()
        assert bad == 0, f"High < Low が {bad} 件存在する"

    def test_volume_non_negative(self, e2e_db_env):
        """Volume が非負であること。"""
        df = self._load(e2e_db_env)
        if "Volume" not in df.columns:
            pytest.skip("Volume 列が存在しないためスキップ")
        bad = (df["Volume"] < 0).sum()
        assert bad == 0, f"Volume に負値が {bad} 件存在する"

    def test_date_continuity(self, e2e_db_env):
        """
        営業日の日付連続性チェック。
        隣接行の日付差が 7 日（週をまたぐ最大ギャップ）以内であること。
        """
        df = self._load(e2e_db_env)
        df_sorted = df.sort_index()
        if len(df_sorted) < 2:
            pytest.skip("行数不足のためスキップ")
        deltas = df_sorted.index.to_series().diff().dt.days.dropna()
        max_gap = int(deltas.max())
        assert max_gap <= 7, f"日付ギャップが大きすぎる: 最大 {max_gap} 日"


# ---------------------------------------------------------------------------
# prediction_results テーブルの品質チェック（E2E 予測後）
# ---------------------------------------------------------------------------
class TestPredictionResultsQuality:
    """e2e 環境の prediction_results テーブルに対する品質アサーション。"""

    def _load(self, e2e_db_env):
        import src.utils.db._connection as _conn

        with _conn._db_connection() as con:
            try:
                df = con.execute(
                    "SELECT * FROM prediction_results WHERE market = ? AND symbol = ?",
                    [e2e_db_env["market"], e2e_db_env["symbol"]],
                ).fetchdf()
            except Exception:
                return None
        return df if not df.empty else None

    def test_prediction_exists(self, e2e_db_env):
        """
        prediction_results に E2ETEST の行が存在すること。
        ※ test_full_pipeline.py の test_prediction_result_saved_to_db が先に実行された場合のみ有効。
        """
        from unittest.mock import patch

        from src.models.predict_single_stock import predict_single_stock
        from src.utils.db import save_prediction_results

        # 予測を実行して保存（冪等: 既存行は DELETE-INSERT で置き換え）
        with patch(
            "src.models.predict_single_stock.data_loader.get_stock_data",
            return_value=e2e_db_env["ohlcv"],
        ):
            result = predict_single_stock(e2e_db_env["market"], e2e_db_env["symbol"])

        assert result is not None, "予測結果が None"
        save_prediction_results("20260412_quality_check", [result])

        df = self._load(e2e_db_env)
        assert df is not None and len(df) > 0, "prediction_results にデータが存在しない"

    def test_predicted_price_within_50pct(self, e2e_db_env):
        """予測終値が現在値の ±50% 以内であること。"""
        df = self._load(e2e_db_env)
        if df is None:
            pytest.skip("prediction_results にデータがないためスキップ")
        if "diff_ratio" not in df.columns:
            pytest.skip("diff_ratio 列が存在しないためスキップ")
        bad = (df["diff_ratio"].abs() > 0.5).sum()
        assert bad == 0, f"予測変化率が ±50% 超の行が {bad} 件存在する"

    def test_current_price_positive(self, e2e_db_env):
        """現在値（current_price）が正であること。"""
        df = self._load(e2e_db_env)
        if df is None:
            pytest.skip("prediction_results にデータがないためスキップ")
        bad = (df["current_price"] <= 0).sum()
        assert bad == 0, f"current_price に非正値が {bad} 件存在する"


# ---------------------------------------------------------------------------
# 本番 DB の品質チェック（存在する場合のみ実行）
# ---------------------------------------------------------------------------
class TestProductionDBQuality:
    """
    本番 DuckDB が存在する場合にデータ品質を確認する。
    DB がなければ自動スキップ。
    """

    @pytest.fixture(scope="class")
    def prod_db_conn(self):
        """本番 DB の読み込み接続を返す（存在しなければスキップ）。"""
        import os

        import duckdb

        from src.utils.data_path_utils import get_db_path

        db_path = get_db_path()
        if not os.path.exists(db_path):
            pytest.skip(f"本番 DB が存在しないためスキップ: {db_path}")
        try:
            con = duckdb.connect(db_path, read_only=True)
            yield con
            con.close()
        except Exception as e:
            pytest.skip(f"本番 DB に接続できないためスキップ: {e}")

    def test_stock_features_no_nan(self, prod_db_conn):
        """本番 stock_features の数値列に NaN が存在しないこと（サンプル 1000 行）。
        \n        lag 特徴量は時系列の先頭行が自然に NaN になるため除外する。
        """
        try:
            df = prod_db_conn.execute("SELECT * FROM stock_features LIMIT 1000").fetchdf()
        except Exception as e:
            pytest.skip(f"stock_features テーブルがない: {e}")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        # lag 特徴量は先頭行が NaN になるため隠れている可能性がある -> 除外
        exclude = {"y", "market_encoded", "row_num"}
        exclude_patterns = ("_lag",)
        check_cols = [
            c
            for c in numeric_cols
            if c not in exclude and not any(p in c for p in exclude_patterns)
        ]
        nan_counts = df[check_cols].isnull().sum()
        bad = nan_counts[nan_counts > 0]
        assert bad.empty, f"本番 stock_features に NaN が存在する列:\n{bad.to_dict()}"

    def test_stock_features_no_dup(self, prod_db_conn):
        """本番 stock_features に (market, symbol, date) 重複がないこと（サンプル確認）。"""
        try:
            df = prod_db_conn.execute(
                "SELECT market, symbol, date, COUNT(*) as cnt "
                "FROM stock_features "
                "GROUP BY market, symbol, date "
                "HAVING COUNT(*) > 1 "
                "LIMIT 10"
            ).fetchdf()
        except Exception as e:
            pytest.skip(f"クエリ失敗: {e}")
        assert len(df) == 0, f"重複行が {len(df)} 件存在する（先頭表示）:\n{df}"

    def test_prediction_results_diff_ratio_range(self, prod_db_conn):
        """本番 prediction_results の diff_ratio が ±50% 以内であること（最新 100 件）。"""
        try:
            df = prod_db_conn.execute(
                "SELECT diff_ratio FROM prediction_results " "ORDER BY predicted_at DESC LIMIT 100"
            ).fetchdf()
        except Exception as e:
            pytest.skip(f"prediction_results テーブルがない: {e}")

        if df.empty:
            pytest.skip("prediction_results が空")
        bad = (df["diff_ratio"].abs() > 0.5).sum()
        assert bad == 0, f"diff_ratio が ±50% 超の行が {bad} 件存在する"
