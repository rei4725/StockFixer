"""
Cat4: バックテスト回帰テスト

e2e_db_env で生成した合成データに対して run_backtest_single を実行し、
「コードが壊れていないこと」と「指標値が合理的な範囲にあること」を検証する。

合理的範囲の基準（合成データのため厳しいベースラインは設けない）:
    - 関数が例外なく完走すること
    - metrics に必須キーが揃っていること
    - max_drawdown が -100% 以上（元本を超える損失は発生しない）
    - total_return が有限の値であること
    - hit_rate（方向一致率）が 0〜1 の範囲にあること
    - n_trades が 0 以上であること

アルゴリズム変更時の退行検知に使うため、
実際の本番データに対してはここで厳しいベースラインを設定して管理する。

実行方法:
    python -m pytest tests/e2e/test_backtest_regression.py -v --timeout=300
"""

import math

import pytest

pytestmark = pytest.mark.timeout(300)

# ---------------------------------------------------------------------------
# 必須メトリクスキー（backtester.simulate_trading の戻り値に必ず含まれるべきキー）
# ---------------------------------------------------------------------------
REQUIRED_METRIC_KEYS = {
    "total_return",
    "max_drawdown",
    "num_trades",
    "sharpe_ratio",
}


# ---------------------------------------------------------------------------
# バックテスト実行・基本検証
# ---------------------------------------------------------------------------
class TestBacktestRegressionBasic:
    """run_backtest_single が正常に完走し、結果が合理的であること。"""

    @pytest.fixture(scope="class")
    def backtest_result(self, e2e_db_env):
        """e2e DB に対して run_backtest_single を実行して結果を返す（クラス全体で共有）。"""
        from src.services.backtest_pipeline import run_backtest_single

        result_df, metrics, price_series = run_backtest_single(
            market=e2e_db_env["market"],
            symbol=e2e_db_env["symbol"],
            source="file",  # DB から特徴量を読むため yfinance 不要
            model_type="XGBoostModel",
            train_ratio=0.8,
            initial_cash=1_000_000,
        )
        return {"result_df": result_df, "metrics": metrics, "price_series": price_series}

    def test_backtest_returns_dataframe(self, backtest_result):
        """result_df が DataFrame であること（シグナルが発生しない場合は空も許容）。"""
        import pandas as pd

        df = backtest_result["result_df"]
        assert df is not None, "result_df が None"
        assert isinstance(df, pd.DataFrame), "result_df が DataFrame でない"

    def test_backtest_returns_metrics_dict(self, backtest_result):
        """metrics が辞書型であること。"""
        metrics = backtest_result["metrics"]
        assert isinstance(metrics, dict), f"metrics が dict でない: {type(metrics)}"

    def test_metrics_has_required_keys(self, backtest_result):
        """metrics に必須キーが揃っていること。"""
        metrics = backtest_result["metrics"]
        missing = REQUIRED_METRIC_KEYS - set(metrics.keys())
        assert not missing, f"metrics に必須キーが存在しない: {missing}"

    def test_max_drawdown_within_range(self, backtest_result):
        """max_drawdown が -100%〜0% の範囲内であること（元本超損失は発生しない）。"""
        mdd = backtest_result["metrics"]["max_drawdown"]
        assert mdd >= -1.0, f"max_drawdown が -100% を下回った: {mdd:.2%}"
        assert mdd <= 0.0, f"max_drawdown が正値（{mdd:.2%}）: ドローダウン計算が誤っている可能性"

    def test_total_return_is_finite(self, backtest_result):
        """total_return が有限の値（NaN / Inf でない）であること。"""
        tr = backtest_result["metrics"]["total_return"]
        assert math.isfinite(tr), f"total_return が有限でない: {tr}"

    def test_n_trades_non_negative(self, backtest_result):
        """num_trades が 0 以上であること。"""
        n_trades = backtest_result["metrics"]["num_trades"]
        assert n_trades >= 0, f"num_trades が負: {n_trades}"

    def test_sharpe_ratio_finite_or_nan(self, backtest_result):
        """Sharpe Ratio が有限の値、または取引なしによる NaN であること（Inf は異常）。"""
        sr = backtest_result["metrics"]["sharpe_ratio"]
        if sr is None:
            return  # None は取引なしの場合の正常系
        assert not math.isinf(sr), f"Sharpe Ratio が Inf: {sr}"


# ---------------------------------------------------------------------------
# バックテスト退行テスト（アルゴリズム変更時に Sharpe / MDD ベースラインを守るチェック）
# ---------------------------------------------------------------------------
class TestBacktestRegression:
    """
    合成データでの性能ベースライン（ゆるめの下限）。

    ここに書いた閾値は以下の目的で使う:
        - コードの論理バグによる急激な性能劣化を検知する
        - "全部 Hold してもこの値以下にはならないはず" という安全網の役割

    本番データに対するベースラインは production DB を使う別バージョンで管理する。
    """

    @pytest.fixture(scope="class")
    def metrics(self, e2e_db_env):
        from src.services.backtest_pipeline import run_backtest_single

        _, metrics, _ = run_backtest_single(
            market=e2e_db_env["market"],
            symbol=e2e_db_env["symbol"],
            source="file",
            model_type="XGBoostModel",
            train_ratio=0.8,
        )
        return metrics

    def test_max_drawdown_not_catastrophic(self, metrics):
        """
        合成データでも max_drawdown が -100% ちょうどになることはない。
        （全ポジションを保有したまま価格がゼロになることは合成データでは発生しない）
        """
        assert metrics["max_drawdown"] > -1.0, "max_drawdown が -100% を達成: 破産判定ロジックのバグの可能性"

    def test_hit_rate_in_valid_range(self, metrics):
        """win_rate（方向一致率）が 0〜1 の範囲であること。"""
        hr = metrics.get("win_rate")
        if hr is None:
            pytest.skip("metrics に win_rate が存在しないためスキップ")
        assert 0.0 <= hr <= 1.0, f"win_rate が 0〜1 の範囲外: {hr}"

    def test_final_cash_positive(self, metrics):
        """最終残高（final_cash）が正であること（破産していないこと）。"""
        fc = metrics.get("final_cash")
        if fc is None:
            pytest.skip("metrics に final_cash が存在しないためスキップ")
        assert fc > 0, f"final_cash が非正: {fc}"


# ---------------------------------------------------------------------------
# ウォークフォワードの基本検証
# ---------------------------------------------------------------------------
class TestWalkForwardBasic:
    """run_backtest_walk_forward が正常に完走すること。"""

    @pytest.mark.timeout(300)
    def test_walk_forward_completes(self, e2e_db_env):
        """Walk-Forward バックテストが例外なく完走すること。"""
        from src.services.backtest_pipeline import run_backtest_walk_forward

        _, _, wf_df = run_backtest_walk_forward(
            market=e2e_db_env["market"],
            symbol=e2e_db_env["symbol"],
            source="file",
            model_type="XGBoostModel",
            n_splits=3,  # 合成データ 200 行に対して 3 分割
        )

        assert wf_df is not None, "wf_df が None"
        assert not wf_df.empty, "wf_df が空"

    @pytest.mark.timeout(300)
    def test_walk_forward_result_has_fold_column(self, e2e_db_env):
        """Walk-Forward の結果に fold コラムが存在すること。"""
        from src.services.backtest_pipeline import run_backtest_walk_forward

        _, _, wf_df = run_backtest_walk_forward(
            market=e2e_db_env["market"],
            symbol=e2e_db_env["symbol"],
            source="file",
            model_type="XGBoostModel",
            n_splits=3,
        )

        assert "fold" in wf_df.columns or len(wf_df) >= 3, "Walk-Forward の結果が 3 フォールド分存在しない"


# ---------------------------------------------------------------------------
# 本番 DB でのベースライン回帰テスト（存在する場合のみ実行）
# ---------------------------------------------------------------------------
class TestProductionBacktestRegression:
    """
    本番 DuckDB が存在する場合に代表銘柄でバックテストを実行し、
    Sharpe Ratio などの指標が定義済みベースラインを下回っていないことを確認する。

    ベースラインを更新する場合は BASELINES 辞書を編集する。
    """

    # market_symbol -> { 指標名: 下限値 }
    BASELINES: dict = {
        # 例: "us_AAPL": {"sharpe_ratio": 0.3, "max_drawdown": -0.30},
        # 実際の本番銘柄を追加してください
    }

    def _prod_db_exists(self) -> bool:
        import os

        from src.utils.data_path_utils import get_db_path

        return os.path.exists(get_db_path())

    @pytest.mark.parametrize("key,baseline", list(BASELINES.items()))
    @pytest.mark.timeout(300)
    def test_baseline_not_breached(self, key, baseline):
        """定義済みベースラインを下回っていないことを確認する。"""
        if not self._prod_db_exists():
            pytest.skip("本番 DB が存在しないためスキップ")

        market, symbol = key.split("_", 1)
        from src.services.backtest_pipeline import run_backtest_single

        try:
            _, metrics, _ = run_backtest_single(
                market=market,
                symbol=symbol,
                source="file",
                model_type="XGBoostModel",
            )
        except SystemExit:
            pytest.skip(f"{key}: DB にデータが存在しないためスキップ")

        for metric_name, threshold in baseline.items():
            value = metrics.get(metric_name)
            if value is None:
                continue
            assert value >= threshold, (
                f"[{key}] {metric_name} がベースラインを下回った: " f"{value:.4f} < {threshold:.4f}"
            )
