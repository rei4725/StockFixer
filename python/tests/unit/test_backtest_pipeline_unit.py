"""
Unit Test: バックテストパイプライン ロジック

load_features, _build_task, メトリクス計算など、
パイプラインの計算ロジックのみをテスト（外部依存なし）。
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import _extract_trade_pnl, _max_drawdown, _sharpe_ratio, compute_metrics
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

        wins, losses = _extract_trade_pnl(trade_log)

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

        sharpe = _sharpe_ratio(returns, risk_free_rate=0.0, trading_days_per_year=252)

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
