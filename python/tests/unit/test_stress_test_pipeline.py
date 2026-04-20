"""
Unit Test: ストレステストパイプライン

STRESS_SCENARIOS / _compute_max_consecutive_losses / StressTestResult 変換など、
ピュアなロジックのみをテスト（外部依存なし・Mock完全）。
"""

import re
from unittest.mock import patch

import pandas as pd

from src.domain.types import StressTestResult, SymbolTask
from src.services.stress_test_pipeline import (
    MDD_THRESHOLD,
    STRESS_SCENARIOS,
    _compute_max_consecutive_losses,
    print_stress_test_summary,
    run_stress_test_batch,
    run_stress_test_single,
    save_stress_test_results,
)


class TestStressScenarios:
    """STRESS_SCENARIOS 定数の検証"""

    def test_stress_scenarios_keys(self):
        """corona / lehman の両キーが存在すること"""
        assert "corona" in STRESS_SCENARIOS
        assert "lehman" in STRESS_SCENARIOS

    def test_stress_scenarios_date_format(self):
        """各シナリオの日付が YYYY-MM-DD 形式であること"""
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for key, scenario in STRESS_SCENARIOS.items():
            assert date_pattern.match(
                scenario["start_date"]
            ), f"{key}.start_date が YYYY-MM-DD 形式ではありません: {scenario['start_date']}"
            assert date_pattern.match(
                scenario["end_date"]
            ), f"{key}.end_date が YYYY-MM-DD 形式ではありません: {scenario['end_date']}"

    def test_stress_scenarios_label_exists(self):
        """各シナリオに label フィールドが存在すること"""
        for key, scenario in STRESS_SCENARIOS.items():
            assert "label" in scenario, f"{key} に label フィールドがありません"
            assert scenario["label"], f"{key}.label が空です"

    def test_mdd_threshold_value(self):
        """MDD_THRESHOLD が 0.15 であること"""
        assert MDD_THRESHOLD == 0.15


class TestComputeMaxConsecutiveLosses:
    """_compute_max_consecutive_losses のロジック検証"""

    def _make_trade_log(self, buy_prices, sell_prices):
        """buy→sell のペアから trade_log を生成するヘルパー"""
        rows = []
        dates = pd.date_range("2020-01-01", periods=len(buy_prices) * 2, freq="B")
        idx = 0
        for buy_p, sell_p in zip(buy_prices, sell_prices):
            rows.append({"date": dates[idx], "action": "buy", "price": buy_p, "qty": 1, "cash": 0})
            idx += 1
            rows.append(
                {
                    "date": dates[idx],
                    "action": "sell",
                    "price": sell_p,
                    "qty": 1,
                    "cash": sell_p - buy_p,
                }
            )
            idx += 1
        return pd.DataFrame(rows)

    def test_compute_max_consecutive_losses_basic(self):
        """win-loss-loss-win-loss パターンで連敗最大2が返ること"""
        # 買値100で売値: 110(勝), 90(負), 80(負), 120(勝), 95(負)
        trade_log = self._make_trade_log(
            buy_prices=[100, 100, 100, 100, 100],
            sell_prices=[110, 90, 80, 120, 95],
        )
        result = _compute_max_consecutive_losses(trade_log)
        assert result == 2

    def test_compute_max_consecutive_losses_all_wins(self):
        """全勝の場合は 0 が返ること"""
        trade_log = self._make_trade_log(
            buy_prices=[100, 100, 100],
            sell_prices=[110, 120, 130],
        )
        assert _compute_max_consecutive_losses(trade_log) == 0

    def test_compute_max_consecutive_losses_no_trades(self):
        """取引なし（空 DataFrame）の場合は 0 が返ること"""
        assert _compute_max_consecutive_losses(pd.DataFrame()) == 0

    def test_compute_max_consecutive_losses_none(self):
        """None 入力でも 0 が返ること"""
        assert _compute_max_consecutive_losses(None) == 0

    def test_compute_max_consecutive_losses_all_losses(self):
        """3連敗の場合は 3 が返ること"""
        trade_log = self._make_trade_log(
            buy_prices=[100, 100, 100],
            sell_prices=[90, 80, 70],
        )
        assert _compute_max_consecutive_losses(trade_log) == 3


class TestStressTestResultMddPass:
    """StressTestResult.mdd_pass の判定テスト"""

    def test_mdd_pass_within_threshold(self):
        """MDD が閾値以内なら mdd_pass=True"""
        result = StressTestResult(
            market="jp",
            symbol="7203",
            scenario_name="corona",
            period_start="2020-02-01",
            period_end="2020-03-31",
            mdd=-0.10,
            sharpe_ratio=0.5,
            total_return=-0.05,
            win_rate=0.45,
            num_trades=10,
            max_consecutive_losses=2,
            mdd_pass=abs(-0.10) <= 0.15,
        )
        assert result.mdd_pass is True

    def test_mdd_fail_exceeds_threshold(self):
        """MDD が閾値を超えたら mdd_pass=False"""
        result = StressTestResult(
            market="jp",
            symbol="7203",
            scenario_name="lehman",
            period_start="2008-09-01",
            period_end="2008-10-31",
            mdd=-0.30,
            sharpe_ratio=-1.0,
            total_return=-0.25,
            win_rate=0.30,
            num_trades=8,
            max_consecutive_losses=5,
            mdd_pass=abs(-0.30) <= 0.15,
        )
        assert result.mdd_pass is False

    def test_mdd_pass_exactly_at_threshold(self):
        """MDD がちょうど閾値と等しい場合は mdd_pass=True"""
        result = StressTestResult(
            market="us",
            symbol="AAPL",
            scenario_name="corona",
            period_start="2020-02-01",
            period_end="2020-03-31",
            mdd=-0.15,
            sharpe_ratio=0.0,
            total_return=0.0,
            win_rate=0.5,
            num_trades=5,
            max_consecutive_losses=1,
            mdd_pass=abs(-0.15) <= 0.15,
        )
        assert result.mdd_pass is True


class TestRunStressTestSingleMock:
    """run_stress_test_single の Mock テスト"""

    def _make_mock_trade_log(self):
        """勝ち1・負け1の取引ログを生成"""
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-02-01", periods=4, freq="B"),
                "action": ["buy", "sell", "buy", "sell"],
                "price": [100, 110, 110, 100],
                "qty": [1, 1, 1, 1],
                "cash": [0, 110, 0, 100],
            }
        )

    @patch("src.services.stress_test_pipeline.run_backtest_single")
    def test_run_stress_test_single_mock(self, mock_backtest):
        """run_backtest_single をモックして StressTestResult 変換を検証する"""
        from src.services.stress_test_pipeline import run_stress_test_single

        trade_log = self._make_mock_trade_log()
        mock_metrics = {
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.2,
            "total_return": 0.05,
            "win_rate": 0.6,
            "num_trades": 2,
        }
        mock_backtest.return_value = (trade_log, mock_metrics, pd.Series([100, 110], dtype=float))

        result = run_stress_test_single(
            market="jp",
            symbol="7203",
            scenario_name="corona",
        )

        assert result is not None
        assert isinstance(result, StressTestResult)
        assert result.market == "jp"
        assert result.symbol == "7203"
        assert result.scenario_name == "corona"
        assert result.period_start == "2020-02-01"
        assert result.period_end == "2020-03-31"
        assert result.mdd == -0.08
        assert result.sharpe_ratio == 1.2
        assert result.num_trades == 2
        assert result.mdd_pass is True  # abs(-0.08) <= 0.15

    def test_run_stress_test_single_unknown_scenario(self):
        """不明なシナリオでは None が返ること"""
        result = run_stress_test_single(
            market="jp",
            symbol="7203",
            scenario_name="unknown_scenario",
        )
        assert result is None

    @patch("src.services.stress_test_pipeline.run_backtest_single")
    def test_run_stress_test_single_backtest_failure(self, mock_backtest):
        """バックテストが例外を投げた場合 None が返ること"""
        mock_backtest.side_effect = RuntimeError("データ取得失敗")

        result = run_stress_test_single(
            market="jp",
            symbol="7203",
            scenario_name="corona",
        )
        assert result is None


class TestSaveStressTestResultsMock:
    """save_stress_test_results の CSV 保存モックテスト"""

    def _make_results(self):
        return [
            StressTestResult(
                market="jp",
                symbol="7203",
                scenario_name="corona",
                period_start="2020-02-01",
                period_end="2020-03-31",
                mdd=-0.10,
                sharpe_ratio=0.8,
                total_return=-0.05,
                win_rate=0.5,
                num_trades=6,
                max_consecutive_losses=2,
                mdd_pass=True,
            )
        ]

    @patch("src.services.stress_test_pipeline.os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_stress_test_results_mock(self, mock_to_csv, mock_makedirs):
        """to_csv が呼ばれ、返り値がファイルパス文字列であること"""
        results = self._make_results()
        filepath = save_stress_test_results(results, output_dir="results/stress_test")

        assert mock_makedirs.called
        assert mock_to_csv.called
        assert isinstance(filepath, str)
        assert "stress_test" in filepath
        assert filepath.endswith(".csv")

    def test_save_stress_test_results_empty(self):
        """空リスト時は空文字を返すこと"""
        result = save_stress_test_results([], output_dir="results/stress_test")
        assert result == ""


class TestPrintStressTestSummarySmoke:
    """print_stress_test_summary のスモークテスト"""

    def test_print_summary_no_exception(self, capsys):
        """例外なく実行でき、出力に期待文字列が含まれること"""
        results = [
            StressTestResult(
                market="jp",
                symbol="7203",
                scenario_name="corona",
                period_start="2020-02-01",
                period_end="2020-03-31",
                mdd=-0.10,
                sharpe_ratio=0.5,
                total_return=-0.05,
                win_rate=0.45,
                num_trades=5,
                max_consecutive_losses=2,
                mdd_pass=True,
            )
        ]
        print_stress_test_summary(results)
        captured = capsys.readouterr()
        assert "7203" in captured.out
        assert "コロナショック" in captured.out
        assert "PASS" in captured.out

    def test_print_summary_empty_no_exception(self, capsys):
        """空リストでも例外なく実行されること"""
        print_stress_test_summary([])
        captured = capsys.readouterr()
        assert "結果がありません" in captured.out


class TestRunStressTestBatch:
    """run_stress_test_batch のモックテスト"""

    def _make_symbol_tasks(self):
        return [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="us", symbol="AAPL"),
        ]

    @patch("src.services.stress_test_pipeline.run_stress_test_single")
    def test_batch_returns_all_results(self, mock_single):
        """全銘柄・全シナリオの結果が返ること"""
        mock_single.return_value = StressTestResult(
            market="jp",
            symbol="7203",
            scenario_name="corona",
            period_start="2020-02-01",
            period_end="2020-03-31",
            mdd=-0.08,
            sharpe_ratio=1.0,
            total_return=0.05,
            win_rate=0.6,
            num_trades=4,
            max_consecutive_losses=1,
            mdd_pass=True,
        )
        targets = self._make_symbol_tasks()
        results = run_stress_test_batch(targets=targets)

        # 2銘柄 × 2シナリオ = 4件
        assert len(results) == 4
        assert mock_single.call_count == 4

    @patch("src.services.stress_test_pipeline.run_stress_test_single")
    def test_batch_skips_failed_results(self, mock_single):
        """run_stress_test_single が None を返した銘柄はスキップされること"""
        mock_single.return_value = None
        targets = [SymbolTask(market="jp", symbol="7203")]
        results = run_stress_test_batch(targets=targets)

        assert results == []

    @patch("src.services.stress_test_pipeline.run_stress_test_single")
    def test_batch_default_all_scenarios(self, mock_single):
        """scenarios=None のとき全シナリオ（corona + lehman）が実行されること"""
        mock_single.return_value = None
        targets = [SymbolTask(market="jp", symbol="7203")]
        run_stress_test_batch(targets=targets, scenarios=None)

        called_scenarios = [call.kwargs["scenario_name"] for call in mock_single.call_args_list]
        assert set(called_scenarios) == {"corona", "lehman"}

    @patch("src.services.stress_test_pipeline.run_stress_test_single")
    def test_batch_single_scenario_filter(self, mock_single):
        """scenarios=[corona] のとき corona のみ実行されること"""
        mock_single.return_value = None
        targets = [SymbolTask(market="jp", symbol="7203")]
        run_stress_test_batch(targets=targets, scenarios=["corona"])

        assert mock_single.call_count == 1
        assert mock_single.call_args.kwargs["scenario_name"] == "corona"

    def test_batch_empty_targets(self):
        """空ターゲットでは空リストが返ること"""
        results = run_stress_test_batch(targets=[])
        assert results == []
