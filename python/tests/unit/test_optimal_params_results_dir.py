"""
ユニットテスト: optimal_params.json の置き場（#1081）

週の最適化の結果は results/（本番でホストにマウント）に書き、発注の読む側も
results/ を先に読む。config/ の写しはイメージに焼かれた初めの値で、
デプロイの起き直しで消える置き場には書かない。
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _best_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "threshold": [0.002],
            "stop_loss_pct": [0.05],
            "take_profit_pct": [0.1],
            "total_return": [0.1],
            "sharpe_ratio": [1.5],
            "max_drawdown": [-0.05],
            "win_rate": [0.6],
            "trade_count": [10],
        }
    )


class TestLoaderReadsResultsFirst:
    def test_reads_results_file_when_present(self, tmp_path, monkeypatch):
        from src.utils.optimal_params_loader import get_optimal_params

        results = tmp_path / "results"
        results.mkdir()
        seed = tmp_path / "seed.json"
        _write(results / "optimal_params.json", {"jp_7203": {"threshold": 0.02}})
        _write(seed, {"jp_7203": {"threshold": 0.01}})
        monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(results))

        with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", seed):
            params = get_optimal_params("jp", "7203")

        assert params == {"threshold": 0.02}

    def test_falls_back_to_seed_when_results_file_missing(self, tmp_path, monkeypatch):
        from src.utils.optimal_params_loader import get_optimal_params

        seed = tmp_path / "seed.json"
        _write(seed, {"jp_7203": {"threshold": 0.01}})
        monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(tmp_path / "results"))

        with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", seed):
            params = get_optimal_params("jp", "7203")

        assert params == {"threshold": 0.01}


class TestSaveWritesToResults:
    def test_saves_into_results_and_keeps_seed_entries(self, tmp_path, monkeypatch):
        from src.backtest.optimizer import save_optimal_params_json

        results = tmp_path / "results"
        seed = tmp_path / "seed.json"
        _write(seed, {"jp_1332": {"threshold": 0.001}})
        monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(results))

        with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", seed):
            path = save_optimal_params_json(_best_df(), "jp", "7203")

        assert Path(path) == results / "optimal_params.json"
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        assert saved["jp_7203"]["threshold"] == 0.002
        # まだ最適化し直していない銘柄は、初めの値のまま残す
        assert saved["jp_1332"] == {"threshold": 0.001}
        # 初めの値（イメージに焼かれた写し）は書き換えない
        assert json.loads(seed.read_text(encoding="utf-8")) == {"jp_1332": {"threshold": 0.001}}

    def test_saved_result_is_what_the_loader_reads(self, tmp_path, monkeypatch):
        from src.backtest.optimizer import save_optimal_params_json
        from src.utils.optimal_params_loader import get_optimal_params

        seed = tmp_path / "seed.json"
        _write(seed, {"jp_7203": {"threshold": 0.01}})
        monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(tmp_path / "results"))

        with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", seed):
            save_optimal_params_json(_best_df(), "jp", "7203")
            params = get_optimal_params("jp", "7203")

        assert params["threshold"] == 0.002


class TestWeeklyOptimizationResumes:
    @patch("src.reporting.discord.discord_utils.send_optimization_completion")
    @patch("src.backtest.optimizer.run_optimize_batch")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_skips_symbols_optimized_this_week(self, mock_symbols, mock_run, _mock_send):
        """デプロイの起き直しで消えた回の補完が、済んだ銘柄から始め直さない"""
        from src.orchestration.jobs.weekly import run_weekly_optimization

        mock_symbols.return_value = [{"market": "jp", "symbol": "7203"}]
        mock_run.return_value = [{"status": "success"}]

        run_weekly_optimization()

        assert mock_run.call_args.kwargs["skip_days"] == 6
