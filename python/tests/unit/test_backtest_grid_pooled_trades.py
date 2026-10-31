"""run_optimization の候補ごとの要約で、Kelly の入力を全 fold の取引をまとめて出すことのテスト（#1436）。"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


def _fold(trade_returns, sharpe, total_return):
    """compute_metrics と同じ分け方（0 以上を勝ち）で fold の行を作る。"""
    wins = [r for r in trade_returns if r >= 0]
    losses = [abs(r) for r in trade_returns if r < 0]
    n = len(trade_returns)
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": -0.05,
        "num_trades": n,
        "win_rate": len(wins) / n if n else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "trade_returns": list(trade_returns),
    }


def _issue_wf_df():
    """Issue の例: A が10取引（勝率 0.4・±2%）、B と C が1勝ずつ、D と E が取引なし。"""
    return pd.DataFrame(
        [
            _fold([0.02] * 4 + [-0.02] * 6, 0.5, -0.04),
            _fold([0.02], 1.0, 0.02),
            _fold([0.02], 1.5, 0.02),
            _fold([], 0.0, 0.0),
            _fold([], 0.0, 0.0),
        ]
    )


def _no_trade_wf_df():
    return pd.DataFrame([_fold([], 0.0, 0.0) for _ in range(5)])


def _run(wf_dfs):
    from src.backtest.optimizer import run_optimization

    with patch("src.backtest.optimizer.grid.run_backtest_walk_forward") as mock_wf:
        mock_wf.side_effect = [(None, None, df) for df in wf_dfs]
        return run_optimization(
            market="jp",
            symbol="7203",
            threshold_min=0.01,
            threshold_max=0.01 * len(wf_dfs),
            threshold_step=0.01,
        )


class TestGridPooledKellyInputs(unittest.TestCase):
    def test_pools_trades_across_folds(self):
        """取引の無い fold・負けの無い fold の 0.0 を混ぜず、全取引をまとめた値にする"""
        row = _run([_issue_wf_df()]).iloc[0]
        self.assertAlmostEqual(float(row["win_rate"]), 0.5)
        self.assertAlmostEqual(float(row["avg_win"]), 0.02)
        self.assertAlmostEqual(float(row["avg_loss"]), 0.02)

    def test_no_trades_in_any_fold_stays_zero(self):
        """全 fold で取引が無い候補は3つとも 0.0 のまま"""
        row = _run([_no_trade_wf_df()]).iloc[0]
        self.assertEqual(float(row["win_rate"]), 0.0)
        self.assertEqual(float(row["avg_win"]), 0.0)
        self.assertEqual(float(row["avg_loss"]), 0.0)

    def test_other_columns_keep_fold_mean(self):
        """sharpe_ratio など他の列は今までどおり fold の単純平均"""
        wf_df = _issue_wf_df()
        row = _run([wf_df]).iloc[0]
        for col in ("sharpe_ratio", "total_return", "max_drawdown", "num_trades"):
            self.assertAlmostEqual(float(row[col]), float(wf_df[col].mean()), msg=col)
        self.assertNotIn("trade_returns", row.index)

    def test_without_trade_returns_column_keeps_fold_mean(self):
        """trade_returns の列が無い wf_df では今までどおり fold の平均"""
        wf_df = _issue_wf_df().drop(columns=["trade_returns"])
        row = _run([wf_df]).iloc[0]
        self.assertAlmostEqual(float(row["win_rate"]), float(wf_df["win_rate"].mean()))
        self.assertAlmostEqual(float(row["avg_loss"]), float(wf_df["avg_loss"].mean()))

    def test_error_fold_is_skipped(self):
        """エラーで trade_returns が欠けた fold は取引なしとして扱う"""
        wf_df = pd.concat(
            [_issue_wf_df(), pd.DataFrame([{"fold": 6, "error": "x"}])], ignore_index=True
        )
        row = _run([wf_df]).iloc[0]
        self.assertAlmostEqual(float(row["win_rate"]), 0.5)
        self.assertAlmostEqual(float(row["avg_loss"]), 0.02)

    def test_best_row_selection_unchanged(self):
        """save_optimal_params_json が選ぶ最良の行（sharpe_ratio の最大）は変わらない"""
        from src.backtest.optimizer import persistence, save_optimal_params_json

        low = _issue_wf_df()
        high = _issue_wf_df()
        high["sharpe_ratio"] = high["sharpe_ratio"] + 1.0
        result = _run([low, high])

        with tempfile.TemporaryDirectory() as root:
            fake_file = os.path.join(root, "src", "backtest", "optimizer", "persistence.py")
            with patch.object(persistence, "__file__", fake_file):
                path = save_optimal_params_json(result, "jp", "7203", sort_by="sharpe_ratio")
                with open(path, "r", encoding="utf-8") as f:
                    saved = json.load(f)["jp_7203"]

        self.assertAlmostEqual(saved["threshold"], 0.02)
        self.assertAlmostEqual(saved["metrics"]["sharpe_ratio"], float(high["sharpe_ratio"].mean()))
        self.assertAlmostEqual(saved["metrics"]["win_rate"], 0.5)
        self.assertAlmostEqual(saved["metrics"]["avg_win"], 0.02)
        self.assertAlmostEqual(saved["metrics"]["avg_loss"], 0.02)


if __name__ == "__main__":
    unittest.main()
