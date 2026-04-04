import unittest
from unittest.mock import patch

import pandas as pd

from src.services.portfolio_backtest import (
    _attach_regime_metrics,
    _limit_portfolio_candidates_by_sector,
)


class TestPortfolioBacktestRegimeMetrics(unittest.TestCase):
    def test_attach_regime_metrics_adds_regime_column_and_summary(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        equity_df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "portfolio_value": [100.0, 103.0, 101.0, 104.0, 106.0, 105.0],
                "equal_weight_value": [100.0, 101.0, 100.0, 101.0, 102.0, 101.0],
            }
        )
        close_matrix = pd.DataFrame(
            {
                "jp_7203": [100, 101, 99, 98, 100, 101],
                "jp_6758": [80, 81, 82, 81, 80, 79],
            },
            index=dates,
        )
        mocked_regime = pd.Series(
            ["bull", "bull", "bear", "bear", "range", "range"],
            index=dates,
            dtype=str,
        )

        with patch("src.services.portfolio_backtest.get_market_regime", return_value=mocked_regime):
            enriched, regime_metrics = _attach_regime_metrics(equity_df, close_matrix)

        self.assertIn("regime", enriched.columns)
        self.assertEqual(enriched["regime"].tolist(), mocked_regime.tolist())
        self.assertEqual(regime_metrics["bull"]["days"], 2)
        self.assertEqual(regime_metrics["bear"]["days"], 2)
        self.assertEqual(regime_metrics["range"]["days"], 2)
        self.assertIn("all", regime_metrics)
        self.assertIn("hit_rate", regime_metrics["bull"])


class TestPortfolioBacktestSectorLimit(unittest.TestCase):
    @patch("src.services.portfolio_backtest.get_symbol_sector")
    def test_limit_portfolio_candidates_by_sector_caps_same_sector(self, mock_get_sector):
        mock_get_sector.side_effect = ["Auto", "Auto", "Tech", "Bank"]
        top_candidates = pd.Series(
            [0.04, 0.03, 0.025, 0.02],
            index=["jp_7203", "jp_7267", "jp_6758", "jp_8306"],
            dtype=float,
        )

        limited = _limit_portfolio_candidates_by_sector(top_candidates, max_sector_positions=1)

        self.assertEqual(limited.index.tolist(), ["jp_7203", "jp_6758", "jp_8306"])
        self.assertEqual(limited.tolist(), [0.04, 0.025, 0.02])


if __name__ == "__main__":
    unittest.main()
