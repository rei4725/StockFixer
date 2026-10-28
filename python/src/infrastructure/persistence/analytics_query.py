"""reporting 向けの読み取り専用クエリのアダプタ。

paper_real_diff は trading が書き reporting だけが読むため、SQL をここに持つ。
予測精度系（prediction_accuracy / accuracy_weekly_snapshots）は prediction BC 自身も
promotion_gate / shadow_evaluation / drift_monitor で読み書きする所有テーブルであるため、
SQL は src.prediction.db に残し、ここでは委譲するだけに留める（SQL を二重化しない）。
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.domain.ports import AnalyticsQuery
from src.prediction.db import accuracy as _accuracy
from src.utils.db import db_connection


class PostgresAnalyticsQuery(AnalyticsQuery):
    """Postgres から reporting 向けの集計を読む AnalyticsQuery 実装。"""

    def paper_real_diff_summary(self, recent_days: int = 7) -> dict:
        """スリッページの平均は損の向きを正にした値で返す（#1261）。

        書く所は side に関わらず (約定値-シグナル値)/シグナル値 を入れる。
        side は OrderSide の BUY=1・SELL=2・SHORT=3・SHORT_COVER=4 で、
        売りの向き（SELL・SHORT）は約定値が低いほど損なので符号を反転して平均する。
        """
        since = datetime.now() - timedelta(days=recent_days)
        with db_connection() as con:
            row = con.execute(
                """
            SELECT
                COUNT(*) AS tracked_count,
                COUNT(*) FILTER (
                    WHERE paper_price IS NOT NULL AND real_price IS NOT NULL
                ) AS comparable_count,
                AVG(
                    CASE WHEN side IN (2, 3) THEN -paper_slippage ELSE paper_slippage END
                ) AS avg_paper_slippage,
                AVG(
                    CASE WHEN side IN (2, 3) THEN -real_slippage ELSE real_slippage END
                ) AS avg_real_slippage,
                AVG(ABS(price_diff)) AS avg_abs_price_diff,
                AVG(ABS(price_diff / NULLIF(signal_price, 0))) AS avg_abs_diff_ratio,
                MAX(ABS(price_diff)) AS max_abs_price_diff
            FROM paper_real_diff
            WHERE COALESCE(paper_filled_at, real_checked_at, created_at) >= %s
            """,
                [since],
            ).fetchone()

        if row is None:
            # COUNT / AVG の集約クエリは必ず 1 行返るが、型は Optional のため
            # 防御的に全ゼロ相当の行として扱う。
            row = (0, 0, None, None, None, None, None)

        return {
            "tracked_count": int(row[0] or 0),
            "comparable_count": int(row[1] or 0),
            "avg_paper_slippage": float(row[2] or 0.0),
            "avg_real_slippage": float(row[3] or 0.0),
            "avg_abs_price_diff": float(row[4] or 0.0),
            "avg_abs_diff_ratio": float(row[5] or 0.0),
            "max_abs_price_diff": float(row[6] or 0.0),
        }

    def drift_summary(
        self,
        horizon: int = 1,
        recent_n: int = 30,
        model_name: str = "production",
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        return _accuracy.load_drift_summary(
            horizon=horizon,
            recent_n=recent_n,
            model_name=model_name,
            raise_on_error=raise_on_error,
        )

    def prediction_accuracy(
        self,
        market: Optional[str] = None,
        symbol: Optional[str] = None,
        horizon: int = 1,
        limit: int = 500,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        return _accuracy.load_prediction_accuracy(
            market=market,
            symbol=symbol,
            horizon=horizon,
            limit=limit,
            raise_on_error=raise_on_error,
        )

    def weekly_accuracy_snapshots(self, n_weeks: int = 4) -> pd.DataFrame:
        return _accuracy.load_weekly_accuracy_snapshots(n_weeks=n_weeks)
