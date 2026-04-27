"""
レポートドメイン（Reporting Domain）

月次レポート・Discord クエリサービスに関する公開 API。
DDD への移行を見据えた境界コンテキストの入口。

# 利用例
    from src.services.reporting import get_ranked_prediction_results
"""

from src.services.reporting.discord_query_service import (  # noqa: F401
    get_latest_market_prediction_snapshots,
    get_monthly_report_summary,
    get_ranked_prediction_results,
    get_scheduler_job_statuses,
    get_signal_snapshot,
    get_watchlist_prediction_view,
)

__all__ = [
    "get_ranked_prediction_results",
    "get_latest_market_prediction_snapshots",
    "get_watchlist_prediction_view",
    "get_signal_snapshot",
    "get_scheduler_job_statuses",
    "get_monthly_report_summary",
]
