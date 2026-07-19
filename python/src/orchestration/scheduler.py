"""
スケジューラパイプラインの業務ロジック（ファサード）。

定期実行スケジューラから呼び出されるオーケストレーション層。
個々のジョブ定義は cadence 別に `src/orchestration/jobs/` へ分割されており、
本モジュールは後方互換のための re-export ファサードとして機能する（#497）。

加えて、import 時にポート配線・モデルマネージャ登録・予測関数登録という
モジュールレベルの初期化副作用をここで一度だけ実行する。
"""

from src.backtest.ports import set_model_manager_factory
from src.orchestration.jobs.common import _handle_stage_error, _is_first_week_of_month
from src.orchestration.jobs.daily import (
    run_daily_auto_order,
    run_daily_backup,
    run_daily_drift_check,
    run_daily_paper_trade_report,
    run_daily_pipeline,
    run_daily_rule_signals,
    run_daily_settle_orders,
    run_horizon_exit_check,
    run_pre_close_alert,
)
from src.orchestration.jobs.periodic import (
    run_monthly_report_job,
    run_nightly_strategy_factory,
    run_strategy_promotion_check,
)
from src.orchestration.jobs.weekly import (
    run_weekly_db_maintenance,
    run_weekly_optimization,
    run_weekly_report,
    run_weekly_rule_evaluation,
    run_weekly_shadow_evaluation,
    run_weekly_training,
    run_weekly_walk_forward_report,
    run_weekly_watchlist_refresh,
)
from src.orchestration.port_wiring import wire_ports
from src.orchestration.types import PipelineStage
from src.prediction.manager import ModelManager
from src.prediction.predict_single import explain_prediction_shap, predict_single_stock
from src.reporting.query_service import register_prediction_fns
from src.utils.logger import get_logger

# モジュール初期化副作用（import 時に一度だけ実行）
wire_ports()
set_model_manager_factory(ModelManager)
register_prediction_fns(predict_single_stock, explain_prediction_shap)

logger = get_logger(__name__)

__all__ = [
    "PipelineStage",
    "_handle_stage_error",
    "_is_first_week_of_month",
    # daily
    "run_daily_pipeline",
    "run_daily_auto_order",
    "run_horizon_exit_check",
    "run_daily_settle_orders",
    "run_daily_paper_trade_report",
    "run_daily_drift_check",
    "run_pre_close_alert",
    "run_daily_backup",
    "run_daily_rule_signals",
    # weekly
    "run_weekly_training",
    "run_weekly_report",
    "run_weekly_optimization",
    "run_weekly_walk_forward_report",
    "run_weekly_watchlist_refresh",
    "run_weekly_db_maintenance",
    "run_weekly_shadow_evaluation",
    "run_weekly_rule_evaluation",
    # periodic
    "run_monthly_report_job",
    "run_nightly_strategy_factory",
    "run_strategy_promotion_check",
]
