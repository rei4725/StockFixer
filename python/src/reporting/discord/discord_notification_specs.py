"""Discord 通知仕様の定義。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NotificationSpec:
    """通知のタイトルと色を保持する。"""

    title: str
    color: int


COLOR_SUCCESS = 0x00FF00
COLOR_ERROR = 0xFF0000
COLOR_INFO = 0x00BFFF
COLOR_WARNING = 0xFF9900
COLOR_CAUTION = 0xFFAA00

DAILY_PIPELINE_COMPLETION = NotificationSpec("✅ 日次パイプライン完了", COLOR_SUCCESS)
DAILY_PIPELINE_ERROR = NotificationSpec("❌ 日次パイプライン失敗", COLOR_ERROR)
WEEKLY_TRAINING_COMPLETION = NotificationSpec("✅ 週次モデル学習完了", COLOR_SUCCESS)
DAILY_ORDER_COMPLETION = NotificationSpec("✅ 自動発注完了", COLOR_INFO)
DAILY_ORDER_STOPPED = NotificationSpec("⚠️ 自動発注停止", COLOR_WARNING)
DAILY_SETTLE_COMPLETION = NotificationSpec("✅ 約定処理完了", COLOR_INFO)
OPTIMIZATION_COMPLETION = NotificationSpec("✅ 週次バックテスト最適化完了", COLOR_SUCCESS)
OPTIMIZATION_COMPLETION_WITH_ERRORS = NotificationSpec("✅ 週次バックテスト最適化完了", COLOR_CAUTION)
WALK_FORWARD_REPORT_COMPLETION = NotificationSpec("✅ Walk-Forward 比較レポート完了", COLOR_SUCCESS)
WALK_FORWARD_REPORT_WITH_ERRORS = NotificationSpec("⚠️ Walk-Forward 比較レポート完了", COLOR_CAUTION)
DB_MAINTENANCE_COMPLETION = NotificationSpec("✅ DB メンテナンス完了", COLOR_SUCCESS)
DB_MAINTENANCE_ERROR = NotificationSpec("❌ DB メンテナンス失敗", COLOR_ERROR)
PRE_CLOSE_ALERT = NotificationSpec("📋 引け前ポジション再評価", COLOR_INFO)
SHADOW_EVALUATION_CHALLENGER_WINS = NotificationSpec("🏆 A/Bテスト: Challenger 昇格候補", COLOR_WARNING)
SHADOW_EVALUATION_NO_WINNER = NotificationSpec("ℹ️ A/Bテスト: 評価完了", COLOR_INFO)


def get_daily_order_spec(*, trading_stopped: bool) -> NotificationSpec:
    return DAILY_ORDER_STOPPED if trading_stopped else DAILY_ORDER_COMPLETION


def get_optimization_spec(*, failed: int) -> NotificationSpec:
    return OPTIMIZATION_COMPLETION_WITH_ERRORS if failed > 0 else OPTIMIZATION_COMPLETION


def get_walk_forward_report_spec(*, failed_count: int) -> NotificationSpec:
    return WALK_FORWARD_REPORT_WITH_ERRORS if failed_count > 0 else WALK_FORWARD_REPORT_COMPLETION
