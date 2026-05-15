"""
定期実行スケジューラ

APSchedulerを使用して各タスクを定期実行する。
Discord Botと同時に起動し、1プロセスで全ジョブを管理する。

使い方:
  py run_scheduler.py                  # スケジューラのみ起動
  py run_scheduler.py --with-bot       # Discord Bot と同時起動
  py run_scheduler.py --run-now daily        # daily パイプラインを即時実行して終了
  py run_scheduler.py --run-now weekly       # weekly パイプラインを即時実行して終了
  py run_scheduler.py --run-now paper        # 自動発注→約定処理を順番に即時実行して終了
  py run_scheduler.py --run-now optimization # 全銘柄バックテスト最適化を即時実行して終了
  py run_scheduler.py --run-now wf_report    # Walk-Forward比較レポートを即時実行して終了
    py run_scheduler.py --run-now drift        # ドリフト監視と自動再学習を即時実行して終了
"""

import argparse
import sys
from datetime import datetime

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

from src.utils.logger import get_logger

# ログ設定（logger.pyに集約、basicConfigは不要）
logger = get_logger(__name__)


def _build_queue_manager():
    """キュー方式ジョブ管理を初期化する"""
    from src.orchestration.scheduler_queue import SchedulerQueueManager

    manager = SchedulerQueueManager(SCHEDULE_CONFIG)
    logger.info(f"キュー状態ログ: {manager.state_file_path}")
    return manager


# ── ジョブ定義 ────────────────────────────────────────
def job_daily_pipeline():
    """毎日実行: データ取得 → 予測 → Discord通知用CSV出力"""
    from src.orchestration.scheduler import run_daily_pipeline

    run_daily_pipeline()


def job_weekly_model_training():
    """週次実行: 統合モデル再学習 + 予測精度チェック"""
    from src.orchestration.scheduler import run_weekly_training

    run_weekly_training()


def job_weekly_report():
    """週次実行: パフォーマンスレポート Discord 送信"""
    from src.orchestration.scheduler import run_weekly_report

    run_weekly_report()


def job_daily_auto_order():
    """毎営業日 8:50 - ペーパートレード注文発注（前日予測シグナルを使用）"""
    from src.orchestration.scheduler import run_daily_auto_order

    run_daily_auto_order()


def job_daily_settle_orders():
    """毎営業日 9:05 - ペーパートレード pending 注文を当日始値で約定処理"""
    from src.orchestration.scheduler import run_daily_settle_orders

    run_daily_settle_orders()


def job_daily_paper_trade_report():
    """毎営業日 15:30 - ペーパートレードポジション・損益レポートを Discord に送信"""
    from src.orchestration.scheduler import run_daily_paper_trade_report

    run_daily_paper_trade_report()


def job_weekly_optimization():
    """週次実行: 全銘柄バックテスト最適化 → 最適パラメータ更新"""
    from src.orchestration.scheduler import run_weekly_optimization

    run_weekly_optimization()


def job_weekly_walk_forward_report():
    """週次実行: Walk-Forward 比較レポート生成"""
    from src.orchestration.scheduler import run_weekly_walk_forward_report

    run_weekly_walk_forward_report()


def job_daily_drift_check():
    """毎営業日実行: ドリフト監視と閾値超過銘柄の再学習"""
    from src.orchestration.scheduler import run_daily_drift_check

    run_daily_drift_check()


def job_weekly_db_maintenance():
    """週次実行 (土曜 03:00): DuckDB CHECKPOINT + VACUUM"""
    from src.orchestration.scheduler import run_weekly_db_maintenance

    run_weekly_db_maintenance()


def job_weekly_rule_evaluation():
    """週次実行 (日曜 02:00): 全銘柄 × 全ルールのバックテスト評価"""
    from src.orchestration.scheduler import run_weekly_rule_evaluation

    run_weekly_rule_evaluation()


def job_daily_rule_signals():
    """日次実行 (平日 16:30): 最優秀ルールによるシグナル生成 + ペーパートレード発注"""
    from src.orchestration.scheduler import run_daily_rule_signals

    run_daily_rule_signals()


def job_pre_close_alert():
    """毎営業日 15:00 - 引け前ポジション再評価アラート送信"""
    from src.orchestration.scheduler import run_pre_close_alert

    run_pre_close_alert()


def job_daily_backup():
    """毎日 03:30 - DuckDB バックアップ（最大5世代保持）"""
    from src.orchestration.scheduler import run_daily_backup

    run_daily_backup()


def job_monthly_report():
    """毎月1日 03:00 - 月次KPIレポート生成・保存・Discord通知"""
    from src.orchestration.scheduler import run_monthly_report_job

    run_monthly_report_job()


# ── イベントリスナー ──────────────────────────────────
def _job_listener(event):
    """ジョブ実行結果のログ出力"""
    if event.exception:
        logger.error(f"ジョブ '{event.job_id}' が異常終了しました: {event.exception}")
    else:
        logger.info(f"ジョブ '{event.job_id}' が正常終了しました")


# ── スケジュール設定 ──────────────────────────────────
# 時刻はすべてJST（Asia/Tokyo）想定
SCHEDULE_CONFIG = {
    "daily_pipeline": {
        "func": job_daily_pipeline,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 7,
        "minute": 30,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 3,
        "description": "毎営業日 07:30 - データ取得 → 予測",
    },
    "weekly_model_training": {
        "func": job_weekly_model_training,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "tue",
        "hour": 1,
        "minute": 0,
        "recovery_delay_minutes": 15,
        "max_executions_per_period": 2,
        "description": "毎週火曜 01:00 - 統合モデル再学習 + 精度チェック",
    },
    "weekly_report": {
        "func": job_weekly_report,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "thu",
        "hour": 2,
        "minute": 0,
        "recovery_delay_minutes": 15,
        "max_executions_per_period": 2,
        "description": "毎週木曜 02:00 - パフォーマンスレポート送信",
    },
    "daily_auto_order": {
        "func": job_daily_auto_order,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 8,
        "minute": 50,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 1,
        "description": "毎営業日 08:50 - 自動発注（ペーパートレード）",
    },
    "daily_settle_orders": {
        "func": job_daily_settle_orders,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 9,
        "minute": 5,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 1,
        "description": "毎営業日 09:05 - pending 注文の約定処理（ペーパートレード）",
    },
    "daily_paper_trade_report": {
        "func": job_daily_paper_trade_report,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 15,
        "minute": 30,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 1,
        "description": "毎営業日 15:30 - ペーパートレード ポジション・損益レポート",
    },
    "weekly_optimization": {
        "func": job_weekly_optimization,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "wed",
        "hour": 1,
        "minute": 0,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 1,
        "description": "毎週水曜 01:00 - 全銘柄バックテスト最適化",
    },
    "weekly_walk_forward_report": {
        "func": job_weekly_walk_forward_report,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "thu",
        "hour": 1,
        "minute": 0,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 1,
        "description": "毎週木曜 01:00 - Walk-Forward 比較レポート生成",
    },
    "daily_drift_check": {
        "func": job_daily_drift_check,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 16,
        "minute": 0,
        "recovery_delay_minutes": 20,
        "max_executions_per_period": 1,
        "description": "毎営業日 16:00 - ドリフト監視と再学習トリガー",
    },
    "weekly_db_maintenance": {
        "func": job_weekly_db_maintenance,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sat",
        "hour": 3,
        "minute": 0,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 1,
        "description": "毎週土曜 03:00 - DuckDB CHECKPOINT + VACUUM",
    },
    "weekly_rule_evaluation": {
        "func": job_weekly_rule_evaluation,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sun",
        "hour": 2,
        "minute": 0,
        "recovery_delay_minutes": 60,
        "max_executions_per_period": 1,
        "description": "毎週日曜 02:00 - 全銘柄ルールバックテスト評価",
    },
    "daily_rule_signals": {
        "func": job_daily_rule_signals,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 16,
        "minute": 30,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 1,
        "description": "毎営業日 16:30 - ルールシグナル生成 + ペーパートレード発注",
    },
    "pre_close_alert": {
        "func": job_pre_close_alert,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 15,
        "minute": 0,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 1,
        "description": "毎営業日 15:00 - 引け前ポジション再評価アラート",
    },
    "daily_backup": {
        "func": job_daily_backup,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": 3,
        "minute": 30,
        "recovery_delay_minutes": 60,
        "max_executions_per_period": 1,
        "description": "毎日 03:30 - DuckDB バックアップ（最大5世代保持）",
    },
    "monthly_report": {
        "func": job_monthly_report,
        "trigger": "cron",
        "period": "monthly",
        "day_of_week": "mon-sun",
        "day": 1,
        "hour": 3,
        "minute": 0,
        "recovery_delay_minutes": 60,
        "max_executions_per_period": 1,
        "description": "毎月1日 03:00 - 月次KPIレポート生成・保存・Discord通知",
    },
}


def _register_jobs(scheduler, queue_manager):
    """スケジューラにジョブを登録する"""
    for job_id, config in SCHEDULE_CONFIG.items():

        def _managed_runner(_job_id=job_id):
            queue_manager.run_job(_job_id, reason="scheduled")

        cron_kwargs = {
            "trigger": config["trigger"],
            "day_of_week": config["day_of_week"],
            "hour": config["hour"],
            "minute": config["minute"],
        }
        if "day" in config:
            cron_kwargs["day"] = config["day"]
        scheduler.add_job(
            _managed_runner,
            **cron_kwargs,
            id=job_id,
            name=config["description"],
            misfire_grace_time=3600,  # 1時間以内なら遅延実行
            coalesce=True,  # 複数回分溜まっても1回だけ実行
            max_instances=1,
        )
        logger.info(f"ジョブ登録: {job_id} - {config['description']}")

    scheduler.add_job(
        lambda: _poll_recovery_jobs(queue_manager),
        trigger="interval",
        minutes=5,
        id="recovery_poller",
        name="未実行補完ポーリング",
        coalesce=True,
        max_instances=1,
    )
    logger.info("ジョブ登録: recovery_poller - 5分間隔で未実行ジョブを補完")


def _poll_recovery_jobs(queue_manager):
    """所定時刻を過ぎても未実行のジョブを補完実行する（前日分も含む）"""
    now = queue_manager.now()

    for job_id in SCHEDULE_CONFIG.keys():
        # 当日分の補完
        if queue_manager.should_recover(job_id, now=now):
            logger.warning(f"補完実行を開始: {job_id}")
            queue_manager.run_job(job_id, reason="recovery")
            continue

        # 前日分の補完（コンテナ再起動などで当日スケジュール前に前日分が欠落した場合）
        missed_periods = queue_manager.get_missed_past_periods(job_id, lookback_days=2, now=now)
        for missed_period_key in missed_periods:
            logger.warning(f"前日分補完実行を開始: {job_id} (period={missed_period_key})")
            queue_manager.run_job(job_id, reason="recovery_backfill", period_key=missed_period_key)


def _print_schedule():
    """登録済みスケジュールを表示する"""
    print("\n" + "=" * 60)
    print("  StockFixer スケジューラ")
    print("=" * 60)
    for job_id, config in SCHEDULE_CONFIG.items():
        print(f"  {job_id:<30} {config['description']}")
    print("=" * 60)
    print(f"  起動日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  終了するには Ctrl+C を押してください")
    print("=" * 60 + "\n")


def _start_health_server() -> None:
    """ヘルスチェックサーバーをデーモンスレッドで起動する"""
    import os

    from src.api.health import start_health_server

    port = int(os.getenv("HEALTH_PORT", "5100"))
    start_health_server(port=port)


def run_with_bot():
    """Discord Botとスケジューラを同時に起動する"""
    from src.reporting.discord.discord_bot import TOKEN, bot

    if not TOKEN:
        logger.error("DISCORD_BOT_TOKEN環境変数が設定されていません。")
        sys.exit(1)

    # BackgroundSchedulerを使用（Botのイベントループをブロックしない）
    scheduler = BackgroundScheduler(timezone="Asia/Tokyo")
    queue_manager = _build_queue_manager()
    _register_jobs(scheduler, queue_manager)
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    _print_schedule()
    print("  Discord Bot と同時起動中\n")

    _start_health_server()
    scheduler.start()
    try:
        bot.run(TOKEN)
    finally:
        scheduler.shutdown()


def run_scheduler_only():
    """スケジューラのみ起動する"""
    scheduler = BlockingScheduler(timezone="Asia/Tokyo")
    queue_manager = _build_queue_manager()
    _register_jobs(scheduler, queue_manager)
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    _print_schedule()
    _start_health_server()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("スケジューラを停止します。")


def run_now(pipeline: str):
    """指定パイプラインを即時実行する（テスト・手動実行用）"""
    queue_manager = _build_queue_manager()

    if pipeline == "daily":
        queue_manager.run_job("daily_pipeline", reason="manual", force=True)
    elif pipeline == "weekly":
        queue_manager.run_job("weekly_model_training", reason="manual", force=True)
    elif pipeline == "paper":
        logger.info("=== ペーパートレード テスト実行開始 ===")
        queue_manager.run_job("daily_auto_order", reason="manual", force=True)
        logger.info("--- 注文発注完了 → 約定処理へ ---")
        queue_manager.run_job("daily_settle_orders", reason="manual", force=True)
        logger.info("=== ペーパートレード テスト実行完了 ===")
    elif pipeline == "paper_report":
        queue_manager.run_job("daily_paper_trade_report", reason="manual", force=True)
    elif pipeline == "optimization":
        queue_manager.run_job("weekly_optimization", reason="manual", force=True)
    elif pipeline == "wf_report":
        queue_manager.run_job("weekly_walk_forward_report", reason="manual", force=True)
    elif pipeline == "watchlist":
        queue_manager.run_job("weekly_watchlist_refresh", reason="manual", force=True)
    elif pipeline == "drift":
        queue_manager.run_job("daily_drift_check", reason="manual", force=True)
    elif pipeline == "db_maintenance":
        queue_manager.run_job("weekly_db_maintenance", reason="manual", force=True)
    elif pipeline == "rule_evaluate":
        queue_manager.run_job("weekly_rule_evaluation", reason="manual", force=True)
    elif pipeline == "rule_signals":
        queue_manager.run_job("daily_rule_signals", reason="manual", force=True)
    elif pipeline == "pre_close_alert":
        queue_manager.run_job("pre_close_alert", reason="manual", force=True)
    elif pipeline == "backup":
        queue_manager.run_job("daily_backup", reason="manual", force=True)
    elif pipeline == "monthly_report":
        queue_manager.run_job("monthly_report", reason="manual", force=True)
    else:
        print(f"不明なパイプライン: {pipeline}")
        print(
            "使用可能: daily, weekly, paper, paper_report, optimization,"
            " wf_report, watchlist, drift, db_maintenance, rule_evaluate, rule_signals,"
            " pre_close_alert"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="StockFixer 定期実行スケジューラ")
    parser.add_argument(
        "--with-bot",
        action="store_true",
        help="Discord Bot と同時に起動する",
    )
    parser.add_argument(
        "--run-now",
        type=str,
        choices=[
            "daily",
            "weekly",
            "paper",
            "paper_report",
            "optimization",
            "wf_report",
            "watchlist",
            "drift",
            "db_maintenance",
            "rule_evaluate",
            "rule_signals",
            "pre_close_alert",
            "backup",
            "monthly_report",
        ],
        help=(
            "指定パイプラインを即時実行して終了する（テスト用）。"
            "paper=自動発注→約定処理を順番に実行、"
            "paper_report=ポジション・損益レポート送信、"
            "optimization=全銘柄最適化、"
            "wf_report=Walk-Forward比較レポート生成、"
            "watchlist=ウォッチリスト自動更新、"
            "drift=ドリフト監視と自動再学習"
        ),
    )
    args = parser.parse_args()

    if args.run_now:
        run_now(args.run_now)
    elif args.with_bot:
        run_with_bot()
    else:
        run_scheduler_only()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"スケジューラ 異常終了: {e}", exc_info=True)
        sys.exit(1)
