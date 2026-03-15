"""
定期実行スケジューラ

APSchedulerを使用して各タスクを定期実行する。
Discord Botと同時に起動し、1プロセスで全ジョブを管理する。

使い方:
  py run_scheduler.py                  # スケジューラのみ起動
  py run_scheduler.py --with-bot       # Discord Bot と同時起動
  py run_scheduler.py --run-now daily  # daily パイプラインを即時実行して終了
  py run_scheduler.py --run-now weekly # weekly パイプラインを即時実行して終了
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
    from src.services.scheduler_queue import SchedulerQueueManager

    manager = SchedulerQueueManager(SCHEDULE_CONFIG)
    logger.info(f"キュー状態ログ: {manager.state_file_path}")
    return manager


# ── ジョブ定義 ────────────────────────────────────────
def job_daily_pipeline():
    """毎日実行: データ取得 → 予測 → Discord通知用CSV出力"""
    from src.services.scheduler_pipeline import run_daily_pipeline

    run_daily_pipeline()


def job_weekly_model_training():
    """週次実行: 統合モデル再学習 + 予測精度チェック"""
    from src.services.scheduler_pipeline import run_weekly_training

    run_weekly_training()


def job_weekly_report():
    """週次実行: パフォーマンスレポート Discord 送信"""
    from src.services.scheduler_pipeline import run_weekly_report

    run_weekly_report()


def job_daily_auto_order():
    """毎営業日 8:50 - ペーパートレード注文発注（前日予測シグナルを使用）"""
    from src.services.scheduler_pipeline import run_daily_auto_order

    run_daily_auto_order()


def job_daily_settle_orders():
    """毎営業日 9:05 - ペーパートレード pending 注文を当日始値で約定処理"""
    from src.services.scheduler_pipeline import run_daily_settle_orders

    run_daily_settle_orders()


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
        "hour": 19,
        "minute": 0,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 3,
        "description": "毎営業日 19:00 - データ取得 → 予測",
    },
    "weekly_model_training": {
        "func": job_weekly_model_training,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sat",
        "hour": 3,
        "minute": 0,
        "recovery_delay_minutes": 15,
        "max_executions_per_period": 2,
        "description": "毎週土曜 03:00 - 統合モデル再学習 + 精度チェック",
    },
    "weekly_report": {
        "func": job_weekly_report,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sat",
        "hour": 4,
        "minute": 0,
        "recovery_delay_minutes": 15,
        "max_executions_per_period": 2,
        "description": "毎週土曜 04:00 - パフォーマンスレポート送信",
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
}


def _register_jobs(scheduler, queue_manager):
    """スケジューラにジョブを登録する"""
    for job_id, config in SCHEDULE_CONFIG.items():

        def _managed_runner(_job_id=job_id):
            queue_manager.run_job(_job_id, reason="scheduled")

        scheduler.add_job(
            _managed_runner,
            trigger=config["trigger"],
            day_of_week=config["day_of_week"],
            hour=config["hour"],
            minute=config["minute"],
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
    """所定時刻を過ぎても未実行のジョブを補完実行する"""
    now = queue_manager.now()

    for job_id in SCHEDULE_CONFIG.keys():
        if not queue_manager.should_recover(job_id, now=now):
            continue
        logger.warning(f"補完実行を開始: {job_id}")
        queue_manager.run_job(job_id, reason="recovery")


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


def run_with_bot():
    """Discord Botとスケジューラを同時に起動する"""
    from src.api.discord_bot import TOKEN, bot

    if not TOKEN:
        logger.error("DISCORD_BOT_TOKEN環境変数が設定されていません。")
        sys.exit(1)

    # BackgroundSchedulerを使用（Botのイベントループをブロックしない）
    scheduler = BackgroundScheduler()
    queue_manager = _build_queue_manager()
    _register_jobs(scheduler, queue_manager)
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    _print_schedule()
    print("  Discord Bot と同時起動中\n")

    scheduler.start()
    try:
        bot.run(TOKEN)
    finally:
        scheduler.shutdown()


def run_scheduler_only():
    """スケジューラのみ起動する"""
    scheduler = BlockingScheduler()
    queue_manager = _build_queue_manager()
    _register_jobs(scheduler, queue_manager)
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    _print_schedule()

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
    else:
        print(f"不明なパイプライン: {pipeline}")
        print("使用可能: daily, weekly")
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
        choices=["daily", "weekly"],
        help="指定パイプラインを即時実行して終了する（テスト用）",
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
