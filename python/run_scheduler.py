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
import logging
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# ── ロギング設定 ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")


# ── ジョブ定義 ────────────────────────────────────────
def job_daily_pipeline():
    """毎日実行: データ取得 → 予測 → Discord通知用CSV出力"""
    from src.services.scheduler_pipeline import run_daily_pipeline
    run_daily_pipeline()


def job_weekly_model_training():
    """週次実行: 統合モデル再学習"""
    from src.services.scheduler_pipeline import run_weekly_training
    run_weekly_training()


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
        "day_of_week": "mon-fri",
        "hour": 19,
        "minute": 0,
        "description": "毎営業日 19:00 - データ取得 → 予測",
    },
    "weekly_model_training": {
        "func": job_weekly_model_training,
        "trigger": "cron",
        "day_of_week": "sat",
        "hour": 3,
        "minute": 0,
        "description": "毎週土曜 03:00 - 統合モデル再学習",
    },
}


def _register_jobs(scheduler):
    """スケジューラにジョブを登録する"""
    for job_id, config in SCHEDULE_CONFIG.items():
        scheduler.add_job(
            config["func"],
            trigger=config["trigger"],
            day_of_week=config["day_of_week"],
            hour=config["hour"],
            minute=config["minute"],
            id=job_id,
            name=config["description"],
            misfire_grace_time=3600,  # 1時間以内なら遅延実行
            coalesce=True,           # 複数回分溜まっても1回だけ実行
        )
        logger.info(f"ジョブ登録: {job_id} - {config['description']}")


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
    from src.api.discord_bot import bot, TOKEN

    if not TOKEN:
        logger.error("DISCORD_BOT_TOKEN環境変数が設定されていません。")
        sys.exit(1)

    # BackgroundSchedulerを使用（Botのイベントループをブロックしない）
    scheduler = BackgroundScheduler()
    _register_jobs(scheduler)
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
    _register_jobs(scheduler)
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    _print_schedule()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("スケジューラを停止します。")


def run_now(pipeline: str):
    """指定パイプラインを即時実行する（テスト・手動実行用）"""
    if pipeline == "daily":
        job_daily_pipeline()
    elif pipeline == "weekly":
        job_weekly_model_training()
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
    main()
