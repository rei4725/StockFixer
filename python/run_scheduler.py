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
  py run_scheduler.py --run-now multibagger  # 月次 multibagger スクリーン + 保有/撤退アラートを即時実行して終了
"""

import argparse
import sys
from datetime import datetime

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

from src.utils.heartbeat import SCHEDULER_ALIVE_SLUG, ping_heartbeat
from src.utils.logger import get_logger

# ログ設定（logger.pyに集約、basicConfigは不要）
logger = get_logger(__name__)


def _build_queue_manager():
    """キュー方式ジョブ管理を初期化する"""
    from src.orchestration.jobs.common import notify_job_failure
    from src.orchestration.scheduler_queue import SchedulerQueueManager

    manager = SchedulerQueueManager(SCHEDULE_CONFIG, error_notifier=notify_job_failure)
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


def job_daily_horizon_exit():
    """毎営業日 9:00 - 期限を過ぎたペーパーの保有のホライズン決済（8:50 の回の漏れを補う）"""
    from src.orchestration.scheduler import run_daily_horizon_exit

    run_daily_horizon_exit()


def job_daily_settle_orders():
    """毎営業日 9:05 - ペーパートレード pending 注文を当日始値で約定処理"""
    from src.orchestration.jobs.settle_notify import raise_if_settle_errors
    from src.orchestration.scheduler import run_daily_settle_orders

    raise_if_settle_errors(run_daily_settle_orders())


def job_daily_settle_orders_retry():
    """据え置いた pending 注文の約定のやり直し（毎営業日 09:15〜09:55）"""
    from src.orchestration.jobs.settle_retry import run_daily_settle_retry

    run_daily_settle_retry()


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


def job_weekly_volatility_sizing_compare():
    """週次実行 (日曜 13:00): 値動きに合わせた量と損切りの線の比べを results/ に書く"""
    from src.orchestration.scheduler import run_weekly_volatility_sizing_compare

    run_weekly_volatility_sizing_compare()


def job_daily_rule_signals():
    """日次実行 (平日 16:30): 最優秀ルールによるシグナル生成 + ペーパートレード発注"""
    from src.orchestration.scheduler import run_daily_rule_signals

    run_daily_rule_signals()


def job_daily_account_shadow_report():
    """毎営業日 17:00 - 口座アプリの影の運用の比べの日報（段3）"""
    from src.orchestration.scheduler import run_daily_account_shadow_report

    run_daily_account_shadow_report()


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


def job_monthly_multibagger():
    """毎月1日 04:00 - multibagger 月次スクリーン + 保有/撤退アラート"""
    from src.orchestration.multibagger_job import (
        raise_if_multibagger_failed,
        run_monthly_multibagger_screen,
    )

    raise_if_multibagger_failed(run_monthly_multibagger_screen())


def job_nightly_strategy_factory():
    """毎日 05:00 - 戦略ファクトリー夜間バッチ（FACTORY_ENABLED=false ならスキップ）"""
    from src.orchestration.scheduler import run_nightly_strategy_factory

    run_nightly_strategy_factory()


def job_strategy_promotion_check():
    """2時間ごと(毎時30分) - 戦略ファクトリー由来マージPRの昇格記録（STRATEGY_PROMOTION_CHECK_ENABLEDに従う）"""
    from src.orchestration.scheduler import run_strategy_promotion_check

    run_strategy_promotion_check()


def job_allocation_rebalance():
    """配分戦略(TQQQ/短期債)ペーパートレード実行（毎日06:00にリバランス期日到来をチェック）"""
    from src.orchestration.scheduler import run_allocation_rebalance_job

    run_allocation_rebalance_job()


def job_regime_leverage_weekly():
    """レジームレバレッジ戦略(SPY・レバレッジ2.0倍)の週次判定（毎週土曜06:30）"""
    from src.orchestration.scheduler import run_regime_leverage_weekly_job

    run_regime_leverage_weekly_job()


def job_regime_leverage_daily_margin():
    """レジームレバレッジ戦略の日次マージンコールチェック（毎日07:30）"""
    from src.orchestration.scheduler import run_regime_leverage_daily_margin_job

    run_regime_leverage_daily_margin_job()


def job_nightly_backtest_promotion():
    """毎日 22:00 - 予想のコードが変わったらバックテストで候補を評価し暫定昇格"""
    from src.orchestration.scheduler import (
        raise_if_backtest_promotion_failed,
        run_backtest_promotion,
    )

    raise_if_backtest_promotion_failed(run_backtest_promotion())


# ── イベントリスナー ──────────────────────────────────
def _job_listener(event):
    """ジョブ実行結果のログ出力"""
    if event.exception:
        logger.error(f"ジョブ '{event.job_id}' が異常終了しました: {event.exception}")
    else:
        logger.info(f"ジョブ '{event.job_id}' が正常終了しました")


# ── スケジュール設定 ──────────────────────────────────
# 時刻はすべてJST（Asia/Tokyo）想定
# backfill: True のジョブだけ、前日分の取りこぼしを翌日に補完する。発注・約定・その日の
# 相場の知らせなど、日をまたいでやり直すと当日分と二重になるジョブには付けない（#825）。
# recovery_until: (hour, minute) のジョブは、当日分の補完をその時刻で打ち切る（#835）。
# 定時の回も、遅れてその時刻を過ぎて走り出したら飛ばす（#1298）。
# max_retries: 0 のジョブは、落ちても同じ回の中で頭から呼び直さない。発注のジョブは、やり直しの回が
# 前の回の出した注文を知らずに同じ注文を出し直すため、これを付ける（#836）。
# 8:50 の発注は失敗の回も補完しない。16:30 のルールの発注だけは、約定待ちの BUY と SELL を飛ばし
# 二重にならないため、上限と recovery_min_gap_minutes で失敗の回を1回だけ補完する（#1676）。
# rerun_if_interrupted: True のジョブは、前の起動のプロセスで消えた回（デプロイの起き直しなど）を
# 1期間に1回まで回数に数えずに補完する。ほかの発注のジョブと週の学習（#807）には付けない（#946）。
# ホライズン決済は約定待ちの決済を飛ばし二重にならないため、発注のジョブでも付ける（#1321）。
# 16:30 のルールの発注も、約定待ちの BUY と SELL を飛ばし二重にならないため付ける（#1652）。
# max_interrupted_reruns_per_period: その補完の1回までを、そのジョブだけ増やす。週の最適化は
# 全銘柄に約29時間かかりデプロイの起き直しで週に何度も消え、済んだ銘柄を飛ばして続きから進むため（#1522）。
# 上限の40は、起動が1時間に1回ほどの週でも全銘柄の約29時間に届く数である（#1534）。
# recover_rest_of_week: True の週次ジョブは、定時の曜日より後の同じ ISO 週の日も、その週の成功が無く
# 上限が残っていれば補完する。定時の曜日より前の日は補完しない。全銘柄が1週のうちに終わるため（#1534）。
# long_recovery: True のジョブは、補完を専用のポーリング（recovery_long_job_poller）で判定する。
# 補完が日をまたいで戻らなくても、締めの無いほかのジョブの補完をふさがないため（#1542）。
# 締め（recovery_until）のあるジョブには付けない。締めのポーリングが判定するので印は効かない。
# recovery_min_gap_minutes: 同じ期間の前の回の終わりからこの分数の間は補完しない。回数の上限を
# 短い障害の間に使い切らないため。欄の無いジョブは今までどおり間を空けない（#1607）。
# rerun_if_interrupted のジョブは消えた回の続きをすぐ進めるので、失敗の回だけから数える（#1638）。
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
        "backfill": True,
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
        # 間が無いと、1回の失敗が5〜6分なので約30分の障害で2回を使い切り、火曜のうちの補完も無くなる。
        # 失敗の回の4時間後なら 05:00 台に補完し、07:30 の日次の道の前に終わる。ロックを待ち切って
        # 落ちた回はもう3時間空いたので間を数えず、補完が 8:50 の発注の近くへずれないようにする（#1682）
        "recovery_min_gap_minutes": 240,
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
        # 間が無いと、DB の障害の回は2分ほどで落ちるので約25分の障害で2回を使い切り、紙の週の成績と
        # 週の精度の記録がその週だけ欠ける（翌週の回は抜けた週を書かない）。失敗の回の4時間後なら
        # 06:00 台に補完し、07:30 の日次の道の前に終わる（#1689）
        "recovery_min_gap_minutes": 240,
        "description": "毎週木曜 02:00 - パフォーマンスレポート送信",
        "rerun_if_interrupted": True,
    },
    "daily_auto_order": {
        "func": job_daily_auto_order,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 8,
        "minute": 50,
        "recovery_delay_minutes": 10,
        # 当日分の補完は 09:30 まで。ペーパーは寄り付きの値で約定させるため、寄りから遅れるほど
        # 先読みになり、引けの後の発注は翌朝の定時の回と重なる（#835）
        "recovery_until": (9, 30),
        "max_executions_per_period": 1,
        "max_retries": 0,
        "description": "毎営業日 08:50 - 自動発注（ペーパートレード）",
    },
    # 8:50 の回がデプロイの起き直しなどでホライズン決済の前に消えても、同じ朝のうちに決済し直す。
    # 8:50 の回が出した約定待ちの SELL・SHORT_COVER がある銘柄は飛ばすため重ならない。9:05 の
    # 約定より前に出せば当日始値で約定し、遅れても 9:15〜9:55 の約定のやり直しが拾う（#1321）
    "daily_horizon_exit": {
        "func": job_daily_horizon_exit,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 9,
        "minute": 0,
        "recovery_delay_minutes": 5,
        # 締めは 8:50 の発注と同じ 09:30。寄りから遅れるほど先読みになる（#835）
        "recovery_until": (9, 30),
        "max_executions_per_period": 1,
        "max_retries": 0,
        "description": "毎営業日 09:00 - ホライズン決済（ペーパートレード）",
        "rerun_if_interrupted": True,
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
    # 9:05 に当日の行が無く据え置いた注文を、同じ朝のうちに当日始値で約定し直す。約定待ちが無ければ
    # 約定の処理は動かさない。minute が int でないので当日分の補完はしない（#1008）
    "daily_settle_orders_retry": {
        "func": job_daily_settle_orders_retry,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 9,
        "minute": "15,25,35,45,55",
        "max_executions_per_period": 5,
        "max_retries": 0,
        "description": "毎営業日 09:15〜09:55 - 据え置いた pending 注文の約定のやり直し（ペーパートレード）",
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
        # 結果が空か失敗が線を超えた回の後に、同じ週の補完を1回通す（#1581）
        "max_executions_per_period": 2,
        # 間が無いと、数時間の DB の障害の中で2回を使い切り、丸1週間の発注が古い閾値と SL/TP で動く。
        # 失敗の回の12時間後に補完し、丸半日の障害まで受ける（#1638）
        "recovery_min_gap_minutes": 720,
        "description": "毎週水曜 01:00 - 全銘柄バックテスト最適化",
        "rerun_if_interrupted": True,
        "max_interrupted_reruns_per_period": 40,
        "recover_rest_of_week": True,
        "long_recovery": True,
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
        "rerun_if_interrupted": True,
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
        # run_daily_pipeline() の step4 で同じ run_daily_drift_check() を毎朝実行済みのため、
        # 独立cronとしては自動起動しない（二重実行防止）。--run-now drift による手動実行のみ許可。
        "auto_schedule": False,
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
        "rerun_if_interrupted": True,
    },
    "weekly_rule_evaluation": {
        "func": job_weekly_rule_evaluation,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sun",
        "hour": 2,
        "minute": 0,
        "recovery_delay_minutes": 60,
        # 失敗の回は日曜のうちに1回だけ補完する。日曜が定時なので月曜へは持ち越さない（#1623）。
        # 1銘柄だけ続けて落ちる週も補完が1回走り、数分の評価と知らせが1通増える。増えるのは上限の2回まで
        "max_executions_per_period": 2,
        # 取得の障害が明けるのを待つ。間が無いと定時の1時間後に同じ障害へ当たって2回目を使い切る
        "recovery_min_gap_minutes": 240,
        "description": "毎週日曜 02:00 - 全銘柄ルールバックテスト評価",
        "rerun_if_interrupted": True,
    },
    # 発注の無い日曜の昼間。週の最適化の補完（水曜から週末まで）の後ろで、夜中の評価・バックアップ・
    # 工場・配分の定時とも2時間ごとの30分の昇格の確かめとも重ならない。比べは発注を変えない（#1547）
    "weekly_volatility_sizing_compare": {
        "func": job_weekly_volatility_sizing_compare,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sun",
        "hour": 13,
        "minute": 0,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 1,
        "max_retries": 0,
        "description": "毎週日曜 13:00 - 値動きに合わせた量と損切りの線の比べ",
        "rerun_if_interrupted": True,
    },
    # 16:30 の回がデプロイの起き直しなどで消えたら、同じ日のうちに1回だけ流し直す。その日の足の保有の
    # SELL を判定しないまま進めないため。消えた回の出した約定待ちの BUY と SELL は飛ばす（#1652）
    # 失敗で終わった回も同じ理由で、間を空けて同じ日に1回だけ補完する。同じ回の中では呼び直さず、
    # 間は短い障害が明けるのを待つ。補完の回も失敗したら、その日はもう補完しない（#1676）
    "daily_rule_signals": {
        "func": job_daily_rule_signals,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 16,
        "minute": 30,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 2,
        "max_retries": 0,
        "recovery_min_gap_minutes": 240,
        "description": "毎営業日 16:30 - ルールシグナル生成 + ペーパートレード発注",
        "rerun_if_interrupted": True,
    },
    "daily_account_shadow_report": {
        "func": job_daily_account_shadow_report,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 17,
        "minute": 0,
        "recovery_delay_minutes": 10,
        "max_executions_per_period": 1,
        "description": "毎営業日 17:00 - 口座アプリの影の運用の比べの日報（段3）",
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
        "backfill": True,
        "rerun_if_interrupted": True,
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
        "rerun_if_interrupted": True,
    },
    "monthly_multibagger": {
        "func": job_monthly_multibagger,
        "trigger": "cron",
        "period": "monthly",
        "day_of_week": "mon-sun",
        "day": 1,
        "hour": 4,
        "minute": 0,
        "recovery_delay_minutes": 60,
        # 失敗の回は同じ月の補完に任せる（すぐの呼び直しはしない）
        "max_executions_per_period": 3,
        # 失敗は数秒で終わるので、間が無いと障害の1時間余りで3回を使い切る。12時間なら定時の後の
        # 16時ごろと翌朝4時ごろに補完し、丸1日の障害まで受ける。月の判定は1日遅れても差し支えない
        "recovery_min_gap_minutes": 720,
        "description": "毎月1日 04:00 - multibagger 月次スクリーン + 保有/撤退アラート",
    },
    "nightly_strategy_factory": {
        "func": job_nightly_strategy_factory,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": 5,
        "minute": 0,
        "recovery_delay_minutes": 60,
        "max_executions_per_period": 1,
        "description": "毎日 05:00 - 戦略ファクトリー夜間バッチ（jp/us 日替わり、#369）",
        "backfill": True,
    },
    "nightly_backtest_promotion": {
        "func": job_nightly_backtest_promotion,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": 22,
        "minute": 0,
        "recovery_delay_minutes": 60,
        "max_executions_per_period": 1,
        "description": "毎日 22:00 - 予想のコードが変わったらバックテストで候補を評価し暫定昇格",
    },
    "strategy_promotion_check": {
        "func": job_strategy_promotion_check,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": "*/2",
        "minute": 30,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 12,
        "description": "2時間ごと(毎時30分) - 戦略ファクトリー由来マージPRの昇格記録",
        "backfill": True,
    },
    "allocation_rebalance": {
        "func": job_allocation_rebalance,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": 6,
        "minute": 0,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 1,
        "description": "毎日 06:00 - 配分戦略(TQQQ/短期債)リバランス判定",
        # 初回建玉作成済み（2026-09-01）。以後は毎日この時刻にリバランス期日到来を
        # チェックする（未到来なら service.py 側の判定で即座にno-op）。
        "auto_schedule": True,
    },
    "regime_leverage_daily_margin": {
        "func": job_regime_leverage_daily_margin,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 7,
        "minute": 30,
        "recovery_delay_minutes": 30,
        "description": "毎日07:30 - レジームレバレッジ戦略(SPY)の日次マージンコールチェック",
        # 2026-09-06: --run-now で初回エントリー実行・日次/週次動作確認済み。
        # ユーザーの運用開始判断により自動実行を有効化（allocation_rebalanceと同じ手順）。
        "auto_schedule": True,
    },
    "regime_leverage_weekly": {
        "func": job_regime_leverage_weekly,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "sat",
        "hour": 6,
        "minute": 30,
        "recovery_delay_minutes": 30,
        # レジーム転換の決済はこの判定だけが見るため、落ちた回も土曜を逃した週も日曜まで補完する。
        # 2回目の判定は保有中なら決済の判定だけに進み、建玉は重ならない（#1544）
        "max_executions_per_period": 12,
        "recover_rest_of_week": True,
        # 間が無いと、失敗が約3分半でポーリングが5分おきなので12回を約1時間半で使い切る。4時間なら
        # 約4時間5分おきで、定時の後の10回が日曜の23時台まで届き、上限に余りが出る。足の遅れも回数を
        # 使うので、足が届いた後の次の補完は最長4時間後になる。週末は足が増えず、遅れても見る値段は同じ
        "recovery_min_gap_minutes": 240,
        "description": "毎週土曜06:30 - レジームレバレッジ戦略(SPY)のレジーム転換・新規エントリー判定",
        # 2026-09-06: --run-now で初回エントリー実行・日次/週次動作確認済み。
        # ユーザーの運用開始判断により自動実行を有効化（allocation_rebalanceと同じ手順）。
        "auto_schedule": True,
    },
}


def _register_jobs(scheduler, queue_manager):
    """スケジューラにジョブを登録する"""
    for job_id, config in SCHEDULE_CONFIG.items():
        if not config.get("auto_schedule", True):
            logger.info(f"ジョブ登録スキップ: {job_id}（auto_schedule=False、手動実行のみ）")
            continue

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

    # 外部死活監視（#496）: スケジューラの生存を5分間隔で通知する。補完の呼び出しが戻らない間も
    # 途切れないよう、補完のポーリングと別のジョブで送る（#1542）。長いジョブの負荷でスケジューラの
    # 目覚めが遅れても1回分を飛ばさないよう、遅れの猶予を既定の1秒より広げる
    scheduler.add_job(
        _ping_scheduler_alive,
        trigger="interval",
        minutes=5,
        id="scheduler_alive_ping",
        name="スケジューラの生存の通知",
        misfire_grace_time=60,
        coalesce=True,
        max_instances=1,
    )
    logger.info("ジョブ登録: scheduler_alive_ping - 5分間隔で生存を通知")

    scheduler.add_job(
        lambda: _poll_recovery_jobs(queue_manager, with_cutoff=False, long_recovery=False),
        trigger="interval",
        minutes=5,
        id="recovery_poller",
        name="未実行補完ポーリング",
        coalesce=True,
        max_instances=1,
    )
    logger.info("ジョブ登録: recovery_poller - 5分間隔で未実行ジョブを補完")

    # 締め（recovery_until）を持つジョブは、日次の道などの長い補完でふさがる recovery_poller と
    # 別のスレッドで判定する。ふさがっている間に補完の時刻を迎えても締めの内に補完するため（#1410）
    scheduler.add_job(
        lambda: _poll_recovery_jobs(queue_manager, with_cutoff=True),
        trigger="interval",
        minutes=1,
        id="recovery_cutoff_poller",
        name="締めのあるジョブの未実行補完ポーリング",
        coalesce=True,
        max_instances=1,
    )
    logger.info("ジョブ登録: recovery_cutoff_poller - 1分間隔で締めのあるジョブを補完")

    # long_recovery のジョブ（週の最適化）は、戻るまで日をまたぐ補完で recovery_poller をふさがないよう
    # 別のスレッドで判定する。締めの無いほかのジョブの補完を止めないため（#1542）
    scheduler.add_job(
        lambda: _poll_recovery_jobs(queue_manager, with_cutoff=False, long_recovery=True),
        trigger="interval",
        minutes=5,
        id="recovery_long_job_poller",
        name="長い補完のジョブの未実行補完ポーリング",
        coalesce=True,
        max_instances=1,
    )
    logger.info("ジョブ登録: recovery_long_job_poller - 5分間隔で長い補完のジョブを補完")


def _ping_scheduler_alive() -> None:
    """スケジューラの生存を外部死活監視へ通知する"""
    ping_heartbeat(SCHEDULER_ALIVE_SLUG)


def _poll_recovery_jobs(
    queue_manager, with_cutoff: bool | None = None, long_recovery: bool | None = None
):
    """所定時刻を過ぎても未実行のジョブを補完実行する（前日分も含む）

    with_cutoff が True なら締め（recovery_until）のあるジョブだけ、False なら締めの無いジョブだけ、
    None なら両方を見る。long_recovery も同じで、True なら長い補完の印のあるジョブだけ、False なら
    印の無いジョブだけを見る。同じジョブは1つのポーリングだけが判定する（#1410, #1542）。
    """
    # 締め（recovery_until）を持つジョブを先に回す。締めの無い長い補完（日次の道など）の後ろで
    # 自動発注の補完が締めを過ぎて飛ばされないため（#1395）。sorted は安定なので同じ組の中は今の順のまま。
    ordered = sorted(
        SCHEDULE_CONFIG.items(), key=lambda item: item[1].get("recovery_until") is None
    )
    for job_id, config in ordered:
        if not config.get("auto_schedule", True):
            continue
        has_cutoff = config.get("recovery_until") is not None
        if with_cutoff is not None and has_cutoff != with_cutoff:
            continue
        if long_recovery is not None and config.get("long_recovery", False) != long_recovery:
            continue

        # 1つのジョブが落ちても後のジョブ（発注・約定）の補完を同じ回で進める（#1186）。
        # 失敗の記録と知らせは run_job の中で済んでいる。
        try:
            # 前のジョブの長い補完で締め（recovery_until）を越えることがあるため、
            # 判定の直前に時刻を測り直す（#1286）
            _recover_job(queue_manager, job_id, queue_manager.now())
        except Exception:
            logger.error(f"補完実行に失敗: {job_id}（次のジョブへ進む）", exc_info=True)


def _recover_job(queue_manager, job_id: str, now) -> None:
    """1つのジョブの当日分、なければ前日分を補完実行する"""
    # 当日分の補完
    if queue_manager.should_recover(job_id, now=now):
        logger.warning(f"補完実行を開始: {job_id}")
        queue_manager.run_job(job_id, reason="recovery")
        return

    # 上限や締めで補完をやめた期間を知らせる。例外は返さない（#1647, #1674）
    from src.orchestration.scheduler_cap_notice import notify_recovery_cap

    notify_recovery_cap(queue_manager, job_id, now)

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
        # 週の更新は GitHub Actions（weekly-watchlist-refresh）へ移したため SCHEDULE_CONFIG に無い
        from src.orchestration.scheduler import run_weekly_watchlist_refresh

        run_weekly_watchlist_refresh()
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
    elif pipeline == "multibagger":
        queue_manager.run_job("monthly_multibagger", reason="manual", force=True)
    elif pipeline == "factory":
        queue_manager.run_job("nightly_strategy_factory", reason="manual", force=True)
    elif pipeline == "promotion_check":
        queue_manager.run_job("strategy_promotion_check", reason="manual", force=True)
    elif pipeline == "allocation_rebalance":
        queue_manager.run_job("allocation_rebalance", reason="manual", force=True)
    elif pipeline == "regime_leverage_daily_margin":
        queue_manager.run_job("regime_leverage_daily_margin", reason="manual", force=True)
    elif pipeline == "regime_leverage_weekly":
        queue_manager.run_job("regime_leverage_weekly", reason="manual", force=True)
    else:
        print(f"不明なパイプライン: {pipeline}")
        print(
            "使用可能: daily, weekly, paper, paper_report, optimization,"
            " wf_report, watchlist, drift, db_maintenance, rule_evaluate, rule_signals,"
            " pre_close_alert"
        )
        sys.exit(1)


def main():
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
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
            "multibagger",
            "factory",
            "promotion_check",
            "allocation_rebalance",
            "regime_leverage_daily_margin",
            "regime_leverage_weekly",
        ],
        help=(
            "指定パイプラインを即時実行して終了する（テスト用）。"
            "paper=自動発注→約定処理を順番に実行、"
            "paper_report=ポジション・損益レポート送信、"
            "optimization=全銘柄最適化、"
            "wf_report=Walk-Forward比較レポート生成、"
            "watchlist=ウォッチリスト自動更新、"
            "drift=ドリフト監視と自動再学習、"
            "factory=戦略ファクトリー夜間バッチ（FACTORY_ENABLED に従う）"
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
