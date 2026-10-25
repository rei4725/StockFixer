"""日次・場中スケジューラジョブ。

毎営業日に走るデータパイプライン・自動発注・約定処理・レポート・
ドリフト監視・バックアップ・ルールシグナルのオーケストレーション。
"""

from src.orchestration.jobs.accuracy_json import (
    append_accuracy_json_sectors,
    production_already_predicted_today,
    write_accuracy_json_without_sectors,
)
from src.orchestration.jobs.alert_streaks import (
    record_health_degraded_streak,
    record_loss_limit_streak,
)
from src.orchestration.jobs.alerting import (
    evaluate_and_notify_alerts,
    evaluate_output_invariants_stage,
    fetch_previous_run_stats,
)
from src.orchestration.jobs.backup_job import run_daily_backup  # noqa: F401 (re-export)
from src.orchestration.jobs.common import _handle_stage_error, _notify_stage_failure
from src.orchestration.jobs.drift_check import run_daily_drift_check
from src.orchestration.jobs.prediction_errors import notify_prediction_errors
from src.orchestration.jobs.unavailable_gates import (
    _raise_if_fetch_mostly_unavailable,
    _raise_if_rule_signals_mostly_unavailable,
)
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_pipeline():
    """
    毎日実行: データ取得 → 精度チェック → 予測 → 出力invariant評価
             → Challenger shadow 予測 → ドリフト監視 → Discord通知 → 運用アラート評価

    流れ:
        1. 全マーケットのデータを取得（バッチ）
        2. 前日予測の精度チェック: production / challenger それぞれ記録（非致命的）
            prediction_results は Delete-Insert のスナップショットテーブルなので、
            3 で当日分を保存する前＝前日分がまだ残っている段階で読む必要がある
            （後段で読むと 3 の保存で前日分が上書き消滅済みで永久に0件採点になる）。
            採点の後、results/accuracy/YYYY-MM-DD.json に的中率を書く（非致命的）。
            業種の内訳は yfinance を引くため、6 の後に書き足す（#843）。
        3. Top10/Worst10の予測を実行（production）
        3.1. 出力 invariant 評価（非致命的。結果は 6 で通知する）
        3.5. Challenger shadow 予測（モデルが存在する場合のみ、非致命的）
        4. 日次ドリフトチェック（閾値超過銘柄を自動再学習）
        5. Discord通知
        6. 運用アラート評価（NF-303。条件成立時のみ発報）
    """
    logger.info("=== 日次パイプライン開始 ===")

    from src.reporting.discord.discord_utils import (
        send_accuracy_summary,
        send_daily_pipeline_completion,
        send_daily_pipeline_error,
    )

    # 1. データ取得（CRITICAL: 失敗時はパイプライン停止 + Discord通知）
    logger.info("[1/6] データ取得開始")
    from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter
    from src.market_data.pipeline import run_batch_pipeline
    from src.watchlist.batch_runner import load_target_symbols

    try:
        tasks = load_target_symbols()
        batch_result = run_batch_pipeline(tasks, notification_port=DiscordNotificationAdapter())
        _raise_if_fetch_mostly_unavailable(len(tasks), batch_result)
        logger.info("[1/6] データ取得完了")
    except Exception as e:
        if _handle_stage_error(
            PipelineStage.CRITICAL, "[1/6] データ取得", e, send_daily_pipeline_error
        ):
            raise

    # 2. 前日予測の精度チェック（NON_CRITICAL: 失敗しても後続処理を継続）
    # 3 で当日分の prediction_results を保存する前に読む（Delete-Insert 対策）。
    # やり直しで既に今日の予想に入れ替わっていれば、前の回の JSON を残して飛ばす（#904）。
    logger.info("[2/6] 予測精度チェック開始")
    accuracy_report = None
    if not production_already_predicted_today():
        from src.prediction.prediction_pipeline import run_accuracy_check

        try:
            summary = run_accuracy_check(
                horizon=1, model_name="production", model_version_filter="production"
            )
            send_accuracy_summary(summary, horizon=1)
            logger.info("[2/6] 予測精度チェック完了 (production)")
        except Exception as e:
            # 採点は 3 の保存の後にやり直せないため知らせる。投げ直すと 3 の前に
            # やり直しが入り #904 の飛ばしと絡むため、知らせだけにする（#1006）
            _handle_stage_error(
                PipelineStage.NON_CRITICAL, "[2/6] 予測精度チェック (production)", e
            )
            _notify_stage_failure("[2/6] 予測精度チェック (production)", e)

        try:
            run_accuracy_check(
                horizon=1, model_name="challenger", model_version_filter="challenger"
            )
            logger.info("[2/6] 予測精度チェック完了 (challenger)")
        except Exception as e:
            _handle_stage_error(
                PipelineStage.NON_CRITICAL, "[2/6] 予測精度チェック (challenger)", e
            )

        # 採点の JSON も 3 の保存の前に業種なしで書き、業種は道の最後で書き足す（#843）
        accuracy_report = write_accuracy_json_without_sectors()

    # 3. 予測（CRITICAL: 失敗時はパイプライン停止 + Discord通知）
    logger.info("[3/6] 予測開始 (production)")
    from src.prediction.prediction_pipeline import output_top_worst_results, predict_all_unified
    from src.prediction.types import UNIFIED_PREDICTION_MODEL_NAMES

    # 出力 invariant 用。None は「評価が実行されなかった」を意味する（#615 対策）
    prediction_violation_ids: list[str] | None = None
    prediction_details: dict | None = None

    # daily.py の要求リストと prediction_pipeline.py の実推論リストが別々の
    # リテラルだと設定ドリフトで A-1/A-2 が壊れるため、同じ定数を参照する（I-4 対策）。
    requested_models = list(UNIFIED_PREDICTION_MODEL_NAMES)
    loaded_models: list[str] = []
    output_rows: list = []
    # 保存前に退避しておく前回ラン統計（B-1/B-2/B-3 の急変チェックが使う）。
    # prediction_results は Delete-Insert のスナップショットテーブルなので、
    # 保存後に「前回ラン」を探すと今回の保存で上書き済みで見つからない
    # （毎日 compared_with_previous=False になる、C-1 対策）。
    # 取得に失敗しても致命的ではない（急変チェックがスキップされるだけ）。
    previous_stats = None

    try:
        from src.prediction.predict_unified import preload_models

        loaded_models = preload_models(requested_models)

        previous_stats = fetch_previous_run_stats(model_version="production")

        predict_errors: list[tuple[str, str]] = []  # 例外で予想を出せなかった銘柄（#1118）
        output_rows = predict_all_unified(on_error=lambda m, s: predict_errors.append((m, s)))
        output_top_worst_results(
            output_rows, mode="unified", shadow_mode=True, model_version="production"
        )
        logger.info("[3/6] 予測完了 (production): %d 銘柄", len(output_rows))
        notify_prediction_errors(predict_errors)
    except Exception as e:
        if _handle_stage_error(
            PipelineStage.CRITICAL,
            "[3/6] 予測 (production)",
            e,
            send_daily_pipeline_error,
        ):
            raise

    # 3.1. 出力 invariant 評価（NON_CRITICAL: 健全性チェックが本体を止めてはならない）
    logger.info("[3.1/6] 出力 invariant 評価開始")
    try:
        prediction_violation_ids, prediction_details = evaluate_output_invariants_stage(
            requested_models, loaded_models, output_rows, previous_stats
        )
        logger.info("[3.1/6] 出力 invariant 評価完了: %s", prediction_details)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[3.1/6] 出力 invariant 評価", e)

    # 3.5. Challenger shadow 予測（NON_CRITICAL: モデルが存在しない場合はスキップ、失敗しても継続）
    logger.info("[3.5/6] Challenger shadow 予測開始")
    try:
        from src.prediction.shadow_evaluation import predict_with_challenger_unified

        challenger_rows = predict_with_challenger_unified()
        if challenger_rows:
            output_top_worst_results(
                challenger_rows, mode="unified", shadow_mode=True, model_version="challenger"
            )
            logger.info("[3.5/6] Challenger shadow 予測完了: %d 銘柄", len(challenger_rows))
        else:
            logger.info("[3.5/6] Challenger モデルなし（スキップ）")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[3.5/6] Challenger shadow 予測", e)

    # 4. 日次ドリフトチェック（NON_CRITICAL: 失敗しても後続処理を継続）
    logger.info("[4/6] 日次ドリフトチェック開始")
    try:
        run_daily_drift_check()
        logger.info("[4/6] 日次ドリフトチェック完了")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[4/6] 日次ドリフトチェック", e)

    # 5. Discord通知（CRITICAL: 失敗時はパイプライン停止）
    logger.info("[5/6] Discord通知送信")
    try:
        from src.infrastructure.persistence.prediction_result_repository import (
            PostgresPredictionResultRepository,
        )

        send_daily_pipeline_completion(predictions=PostgresPredictionResultRepository())
        logger.info("[5/6] Discord通知完了")
    except Exception as e:
        if _handle_stage_error(PipelineStage.CRITICAL, "[5/6] Discord通知", e):
            raise

    # 6. 運用アラート評価（NON_CRITICAL: 条件成立時のみ Discord へ発報する）
    logger.info("[6/6] 運用アラート評価開始")
    record_health_degraded_streak()  # NF-303-4 の連続回数（#940）
    try:
        evaluate_and_notify_alerts(prediction_violation_ids, prediction_details)
        logger.info("[6/6] 運用アラート評価完了")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[6/6] 運用アラート評価", e)

    # 採点の JSON に業種の内訳を書き足す（NON_CRITICAL。[2/6] で作った材料を使う）
    append_accuracy_json_sectors(accuracy_report)

    logger.info("=== 日次パイプライン完了 ===")


def run_daily_auto_order():
    """
    毎営業日 8:50 実行: 前夜の予測シグナルに基づきペーパートレード注文を発注する。

    本番切り替え時は環境変数 AUTO_TRADE_MODE=live に変更する。
    """
    import os

    from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter
    from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
    from src.infrastructure.persistence.prediction_result_repository import (
        PostgresPredictionResultRepository,
    )
    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.trading.brokers.account.factory import make_order_broker
    from src.trading.execution import run_daily_orders

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    logger.info("=== 自動発注開始 (mode=%s) ===", mode)

    trade_diff_sink = PostgresTradeDiffSink()

    if mode == "live":
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker()
        market_data = None
    else:
        market_data = YFinanceMarketDataAdapter()
        broker = make_order_broker(
            job="auto",
            market_data_port=market_data,
            trade_diff_sink=trade_diff_sink,
            notifier=DiscordNotificationAdapter(),
        )

    try:
        stats = run_daily_orders(
            broker=broker,
            order_run_sink=PostgresOrderRunSink(),
            trade_diff_sink=trade_diff_sink,
            prediction_repo=PostgresPredictionResultRepository(),
            market="jp",
            mode=mode,
            market_data=market_data,
        )
        logger.info(
            "=== 自動発注完了: 買い=%s 売り=%s ===", stats["buy_orders"], stats["sell_orders"]
        )
    except Exception as e:
        logger.error("自動発注失敗: %s", e, exc_info=True)
        # 発注の道が落ちても、期限を過ぎた保有の決済は走らせてからジョブを失敗にする（#932）
        try:
            run_horizon_exit_check()
        except Exception:
            logger.error("ホライズン決済チェック失敗", exc_info=True)
        raise
    record_loss_limit_streak(stats)  # NF-303-2 の連続回数（#934）

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_daily_order_completion

        send_daily_order_completion(
            buy_orders=stats["buy_orders"],
            sell_orders=stats["sell_orders"],
            mode=mode,
            trading_stopped=stats.get("trading_stopped", False),
            stop_reason=stats.get("stop_reason"),
            daily_loss=stats.get("daily_loss"),
            daily_loss_limit=stats.get("daily_loss_limit"),
            errors=stats.get("errors", 0),
        )
    except Exception as e:
        logger.error("自動発注完了通知失敗: %s", e, exc_info=True)

    # ホライズン期限切れポジションの自動決済。失敗は相関の警告を送った後に投げ直し、
    # 期限を過ぎた保有が決済されずに残ったことをジョブの失敗として見せる（#886）
    horizon_exit_error: Exception | None = None
    try:
        run_horizon_exit_check()
    except Exception as e:
        logger.error("ホライズン決済チェック失敗: %s", e, exc_info=True)
        horizon_exit_error = e

    # 相関リスク警告通知
    if stats.get("correlation_blocked"):
        try:
            from config.settings import CORRELATION_ENC_THRESHOLD
            from src.reporting.discord.discord_utils import send_correlation_alert

            send_correlation_alert(
                enc=stats.get("enc", 0.0),
                enc_threshold=CORRELATION_ENC_THRESHOLD,
                avg_correlation=stats.get("avg_correlation", 0.0),
                n_symbols=stats.get("n_held_symbols", 0),
                symbols=stats.get("held_symbols_list", []),
            )
        except Exception as e:
            logger.error("相関リスク警告通知失敗: %s", e, exc_info=True)

    if horizon_exit_error is not None:
        raise horizon_exit_error
    # 注文ごとの失敗（SL/TP の売りを含む）も知らせの後にジョブの失敗にする（#1074）。
    # daily_auto_order は max_retries: 0 のため、失敗にしても発注は呼び直されない。
    if stats.get("errors", 0) > 0:
        raise RuntimeError(f"自動発注で注文の失敗が {stats['errors']} 件あった")


def run_horizon_exit_check() -> None:
    """
    毎営業日実行: target_exit_date を過ぎたペーパーポジションを成行売りで強制決済する。

    SL/TP によって既に決済済みのポジションは paper_positions に存在しないため
    自然に除外される（SL/TP 優先）。
    銘柄ごとに最後の filled BUY の target_exit_date だけを見て、
    約定前の SELL がある銘柄は飛ばす。
    live モードでは処理をスキップする。
    """
    import os

    from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter
    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.orchestration.jobs.common import today_jst_iso
    from src.trading.brokers.account.factory import make_order_broker
    from src.trading.brokers.base import OrderSide
    from src.utils.db._connection import _db_connection

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのためホライズン決済チェックをスキップ")
        return

    today_str = today_jst_iso()  # target_exit_date は保有の最後の JST の日 (#949)
    with _db_connection() as con:
        # paper_orders の行は売った後も残るため、銘柄ごとに今の保有を作った
        # 最後の filled BUY の期限だけを見る (Issue #945)
        rows = con.execute(
            """
            SELECT symbol
            FROM (
                SELECT DISTINCT ON (symbol) symbol, target_exit_date
                FROM paper_orders
                WHERE side = %s
                  AND status = 'filled'
                ORDER BY symbol, filled_at DESC NULLS LAST, created_at DESC, order_id DESC
            ) AS latest_buy
            WHERE target_exit_date IS NOT NULL
              AND CAST(target_exit_date AS VARCHAR) < %s
            ORDER BY symbol
            """,
            [int(OrderSide.BUY), today_str],
        ).fetchall()
        pending_sell_rows = con.execute(
            "SELECT DISTINCT symbol FROM paper_orders WHERE side = %s AND status = 'pending'",
            [int(OrderSide.SELL)],
        ).fetchall()

    pending_sell_symbols = {row[0] for row in pending_sell_rows}
    symbols_to_exit = [row[0] for row in rows]
    if not symbols_to_exit:
        logger.info("[horizon_exit] 期限切れポジションなし")
        return

    logger.info("[horizon_exit] 期限切れポジション対象: %s", symbols_to_exit)
    broker = make_order_broker(
        job="horizon",
        market_data_port=YFinanceMarketDataAdapter(),
        trade_diff_sink=PostgresTradeDiffSink(),
        notifier=DiscordNotificationAdapter(),
    )
    # 送った売りは約定まで保有を変えないので、保有の一覧はループの前に1回だけ読む（#1095）
    positions = broker.get_positions()
    exited: list[str] = []
    failed: list[str] = []
    for symbol in symbols_to_exit:
        if symbol in pending_sell_symbols:
            # get_positions は約定前の SELL を引かないため、重ねると売りの二重発注になる
            logger.info("[horizon_exit] 約定前の SELL があるためスキップ: %s", symbol)
            continue
        pos = next((p for p in positions if p["symbol"].replace(".T", "") == symbol), None)
        if pos is None or pos.get("qty", 0) <= 0:
            continue
        try:
            broker.send_order(symbol, OrderSide.SELL, pos["qty"])
            logger.info("[horizon_exit] ホライズン決済: %s %d株", symbol, pos["qty"])
            exited.append(symbol)
        except Exception as e:
            logger.error("[horizon_exit] 決済失敗: %s: %s", symbol, e, exc_info=True)
            failed.append(symbol)

    logger.info("[horizon_exit] ホライズン決済完了: %d 件", len(exited))
    if failed:  # 他の銘柄を決済し終えてから失敗にする（#1074）
        raise RuntimeError(f"ホライズン決済に失敗した銘柄: {failed}")


def run_daily_settle_orders(notify_when_none: bool = True) -> int:
    """
    毎営業日 9:05 実行: pending 状態のペーパートレード注文を当日始値で約定処理する。
    本番（live）モードでは kabu STATION® が自動処理するため不要。
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため settle スキップ")
        return 0

    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.trading.brokers.paper.paper_broker import PaperBroker

    logger.info("=== ペーパートレード約定処理開始 ===")
    try:
        broker = PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            trade_diff_sink=PostgresTradeDiffSink(),
        )
        settled = broker.settle_pending_orders()
        logger.info("=== 約定処理完了: %s 件 ===", len(settled))
    except Exception as e:
        logger.error("約定処理失敗: %s", e, exc_info=True)
        raise
    from src.orchestration.jobs.settle_notify import notify_settle_result

    # 例外で約定できなかった件数を返す（締めの知らせで当日の行の無さと分ける #1088）
    return notify_settle_result(len(settled), broker.last_settle_errors, notify_when_none)


def run_daily_paper_trade_report():
    """
    毎営業日 15:30 実行: ペーパートレードのポジション・損益レポートを Discord に送信する。

    内容:
        - 現在のポジション一覧（銘柄・保有数・平均取得価格・現在値・含み損益）
        - 通算損益サマリー（実現損益・含み損益・合計損益）
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのためペーパートレードレポートをスキップ")
        return

    logger.info("=== ペーパートレード損益レポート送信開始 ===")
    try:
        from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.reporting.discord.discord_utils import send_paper_trade_position_report
        from src.trading.brokers.paper.paper_broker import PaperBroker

        broker = PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            trade_diff_sink=PostgresTradeDiffSink(),
        )
        positions = broker.get_positions()
        summary = broker.get_pnl_summary()
        send_paper_trade_position_report(positions, summary)
        logger.info("=== ペーパートレード損益レポート送信完了 ===")
    except Exception as e:
        logger.error("ペーパートレードレポート送信失敗: %s", e, exc_info=True)


def run_pre_close_alert() -> None:
    """
    毎営業日 15:00 実行: 保有ポジションを直近予測と照合し
    利確/損切り推奨度を Discord に通知する（R-409）。
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため引け前アラートをスキップ")
        return

    logger.info("=== 引け前ポジション再評価アラート開始 ===")
    try:
        from src.infrastructure.persistence.prediction_result_repository import (
            PostgresPredictionResultRepository,
        )
        from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.pre_close_alert_service import get_pre_close_alerts

        lines = get_pre_close_alerts(
            market_data_port=YFinanceMarketDataAdapter(),
            trade_diff_sink=PostgresTradeDiffSink(),
            prediction_repo=PostgresPredictionResultRepository(),
        )
        logger.info("引け前アラート評価完了: %d行", len(lines))
    except Exception as e:
        logger.error("引け前アラート評価失敗: %s", e, exc_info=True)
        raise

    try:
        from src.reporting.discord.discord_notification_specs import PRE_CLOSE_ALERT
        from src.reporting.discord.discord_utils import send_webhook_notification

        send_webhook_notification(
            PRE_CLOSE_ALERT.title, "\n".join(lines), color=PRE_CLOSE_ALERT.color
        )
        logger.info("=== 引け前ポジション再評価アラート送信完了 ===")
    except Exception as e:
        logger.error("引け前アラート通知失敗: %s", e, exc_info=True)


def run_daily_rule_signals() -> None:
    """
    日次実行（平日 16:00）: 有効ルールを持つ銘柄に当日シグナルを適用し
    ペーパートレード注文を自動発行する。

    フロー:
        1. rule_best_by_symbol から有効銘柄を読み込む
        2. 各銘柄の最新データにルールを適用して Buy/Sell/Hold を判定
        3. ペーパートレード注文を PaperBroker 経由で発行
        4. Discord にシグナルサマリーを通知
    """
    import os

    logger.info("=== 日次ルールシグナルジョブ開始 ===")
    market = os.environ.get("RULE_EVAL_MARKET", "jp")

    try:
        from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter
        from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.rule_engine.pipeline import generate_rule_signals
        from src.trading.brokers.account.factory import make_order_broker
        from src.trading.rule_execution import execute_rule_paper_trades

        market_data_adapter = YFinanceMarketDataAdapter()
        batch = generate_rule_signals(market=market, market_data_port=market_data_adapter)
        _raise_if_rule_signals_mostly_unavailable(batch)
        signals = batch.signals
        rule_trade_diff_sink = PostgresTradeDiffSink()
        trade_stats = execute_rule_paper_trades(
            signals=signals,
            market=market,
            market_data_port=market_data_adapter,
            trade_diff_sink=rule_trade_diff_sink,
            broker=make_order_broker(
                job="rule",
                market_data_port=market_data_adapter,
                trade_diff_sink=rule_trade_diff_sink,
                notifier=DiscordNotificationAdapter(),
            ),
        )
        logger.info(
            "=== 日次ルールシグナル完了: BUY=%s SELL=%s 注文失敗=%s 取れなかった=%s 対象=%s ===",
            trade_stats["buy_orders"],
            trade_stats["sell_orders"],
            trade_stats.get("errors", 0),
            len(batch.unavailable),
            batch.total,
        )
    except Exception as e:
        logger.error("日次ルールシグナル失敗: %s", e, exc_info=True)
        raise

    try:
        from src.reporting.discord.discord_utils import send_rule_daily_signals

        send_rule_daily_signals(
            signals=signals,
            market=market,
            buy_orders=trade_stats["buy_orders"],
            sell_orders=trade_stats["sell_orders"],
        )
    except Exception as e:
        logger.error("ルールシグナル通知失敗: %s", e, exc_info=True)

    # 注文の失敗は知らせを送った後にジョブの失敗にする（#1075）。run_scheduler の
    # daily_rule_signals は max_retries: 0 のため、失敗にしても発注は呼び直されない。
    errors = trade_stats.get("errors", 0)
    if errors > 0:
        logger.error("日次ルールシグナルで注文の失敗が %d 件あった", errors)
        raise RuntimeError(f"日次ルールシグナルで注文の失敗が {errors} 件あった")
