"""日次・場中スケジューラジョブ。

毎営業日に走るデータパイプライン・自動発注・約定処理・レポート・
ドリフト監視・バックアップ・ルールシグナルのオーケストレーション。
"""

from src.orchestration.jobs.alerting import (
    evaluate_and_notify_alerts,
    evaluate_output_invariants_stage,
    fetch_previous_run_stats,
)
from src.orchestration.jobs.common import _handle_stage_error
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
        run_batch_pipeline(tasks, notification_port=DiscordNotificationAdapter())
        logger.info("[1/6] データ取得完了")
    except Exception as e:
        if _handle_stage_error(
            PipelineStage.CRITICAL, "[1/6] データ取得", e, send_daily_pipeline_error
        ):
            raise

    # 2. 前日予測の精度チェック（NON_CRITICAL: 失敗しても後続処理を継続）
    # 3 で当日分の prediction_results を保存する前に読む（Delete-Insert 対策）。
    logger.info("[2/6] 予測精度チェック開始")
    try:
        from src.prediction.prediction_pipeline import run_accuracy_check

        summary = run_accuracy_check(
            horizon=1, model_name="production", model_version_filter="production"
        )
        send_accuracy_summary(summary, horizon=1)
        logger.info("[2/6] 予測精度チェック完了 (production)")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[2/6] 予測精度チェック (production)", e)

    try:
        from src.prediction.prediction_pipeline import run_accuracy_check

        run_accuracy_check(horizon=1, model_name="challenger", model_version_filter="challenger")
        logger.info("[2/6] 予測精度チェック完了 (challenger)")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[2/6] 予測精度チェック (challenger)", e)

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

        output_rows = predict_all_unified()
        output_top_worst_results(
            output_rows, mode="unified", shadow_mode=True, model_version="production"
        )
        logger.info("[3/6] 予測完了 (production): %d 銘柄", len(output_rows))
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
        send_daily_pipeline_completion()
        logger.info("[5/6] Discord通知完了")
    except Exception as e:
        if _handle_stage_error(PipelineStage.CRITICAL, "[5/6] Discord通知", e):
            raise

    # 6. 運用アラート評価（NON_CRITICAL: 条件成立時のみ Discord へ発報する）
    logger.info("[6/6] 運用アラート評価開始")
    try:
        evaluate_and_notify_alerts(prediction_violation_ids, prediction_details)
        logger.info("[6/6] 運用アラート評価完了")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[6/6] 運用アラート評価", e)

    logger.info("=== 日次パイプライン完了 ===")


def run_daily_auto_order():
    """
    毎営業日 8:50 実行: 前夜の予測シグナルに基づきペーパートレード注文を発注する。

    本番切り替え時は環境変数 AUTO_TRADE_MODE=live に変更する。
    """
    import os

    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.trading.brokers.paper.paper_broker import PaperBroker
    from src.trading.execution import run_daily_orders
    from src.utils.db import upsert_paper_real_diff

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    logger.info("=== 自動発注開始 (mode=%s) ===", mode)

    if mode == "live":
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker()
        market_data = None
    else:
        market_data = YFinanceMarketDataAdapter()
        broker = PaperBroker(
            market_data_port=market_data,
            record_diff=upsert_paper_real_diff,
        )

    try:
        stats = run_daily_orders(broker=broker, market="jp", mode=mode, market_data=market_data)
        logger.info(
            "=== 自動発注完了: 買い=%s 売り=%s ===", stats["buy_orders"], stats["sell_orders"]
        )
    except Exception as e:
        logger.error("自動発注失敗: %s", e, exc_info=True)
        raise

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
        )
    except Exception as e:
        logger.error("自動発注完了通知失敗: %s", e, exc_info=True)

    # ホライズン期限切れポジションの自動決済
    try:
        run_horizon_exit_check()
    except Exception as e:
        logger.error("ホライズン決済チェック失敗: %s", e, exc_info=True)

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


def run_horizon_exit_check() -> None:
    """
    毎営業日実行: target_exit_date を過ぎたペーパーポジションを成行売りで強制決済する。

    SL/TP によって既に決済済みのポジションは paper_positions に存在しないため
    自然に除外される（SL/TP 優先）。
    live モードでは処理をスキップする。
    """
    import os
    from datetime import date

    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.trading.brokers.base import OrderSide
    from src.trading.brokers.paper.paper_broker import PaperBroker
    from src.utils.db._connection import _db_connection

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのためホライズン決済チェックをスキップ")
        return

    today_str = date.today().isoformat()
    with _db_connection() as con:
        rows = con.execute(
            """
            SELECT DISTINCT symbol
            FROM paper_orders
            WHERE side = %s
              AND status = 'filled'
              AND target_exit_date IS NOT NULL
              AND CAST(target_exit_date AS VARCHAR) <= %s
            """,
            [int(OrderSide.BUY), today_str],
        ).fetchall()

    symbols_to_exit = [row[0] for row in rows]
    if not symbols_to_exit:
        logger.info("[horizon_exit] 期限切れポジションなし")
        return

    logger.info("[horizon_exit] 期限切れポジション対象: %s", symbols_to_exit)
    broker = PaperBroker(market_data_port=YFinanceMarketDataAdapter())
    exited: list[str] = []
    for symbol in symbols_to_exit:
        positions = broker.get_positions()
        pos = next((p for p in positions if p["symbol"].replace(".T", "") == symbol), None)
        if pos is None or pos.get("qty", 0) <= 0:
            continue
        try:
            broker.send_order(symbol, OrderSide.SELL, pos["qty"])
            logger.info("[horizon_exit] ホライズン決済: %s %d株", symbol, pos["qty"])
            exited.append(symbol)
        except Exception as e:
            logger.error("[horizon_exit] 決済失敗: %s: %s", symbol, e, exc_info=True)

    logger.info("[horizon_exit] ホライズン決済完了: %d 件", len(exited))


def run_daily_settle_orders():
    """
    毎営業日 9:05 実行: pending 状態のペーパートレード注文を当日始値で約定処理する。
    本番（live）モードでは kabu STATION® が自動処理するため不要。
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため settle スキップ")
        return

    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.prediction.db import upsert_paper_real_diff
    from src.trading.brokers.paper.paper_broker import PaperBroker

    logger.info("=== ペーパートレード約定処理開始 ===")
    try:
        broker = PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            record_diff=upsert_paper_real_diff,
        )
        settled = broker.settle_pending_orders()
        logger.info("=== 約定処理完了: %s 件 ===", len(settled))
    except Exception as e:
        logger.error("約定処理失敗: %s", e, exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_daily_settle_completion

        send_daily_settle_completion(settled_count=len(settled))
    except Exception as e:
        logger.error("約定処理完了通知失敗: %s", e, exc_info=True)


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
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.reporting.discord.discord_utils import send_paper_trade_position_report
        from src.trading.brokers.paper.paper_broker import PaperBroker

        broker = PaperBroker(market_data_port=YFinanceMarketDataAdapter())
        positions = broker.get_positions()
        summary = broker.get_pnl_summary()
        send_paper_trade_position_report(positions, summary)
        logger.info("=== ペーパートレード損益レポート送信完了 ===")
    except Exception as e:
        logger.error("ペーパートレードレポート送信失敗: %s", e, exc_info=True)


def run_daily_drift_check():
    """
    日次ドリフト監視: 直近 20 営業日の MAE / Hit Rate を監視し、閾値超過銘柄を自動再学習する。

    環境変数:
        DRIFT_MAE_THRESHOLD:      MAE 閾値（デフォルト 0.02 = 2%）
        DRIFT_HIT_RATE_THRESHOLD: Hit Rate 閾値（デフォルト 0.45 = 45%）

    完了条件:
        - 閾値超過銘柄が存在する場合に銘柄別モデルを再学習
        - 再学習トリガーを Discord に通知
    """
    import os

    from src.prediction.db import load_drift_summary
    from src.utils.db.system_config import get_config_value

    _mae_default = os.environ.get("DRIFT_MAE_THRESHOLD", "0.02")
    _hr_default = os.environ.get("DRIFT_HIT_RATE_THRESHOLD", "0.45")
    mae_threshold = float(get_config_value("drift.mae_threshold", _mae_default))
    hit_rate_threshold = float(get_config_value("drift.hit_rate_threshold", _hr_default))

    logger.info(
        f"=== 日次ドリフトチェック開始 (MAE閾値={mae_threshold:.2%}, HitRate閾値={hit_rate_threshold:.0%}) ==="
    )

    summary = load_drift_summary(horizon=1, recent_n=20)
    if summary is None or summary.empty:
        logger.info("ドリフトチェック: 精度データなし（スキップ）")
        return

    triggered = summary[
        (summary["mean_abs_error"] >= mae_threshold)
        | (summary["direction_accuracy"] <= hit_rate_threshold)
    ]

    if triggered.empty:
        logger.info("ドリフトチェック: 閾値超過銘柄なし")
        return

    triggered_list = triggered[
        ["market", "symbol", "mean_abs_error", "direction_accuracy"]
    ].to_dict("records")
    logger.warning("ドリフト検知: %s 銀柄が閾値超過 → 自動再学習開始", len(triggered_list))

    # Discord 通知（再学習開始前）
    try:
        from src.reporting.discord.discord_utils import send_drift_retrain_notification

        send_drift_retrain_notification(triggered_list, mae_threshold, hit_rate_threshold)
    except Exception as e:
        logger.error("ドリフト通知失敗: %s", e, exc_info=True)

    # 銘柄別モデル再学習
    from src.prediction.training_pipeline import train_models_for_symbol_task
    from src.watchlist.batch_runner import load_target_symbols

    all_tasks = {(t.market, t.symbol): t for t in load_target_symbols()}
    success_count = 0
    retrained_symbols: list = []
    for sym in triggered_list:
        task = all_tasks.get((sym["market"], sym["symbol"]))
        if task is None:
            logger.warning(
                "ドリフト再学習: タスクが見つかりません (%s/%s)", sym["market"], sym["symbol"]
            )
            continue
        try:
            logger.info("ドリフト再学習開始: %s/%s", sym["market"], sym["symbol"])
            result = train_models_for_symbol_task(task)
            if result.get("status") == "success":
                success_count += 1
                retrained_symbols.append(sym)
            else:
                logger.warning(
                    f"ドリフト再学習スキップ/失敗 ({sym['market']}/{sym['symbol']}): "
                    f"{result.get('reason') or result.get('error') or result.get('status')}"
                )
        except Exception as e:
            logger.error(
                "ドリフト再学習失敗 (%s/%s): %s", sym["market"], sym["symbol"], e, exc_info=True
            )

    logger.info(
        "=== 日次ドリフトチェック完了: 再学習=%s/%s 件 ===", success_count, len(triggered_list)
    )

    # 特徴量除外提案通知（再学習成功銘柄の最新 Permutation Importance を通知）
    if retrained_symbols:
        try:
            from src.prediction.db import load_feature_exclusion_candidates
            from src.reporting.discord.discord_utils import send_feature_suggestion_notification

            feature_suggestions = []
            for sym in retrained_symbols:
                candidates = load_feature_exclusion_candidates(sym["market"], sym["symbol"])
                feature_suggestions.append(
                    {"market": sym["market"], "symbol": sym["symbol"], "candidates": candidates}
                )
            send_feature_suggestion_notification(feature_suggestions)
        except Exception as e:
            logger.error("特徴量除外提案通知失敗: %s", e, exc_info=True)


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
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.pre_close_alert_service import get_pre_close_alerts

        lines = get_pre_close_alerts(market_data_port=YFinanceMarketDataAdapter())
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


def run_daily_backup() -> None:
    """
    毎日深夜実行: PostgreSQL を pg_dump（カスタムフォーマット）でタイムスタンプ付き
    ディレクトリへ出力し、最大5世代を保持する。

    手順:
        1. pg_dump（-Fc）で data/backups/YYYYMMDD_HHMMSS/stockfixer.dump へ出力
        2. 5世代超過分を古い順に削除
        3. Discord に完了通知
    """
    from src.orchestration.backup_pipeline import run_db_backup
    from src.reporting.discord.discord_utils import send_backup_completion

    logger.info("=== 日次バックアップ開始 ===")
    result = run_db_backup()

    try:
        send_backup_completion(
            backup_path=result["backup_path"],
            size_mb=result["size_mb"],
            elapsed_seconds=result["elapsed_seconds"],
            pruned_count=result["pruned_count"],
            error=result["error"],
        )
    except Exception as e:
        logger.error("バックアップ通知失敗: %s", e, exc_info=True)

    if result["error"]:
        raise RuntimeError(f"バックアップ失敗: {result['error']}")

    logger.info("=== 日次バックアップ完了 ===")


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
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.rule_engine.pipeline import run_rule_signal_pipeline
        from src.trading.rule_execution import execute_rule_paper_trades

        market_data_adapter = YFinanceMarketDataAdapter()
        signals = run_rule_signal_pipeline(market=market, market_data_port=market_data_adapter)
        trade_stats = execute_rule_paper_trades(
            signals=signals, market=market, market_data_port=market_data_adapter
        )
        logger.info(
            "=== 日次ルールシグナル完了: BUY=%s SELL=%s ===",
            trade_stats["buy_orders"],
            trade_stats["sell_orders"],
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
