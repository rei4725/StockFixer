"""
Discord通知ユーティリティ

Webhookを使用したDiscord通知機能
"""

import logging
import os
from typing import Optional

import requests

from src.prediction.types import PredictionResult
from src.reporting.discord import rate_limiter as _rate_limiter
from src.reporting.discord.discord_formatters import convert_df_for_discord, get_market_emoji
from src.reporting.discord.discord_notification_specs import (
    DAILY_PIPELINE_COMPLETION,
    DAILY_PIPELINE_ERROR,
    DAILY_SETTLE_COMPLETION,
    DB_MAINTENANCE_COMPLETION,
    DB_MAINTENANCE_ERROR,
    PRE_CLOSE_ALERT,
    SHADOW_EVALUATION_CHALLENGER_WINS,
    SHADOW_EVALUATION_NO_WINNER,
    WEEKLY_TRAINING_COMPLETION,
    NotificationSpec,
    get_daily_order_spec,
    get_optimization_spec,
    get_walk_forward_report_spec,
)
from src.reporting.discord.discord_text import (
    DISCORD_TEXT_LIMIT,
    DISCORD_WIDE_TEXT_LIMIT,
    split_text_chunks,
)
from src.reporting.query_service import get_latest_market_prediction_snapshots
from src.utils.japan_time import format_jst, format_jst_from_iso, isoformat_jst
from src.utils.run_context import get_run_id

logger = logging.getLogger(__name__)

DISCORD_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S JST"
DISCORD_MINUTE_FORMAT = "%Y-%m-%d %H:%M JST"
DISCORD_DATE_FORMAT = "%Y/%m/%d"


def _get_webhook_url() -> str | None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。Webhook通知をスキップします。")
        return None
    return webhook_url


def _post_webhook(
    *,
    json_payload: dict | None = None,
    data_payload: dict | None = None,
    files=None,
    timeout: int = 10,
):
    webhook_url = _get_webhook_url()
    if not webhook_url:
        return None
    _rate_limiter.apply_rate_limit()
    response = requests.post(
        webhook_url, json=json_payload, data=data_payload, files=files, timeout=timeout
    )
    response.raise_for_status()
    return response


def send_webhook_text_chunked(
    text: str,
    *,
    limit: int = DISCORD_TEXT_LIMIT,
    preserve_lines: bool = True,
) -> bool:
    success = True
    for chunk in split_text_chunks(text, limit=limit, preserve_lines=preserve_lines):
        if not send_webhook_text(chunk):
            success = False
    return success


def send_text_file_chunked(
    file_path: str,
    *,
    limit: int = DISCORD_TEXT_LIMIT,
    preserve_lines: bool = False,
) -> bool:
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            return send_webhook_text_chunked(
                file_handle.read(),
                limit=limit,
                preserve_lines=preserve_lines,
            )
    except OSError as exc:
        logger.error("テキストファイル送信失敗: %s", exc, exc_info=True)
        return False


def send_status_notification(spec: NotificationSpec, lines: list[str]) -> bool:
    return send_webhook_notification(spec.title, "\n".join(lines), color=spec.color)


def send_webhook_notification(
    title: str,
    message: str,
    color: int = 0x00FF00,
) -> bool:
    """
    Discordブhookを使用して通知を送信する

    Args:
        title: メッセージタイトル
        message: メッセージ本文
        color: Embedの色（16進数、デフォルトは緑）

    Returns:
        成功時True、失敗時False
    """
    try:
        should_send, suppression_summary = _rate_limiter.check_and_record(f"{title}\n{message}")
        if not should_send:
            return True

        if suppression_summary:
            try:
                _post_webhook(json_payload={"content": suppression_summary}, timeout=10)
            except requests.exceptions.RequestException as e:
                logger.error("抑止サマリー送信失敗: %s", e, exc_info=True)

        run_id = get_run_id()
        embed: dict = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": isoformat_jst(),
        }
        if run_id:
            embed["footer"] = {"text": f"run_id: {run_id}"}
        embed_data = {"embeds": [embed]}

        response = _post_webhook(json_payload=embed_data, timeout=10)
        if response is None:
            return False
        response.raise_for_status()

        logger.info("Discord通知送信成功: %s", title)
        return True

    except requests.exceptions.RequestException as e:
        logger.error("Discord通知送信失敗: %s", e, exc_info=True)
        return False


def send_webhook_text(text: str) -> bool:
    """
    プレーンテキストメッセージをWebhookで送信する

    Args:
        text: 送信テキスト（コードフェンス込み）

    Returns:
        成功時True、失敗時False
    """
    try:
        payload = {"content": text}
        response = _post_webhook(json_payload=payload, timeout=10)
        if response is None:
            return False
        response.raise_for_status()

        logger.info("Discord通知送信成功: テキストメッセージ")
        return True

    except requests.exceptions.RequestException as e:
        logger.error("Discord通知送信失敗: %s", e, exc_info=True)
        return False


def send_daily_pipeline_completion(
    data_count: Optional[int] = None,
    prediction_markets: Optional[list] = None,
    include_forecast: bool = True,
) -> bool:
    """
    日次パイプライン完了通知（完了メッセージ + 予測結果テーブル）

    Args:
        data_count: 取得したデータ件数
        prediction_markets: 予測対象マーケット
        include_forecast: 予測結果テーブルを含めるかどうか

    Returns:
        成功時True、失敗時False
    """
    # 1. 完了メッセージを送信
    message_lines = [
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
    ]

    if data_count is not None:
        message_lines.append(f"取得データ: {data_count}件")

    if prediction_markets:
        markets_str = "、".join(prediction_markets)
        message_lines.append(f"予測市場: {markets_str}")

    success = send_status_notification(DAILY_PIPELINE_COMPLETION, message_lines)

    # 2. 予測結果テーブルを送信（マーケット単位でTop→Worst、1メッセージにまとめる）
    if include_forecast:
        try:
            latest_ts, snapshots = get_latest_market_prediction_snapshots()
            if latest_ts and snapshots:
                ts_label = latest_ts[:16] if len(latest_ts) >= 16 else latest_ts
                parts = [f"📊 予測結果 — {ts_label}\n{'━' * 28}"]

                for snapshot in snapshots:
                    emoji = get_market_emoji(snapshot.market)
                    parts.append(f"\n{emoji} {snapshot.market}")

                    if snapshot.top_results:
                        df_top = convert_df_for_discord(
                            PredictionResult.to_dataframe(snapshot.top_results)
                        )
                        parts.append(f"📈 上位10銘柄\n```\n{df_top.to_string(index=False)}\n```")

                    if snapshot.worst_results:
                        df_worst = convert_df_for_discord(
                            PredictionResult.to_dataframe(snapshot.worst_results)
                        )
                        parts.append(f"📉 下位10銘柄\n```\n{df_worst.to_string(index=False)}\n```")

                if len(parts) > 1:
                    send_webhook_text_chunked(
                        "\n".join(parts),
                        limit=DISCORD_WIDE_TEXT_LIMIT,
                        preserve_lines=False,
                    )

        except Exception as e:
            logger.error("予測結果テーブル送信失敗: %s", e)

    return success


def send_daily_pipeline_error(error_message: str) -> bool:
    """
    日次パイプラインエラー通知

    Args:
        error_message: エラーメッセージ

    Returns:
        成功時True、失敗時False
    """
    message = f"エラー: {error_message}\n時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}"

    return send_status_notification(DAILY_PIPELINE_ERROR, message.split("\n"))


def send_webhook_file(file_path: str, title: str = "") -> bool:
    """
    ファイル（PNG等）を Discord Webhook にアップロードして送信する。

    Args:
        file_path: 送信するファイルの絶対パス
        title: 添付メッセージ（省略可）

    Returns:
        成功時True、失敗時False
    """
    if not os.path.exists(file_path):
        logger.warning("送信対象ファイルが存在しません: %s", file_path)
        return False

    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            payload = {"content": title} if title else {}
            response = _post_webhook(
                data_payload=payload,
                files={"file": (filename, f)},
                timeout=30,
            )
        if response is None:
            return False
        response.raise_for_status()
        logger.info("Discordファイル送信成功: %s", filename)
        return True
    except requests.exceptions.RequestException as e:
        logger.error("Discordファイル送信失敗: %s", e)
        return False


def send_miss_analysis_summary(
    miss_df,
    analysis_results: dict,
    since_days: int = 30,
) -> bool:
    """
    予測外れ原因分析サマリーを Discord Webhook に送信する。

    Args:
        miss_df: load_top_prediction_misses() の戻り値 DataFrame
        analysis_results: run_miss_analysis_batch() の戻り値辞書
        since_days: 分析対象期間（日数）

    Returns:
        成功時 True、失敗時 False
    """
    import pandas as pd

    if miss_df is None or (isinstance(miss_df, pd.DataFrame) and miss_df.empty):
        logger.info("外れ原因分析: データなし — 通知をスキップ")
        return True

    now = format_jst(fmt=DISCORD_DATE_FORMAT)
    lines = [f"**[予測外れ原因分析] {now} （直近{since_days}日）**\n"]

    for _, row in miss_df.iterrows():
        market = row["market"]
        symbol = row["symbol"]
        abs_err = row.get("abs_error", 0)
        pred = row.get("predicted_ratio", 0)
        actual = row.get("actual_ratio", 0)
        sign = "+" if pred >= 0 else ""
        actual_sign = "+" if actual >= 0 else ""
        lines.append(
            f"● `{market}/{symbol}` 外れ幅={abs_err:.2%}"
            f"  予測={sign}{pred:.2%} / 実績={actual_sign}{actual:.2%}"
        )
        causes = analysis_results.get((market, symbol), [])
        if causes:
            cause_parts = [f"{c.feature}(rank#{c.shap_rank},{c.miss_count}回)" for c in causes[:3]]
            lines.append(f"  主要因: {', '.join(cause_parts)}")

    # 全銘柄横断の繰り返し外れ要因
    from collections import Counter

    feature_counts: Counter = Counter()
    for causes in analysis_results.values():
        for c in causes:
            feature_counts[c.feature] += 1
    repeat_features = [(f, n) for f, n in feature_counts.items() if n >= 3]
    if repeat_features:
        lines.append("")
        repeat_strs = [f"{f}（{n}銘柄）" for f, n in sorted(repeat_features, key=lambda x: -x[1])]
        lines.append(f"⚠️ 繰り返し外れ要因: {', '.join(repeat_strs)}")

    return send_webhook_notification(
        title="予測外れ原因分析",
        message="\n".join(lines),
        color=0xFF8C00,
    )


def send_drift_alert(summary_df, horizon: int = 1, threshold: float = 0.45) -> bool:
    """
    モデルドリフト警告を Discord Webhook に送信する。

    方向正解率が threshold 以下の銘柄が存在する場合にのみ送信する。

    Args:
        summary_df: load_drift_summary() の戻り値 (DataFrame)
        horizon: 対象ホライズン（メッセージ表示用）
        threshold: 警告する方向正解率の閾値（デフォルト 0.45 = 45%）

    Returns:
        送信成功時 True、送信不要または失敗時 False
    """
    import pandas as pd

    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
        return False

    drift_rows = summary_df[summary_df["direction_accuracy"] <= threshold]
    if drift_rows.empty:
        logger.info("ドリフト警告なし (horizon=%sd, 閾値=%.0f%%)", horizon, threshold * 100)
        return False

    lines = [f"**[モデルドリフト警告] horizon={horizon}d (方向正解率 ≤ {threshold:.0%})**\n"]
    for _, row in drift_rows.iterrows():
        acc = row.get("direction_accuracy", 0)
        err = row.get("mean_abs_error", 0)
        n = int(row.get("n_samples", 0))
        lines.append(
            f"• `{row['market']}/{row['symbol']}` " f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}"
        )

    return send_webhook_text("\n".join(lines))


def send_accuracy_summary(summary_df, horizon: int = 1) -> bool:
    """
    予測精度サマリー（方向正解率・MAE）を Discord Webhook に送信する。

    Args:
        summary_df: load_drift_summary() の戻り値 (DataFrame)
        horizon: 対象ホライズン（メッセージ表示用）

    Returns:
        送信成功時 True、送信不要または失敗時 False
    """
    import pandas as pd

    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
        logger.info("精度サマリー送信スキップ: データなし (horizon=%sd)", horizon)
        return False

    now = format_jst(fmt=DISCORD_DATE_FORMAT)
    lines = [f"**[予測精度サマリー] {now} (horizon={horizon}d)**\n"]

    df_sorted = summary_df.sort_values("direction_accuracy", ascending=True)
    for _, row in df_sorted.iterrows():
        acc = row.get("direction_accuracy", 0)
        err = row.get("mean_abs_error", 0)
        n = int(row.get("n_samples", 0))
        lines.append(
            f"• `{row['market']}/{row['symbol']}` "
            f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}"
        )

    return send_webhook_text("\n".join(lines))


def send_weekly_report(
    accuracy_df=None, horizon: int = 1, diff_summary: Optional[dict] = None
) -> bool:
    """
    週次パフォーマンスレポートを Discord Webhook に送信する。

    直近の方向正解率・平均誤差を銘柄ごとに集計してレポートする。
    前週スナップショットがあれば先週比変化・改善/悪化上位も表示する。
    全体 Hit Rate が連続 N 週低下していた場合はアラートも送る。

    Args:
        accuracy_df: load_drift_summary() の戻り値 DataFrame（None の場合はDB から取得）
        horizon: 対象ホライズン

    Returns:
        成功時 True、失敗時 False
    """
    import pandas as pd

    from src.utils.db import load_drift_summary, load_paper_real_diff_summary
    from src.utils.db import load_weekly_accuracy_snapshots

    if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
        accuracy_df = load_drift_summary(horizon=horizon)

    if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
        logger.info("週次レポート: 精度データなし")
        return False

    now = format_jst(fmt=DISCORD_DATE_FORMAT)
    lines = [f"**[週次パフォーマンスレポート] {now} (horizon={horizon}d)**\n"]

    # 方向正解率でソート（低い順 = 要注意銘柄を先頭に）
    df_sorted = accuracy_df.sort_values("direction_accuracy", ascending=True)
    lines.append("**銘柄別 方向正解率（低い順）**")
    for _, row in df_sorted.head(20).iterrows():
        acc = row.get("direction_accuracy", 0)
        err = row.get("mean_abs_error", 0)
        n = int(row.get("n_samples", 0))
        flag = " ⚠️" if acc <= 0.45 else ""
        lines.append(
            f"• `{row['market']}/{row['symbol']}` " f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}{flag}"
        )

    # 全体サマリー
    mean_acc = accuracy_df["direction_accuracy"].mean()
    lines.append(f"\n**全体平均正解率**: {mean_acc:.1%} ({len(accuracy_df)}銘柄)")

    # 前週比較（週次スナップショットが 2 週分以上あれば）
    snapshots = load_weekly_accuracy_snapshots(n_weeks=4)
    if not snapshots.empty:
        weeks = sorted(snapshots["week_start"].unique(), reverse=True)
        if len(weeks) >= 2:
            prev_week = weeks[1]
            prev_df = snapshots[snapshots["week_start"] == prev_week][
                ["market", "symbol", "direction_accuracy"]
            ].rename(columns={"direction_accuracy": "prev_accuracy"})

            merged = accuracy_df.merge(prev_df, on=["market", "symbol"], how="inner")
            if not merged.empty:
                merged["delta"] = merged["direction_accuracy"] - merged["prev_accuracy"]

                top_improved = merged.nlargest(5, "delta")
                top_worsened = merged.nsmallest(5, "delta")

                lines.append(f"\n**前週比 Hit Rate 変化（比較週: {prev_week}）**")

                lines.append("改善上位 5 銘柄:")
                for _, r in top_improved.iterrows():
                    sign = "+" if r["delta"] >= 0 else ""
                    lines.append(
                        f"• `{r['market']}/{r['symbol']}` "
                        f"{r['prev_accuracy']:.1%} → {r['direction_accuracy']:.1%} "
                        f"({sign}{r['delta']:.1%})"
                    )

                lines.append("悪化上位 5 銘柄:")
                for _, r in top_worsened.iterrows():
                    sign = "+" if r["delta"] >= 0 else ""
                    lines.append(
                        f"• `{r['market']}/{r['symbol']}` "
                        f"{r['prev_accuracy']:.1%} → {r['direction_accuracy']:.1%} "
                        f"({sign}{r['delta']:.1%})"
                    )

        # 全体 Hit Rate の週次トレンドで連続低下を検出（N=3 週）
        _ALERT_CONSECUTIVE_DECLINE_WEEKS = 3
        weekly_mean = (
            snapshots.groupby("week_start")["direction_accuracy"].mean().sort_index(ascending=False)
        )
        if len(weekly_mean) >= _ALERT_CONSECUTIVE_DECLINE_WEEKS:
            recent = weekly_mean.iloc[:_ALERT_CONSECUTIVE_DECLINE_WEEKS].tolist()
            is_declining = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
            if is_declining:
                weeks_str = ", ".join(weekly_mean.index[:_ALERT_CONSECUTIVE_DECLINE_WEEKS].tolist())
                lines.append(
                    f"\n⚠️ **アラート: 全体 Hit Rate が {_ALERT_CONSECUTIVE_DECLINE_WEEKS} 週連続で低下しています**"
                    f" ({weeks_str})"
                )

    if diff_summary is None:
        diff_summary = load_paper_real_diff_summary(recent_days=7)
    if diff_summary.get("tracked_count", 0) > 0:
        lines.append("\n**paper/real 乖離サマリー（直近7日）**")
        lines.append(
            "• "
            f"tracked={diff_summary['tracked_count']}件, "
            f"comparable={diff_summary['comparable_count']}件"
        )
        lines.append(
            "• "
            f"平均paper slippage={diff_summary['avg_paper_slippage']:.3%}, "
            f"平均real slippage={diff_summary['avg_real_slippage']:.3%}"
        )
        lines.append(
            "• "
            f"平均価格差={diff_summary['avg_abs_price_diff']:.3f}, "
            f"平均乖離率={diff_summary['avg_abs_diff_ratio']:.3%}, "
            f"最大価格差={diff_summary['max_abs_price_diff']:.3f}"
        )

    return send_webhook_text_chunked("\n".join(lines))


def send_weekly_training_completion(models: list) -> bool:
    """
    週次モデル学習完了通知を Discord Webhook に送信する。

    Args:
        models: 学習したモデル名のリスト

    Returns:
        成功時 True、失敗時 False
    """
    models_str = "\n".join(f"• {m}" for m in models)
    message = f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}\n" f"学習済みモデル:\n{models_str}"
    return send_status_notification(WEEKLY_TRAINING_COMPLETION, message.split("\n"))


def send_daily_order_completion(
    buy_orders: int,
    sell_orders: int,
    mode: str = "paper",
    trading_stopped: bool = False,
    stop_reason: Optional[str] = None,
    daily_loss: Optional[float] = None,
    daily_loss_limit: Optional[float] = None,
) -> bool:
    """
    自動発注完了通知を Discord Webhook に送信する。

    Args:
        buy_orders: 買い注文数
        sell_orders: 売り注文数
        mode: 実行モード（paper / live）
        trading_stopped: リスクガードにより停止中かどうか
        stop_reason: 停止理由
        daily_loss: 当日損失額
        daily_loss_limit: 当日損失上限額

    Returns:
        成功時 True、失敗時 False
    """
    spec = get_daily_order_spec(trading_stopped=trading_stopped)
    lines = [
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
        f"モード: {mode}",
        f"買い注文: {buy_orders} 件",
        f"売り注文: {sell_orders} 件",
    ]
    if trading_stopped:
        lines.append(f"停止理由: {stop_reason or 'リスクガードにより停止'}")
        if daily_loss is not None and daily_loss_limit is not None:
            lines.append(f"当日損失: {daily_loss:.0f} 円 / 上限: {daily_loss_limit:.0f} 円")
    return send_status_notification(spec, lines)


def send_daily_settle_completion(settled_count: int) -> bool:
    """
    ペーパートレード約定処理完了通知を Discord Webhook に送信する。

    Args:
        settled_count: 約定処理した注文数

    Returns:
        成功時 True、失敗時 False
    """
    message = f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}\n" f"約定件数: {settled_count} 件"
    return send_status_notification(DAILY_SETTLE_COMPLETION, message.split("\n"))


def send_optimization_completion(success: int, failed: int) -> bool:
    """
    週次バックテスト最適化完了通知を Discord Webhook に送信する。

    Args:
        success: 最適化成功銘柄数
        failed: 最適化失敗銘柄数

    Returns:
        成功時 True、失敗時 False
    """
    spec = get_optimization_spec(failed=failed)
    status_icon = "⚠️" if failed > 0 else "✅"
    message = (
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}\n"
        f"成功: {success} 銘柄\n"
        f"失敗: {failed} 銘柄 {status_icon if failed > 0 else ''}\n"
        f"最適パラメータを `config/optimal_params.json` に保存しました"
    )
    return send_status_notification(spec, message.split("\n"))


def send_paper_trade_position_report(positions: list[dict], summary: dict) -> bool:
    """
    ペーパートレードのポジション・損益レポートを Discord Webhook に送信する。

    Args:
        positions: PaperBroker.get_positions() の戻り値
                   [{"symbol", "qty", "avg_price", "current_price", "unrealized_pnl"}, ...]
        summary: PaperBroker.get_pnl_summary() の戻り値
                 {"realized_pnl", "unrealized_pnl", "total_pnl", "balance",
                  "initial_balance", "trade_count", "started_at"}

    Returns:
        成功時 True、失敗時 False
    """
    now_str = format_jst(fmt=DISCORD_DATETIME_FORMAT)

    # ── ポジション一覧 ──────────────────────────────────
    if positions:
        lines = ["**銘柄         | 保有数 |  平均取得  |  現在値  |  含み損益**"]
        lines.append("─" * 56)
        for p in positions:
            sym = p["symbol"].ljust(8)
            qty = str(p["qty"]).rjust(5)
            avg = f"{p['avg_price']:,.0f}".rjust(8)
            cur = f"{p['current_price']:,.0f}".rjust(8)
            pnl = p["unrealized_pnl"]
            pnl_str = f"{pnl:+,.0f}".rjust(10)
            icon = "📈" if pnl >= 0 else "📉"
            lines.append(f"`{sym}` | {qty}株 | {avg}円 | {cur}円 | {pnl_str}円 {icon}")
        position_block = "\n".join(lines)
    else:
        position_block = "現在保有中のポジションはありません。"

    # ── 通算損益サマリー ────────────────────────────────
    realized = summary["realized_pnl"]
    unrealized = summary["unrealized_pnl"]
    total = summary["total_pnl"]
    balance = summary["balance"]
    initial = summary["initial_balance"]
    trade_count = summary["trade_count"]
    started_at = summary["started_at"] or "-"

    total_icon = "📈" if total >= 0 else "📉"
    return_rate = (total / initial * 100) if initial else 0.0

    summary_lines = [
        f"**[ペーパートレード損益レポート] {now_str}**",
        "",
        "**ポジション一覧**",
        position_block,
        "",
        "**通算損益サマリー**",
        f"• 実現損益　　: `{realized:+,.0f}円`",
        f"• 含み損益　　: `{unrealized:+,.0f}円`",
        f"• 通算損益　　: `{total:+,.0f}円` {total_icon}  ({return_rate:+.2f}%)",
        f"• 現在残高　　: `{balance:,.0f}円`",
        f"• 取引回数　　: `{trade_count}回`",
        f"• 運用開始日　: `{format_jst_from_iso(started_at, fallback='-')}`",
    ]
    return send_webhook_text_chunked("\n".join(summary_lines), preserve_lines=False)


def send_walk_forward_report_completion(result: dict) -> bool:
    """
    Walk-Forward 比較レポート完了通知を Discord Webhook に送信する。

    サマリー（成功/失敗件数）を embed で送信した後、
    Markdown 比較レポートの内容をテキストで分割送信する。

    Args:
        result: run_walk_forward_comparison_report() の戻り値辞書

    Returns:
        成功時 True、失敗時 False
    """
    success_count = result.get("success", 0)
    failed_count = result.get("failed", 0)
    total_count = result.get("total", 0)
    markdown_path = result.get("markdown_path")
    previous_path = result.get("previous_path")

    spec = get_walk_forward_report_spec(failed_count=failed_count)
    message_lines = [
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
        f"成功: {success_count} 銘柄 / 失敗: {failed_count} 銘柄 / 合計: {total_count} 銘柄",
        f"前回比較: {previous_path if previous_path else 'なし（初回実行）'}",
    ]
    ok = send_status_notification(spec, message_lines)

    if markdown_path:
        try:
            if not send_text_file_chunked(markdown_path, preserve_lines=False):
                ok = False
        except Exception as e:
            logger.error("Walk-Forwardレポートテキスト送信失敗: %s", e)
            ok = False

    return ok


def send_watchlist_update_report(diffs) -> bool:
    """
    ウォッチリスト更新結果を Discord Webhook に送信する。

    変更が1件もない場合は何も送信しない。

    Args:
        diffs: WatchlistDiff のリスト

    Returns:
        成功時 True、失敗時 False（送信不要な場合も True）
    """
    changed = [d for d in diffs if d.has_changes or d.removed_unverified]
    if not changed:
        logger.info("ウォッチリスト変更なし。Discord通知をスキップします。")
        return True

    lines = [f"🔄 **ウォッチリスト更新** — {format_jst(fmt=DISCORD_MINUTE_FORMAT)}"]
    lines.append("━" * 28)

    for diff in diffs:
        emoji = get_market_emoji(diff.market)
        lines.append(f"\n{emoji} **{diff.market.upper()}**")

        if diff.added:
            syms = ", ".join(f"`{s}`" for s in diff.added)
            lines.append(f"➕ 追加 ({len(diff.added)}銘柄): {syms}")

        if diff.removed:
            syms = ", ".join(f"`{s}`" for s in diff.removed)
            lines.append(f"➖ 削除 ({len(diff.removed)}銘柄): {syms}")

        if diff.removed_unverified:
            syms = ", ".join(f"`{s}`" for s in diff.removed_unverified)
            lines.append(f"⚠️ 指数除外・取引可能のため保留 ({len(diff.removed_unverified)}銘柄): {syms}")

        if diff.capped:
            lines.append("🛑 安全弁発動: 削除上限（10%）に達したため一部保留")

        total_after = len(diff.kept) + len(diff.added)
        lines.append(f"📋 合計: {total_after}銘柄")

    return send_webhook_text_chunked("\n".join(lines), preserve_lines=False)


def send_shap_notification(
    market: str,
    symbol: str,
    model_name: str,
    shap_top_bottom,
) -> bool:
    """
    SHAP特徴量寄与の上位・下位をDiscordに通知する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名
        shap_top_bottom: [feature, shap_mean, shap_rank] を含むDataFrame

    Returns:
        成功時True、失敗時False
    """
    import pandas as pd

    if not isinstance(shap_top_bottom, pd.DataFrame) or shap_top_bottom.empty:
        return False

    sorted_df = shap_top_bottom.sort_values("shap_rank")
    n_total = sorted_df["shap_rank"].max() if "shap_rank" in sorted_df.columns else len(sorted_df)
    top_df = sorted_df.head(10)
    bottom_df = sorted_df.tail(10).sort_values("shap_rank", ascending=False)

    lines = [f"**SHAP特徴量寄与 [{market}/{symbol}] {model_name}**"]
    lines.append(f"総特徴量数: {n_total}")
    lines.append("")
    lines.append("📈 **上位（寄与大）**")
    for _, row in top_df.iterrows():
        lines.append(f"  #{int(row['shap_rank']):>3} `{row['feature']}` — {row['shap_mean']:.6f}")
    lines.append("")
    lines.append("📉 **下位（寄与小）**")
    for _, row in bottom_df.iterrows():
        lines.append(f"  #{int(row['shap_rank']):>3} `{row['feature']}` — {row['shap_mean']:.6f}")

    return send_webhook_text_chunked("\n".join(lines), preserve_lines=False)


def send_shap_batch_summary(shap_results: list) -> bool:
    """
    複数銘柄の学習後に SHAP 特徴量寄与をまとめて 1 通で Discord に送信する。

    Args:
        shap_results: train_models_for_symbol の戻り値 dict の "shap_results" リストを
            連結したもの。各エントリは
            {"market": str, "symbol": str, "model_name": str, "shap_top_bottom": pd.DataFrame}

    Returns:
        送信成功時 True
    """
    import pandas as pd

    if not shap_results:
        return False

    now = format_jst(fmt="%Y/%m/%d %H:%M JST")
    lines = [f"**[SHAP 特徴量寄与サマリー] {now}** ({len(shap_results)} モデル)"]

    for entry in shap_results:
        market = entry.get("market", "?")
        symbol = entry.get("symbol", "?")
        model_name = entry.get("model_name", "?")
        shap_df = entry.get("shap_top_bottom")
        if not isinstance(shap_df, pd.DataFrame) or shap_df.empty:
            continue
        sorted_df = shap_df.sort_values("shap_rank")
        top3 = sorted_df.head(3)
        lines.append(f"\n**{market}/{symbol}** `{model_name}`")
        for _, row in top3.iterrows():
            lines.append(
                f"  #{int(row['shap_rank']):>3} `{row['feature']}` — {row['shap_mean']:.6f}"
            )

    if len(lines) <= 1:
        return False

    return send_webhook_text_chunked("\n".join(lines), preserve_lines=False)


def send_db_maintenance_completion(
    elapsed_seconds: float,
    size_before_mb: float,
    size_after_mb: float,
    error: Optional[str] = None,
) -> bool:
    """
    DB メンテナンス（CHECKPOINT / VACUUM）完了通知を Discord に送信する。

    Args:
        elapsed_seconds: 処理時間（秒）
        size_before_mb: 実行前 DB ファイルサイズ（MB）
        size_after_mb: 実行後 DB ファイルサイズ（MB）
        error: エラーメッセージ（None なら成功）

    Returns:
        成功時 True、失敗時 False
    """
    if error:
        spec = DB_MAINTENANCE_ERROR
        lines = [
            f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
            f"エラー: {error}",
        ]
    else:
        spec = DB_MAINTENANCE_COMPLETION
        diff_mb = size_after_mb - size_before_mb
        diff_str = f"{diff_mb:+.2f} MB"
        lines = [
            f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
            f"処理時間: {elapsed_seconds:.1f} 秒",
            f"DBサイズ: {size_before_mb:.2f} MB → {size_after_mb:.2f} MB ({diff_str})",
        ]
    return send_status_notification(spec, lines)


def send_drift_retrain_notification(
    triggered_symbols: list, mae_threshold: float, hit_rate_threshold: float
) -> bool:
    """
    ドリフト検知による自動再学習トリガー通知を Discord に送信する。

    Args:
        triggered_symbols: 再学習をトリガーした銘柄リスト
            (dicts: market, symbol, mean_abs_error, direction_accuracy)
        mae_threshold: 使用した MAE 閾値
        hit_rate_threshold: 使用した Hit Rate 閾値

    Returns:
        送信成功時 True
    """
    if not triggered_symbols:
        return False

    now = format_jst(fmt="%Y/%m/%d %H:%M JST")
    lines = [
        f"**[ドリフト検知・自動再学習トリガー] {now}**",
        f"MAE閾値={mae_threshold:.2%} / Hit Rate閾値={hit_rate_threshold:.0%}",
        f"対象銘柄数: {len(triggered_symbols)}",
        "",
    ]
    for sym in triggered_symbols:
        lines.append(
            f"• `{sym['market']}/{sym['symbol']}` "
            f"MAE={sym.get('mean_abs_error', 0):.4f} "
            f"HitRate={sym.get('direction_accuracy', 0):.1%}"
        )

    return send_webhook_text("\n".join(lines))


def send_feature_suggestion_notification(
    feature_suggestions: list,
    global_threshold: int = 2,
) -> bool:
    """
    ドリフト銘柄の特徴量除外提案を Discord Webhook に送信する。

    Args:
        feature_suggestions: [{"market": str, "symbol": str, "candidates": pd.DataFrame}]
            candidates は feature, importance_mean, importance_rank 列を持つ DataFrame
        global_threshold: グローバル警告の閾値（何銘柄以上で共通して除外候補か）

    Returns:
        送信成功時 True
    """
    from collections import Counter

    import pandas as pd

    if not feature_suggestions:
        return False

    now = format_jst(fmt=DISCORD_DATE_FORMAT)
    lines = [
        f"**[特徴量除外提案] {now}** — ドリフト銘柄 {len(feature_suggestions)} 銘柄",
        "次回学習で除外候補となっている特徴量を確認してください。",
        "",
    ]

    feature_counter: Counter = Counter()
    for entry in feature_suggestions:
        market = entry.get("market", "?")
        symbol = entry.get("symbol", "?")
        candidates_df = entry.get("candidates")
        if not isinstance(candidates_df, pd.DataFrame) or candidates_df.empty:
            lines.append(f"• `{market}/{symbol}` — 除外候補なし")
            continue
        lines.append(f"• `{market}/{symbol}` 除外候補 {len(candidates_df)} 件:")
        for _, row in candidates_df.iterrows():
            lines.append(
                f"  - `{row['feature']}` "
                f"(rank #{int(row['importance_rank'])}, mean={row['importance_mean']:.6f})"
            )
            feature_counter[row["feature"]] += 1

    global_features = [(f, n) for f, n in feature_counter.items() if n >= global_threshold]
    if global_features:
        lines.append(f"\n⚠️ **グローバル除外候補** ({global_threshold}銘柄以上で共通):")
        for feat, count in sorted(global_features, key=lambda x: -x[1]):
            lines.append(f"  - `{feat}` ({count}銘柄で除外候補)")

    return send_webhook_notification(
        title="特徴量除外提案",
        message="\n".join(lines),
        color=0xFF8C00,
    )


# ---------------------------------------------------------------------------
# ルールベーストレーディング通知
# ---------------------------------------------------------------------------


def send_rule_evaluation_completion(
    evaluated: int,
    effective: int,
    skipped: int,
    market: str,
) -> bool:
    """週次ルール評価完了通知を Discord に送信する。"""
    now = format_jst(fmt="%Y/%m/%d %H:%M JST")
    lines = [
        f"**[ルール評価完了] {now}**",
        f"マーケット: {market}",
        f"評価銘柄数: {evaluated}",
        f"有効ルール発見: {effective} 銘柄",
        f"スキップ (有効ルールなし): {skipped} 銘柄",
    ]
    return send_webhook_text("\n".join(lines))


def send_correlation_alert(
    enc: float,
    enc_threshold: float,
    avg_correlation: float,
    n_symbols: int,
    symbols: list[str],
) -> bool:
    """相関リスク上昇による新規エントリーブロックを Discord に通知する。

    Args:
        enc: 実効分散度（ENC）の現在値
        enc_threshold: ENC の閾値
        avg_correlation: 保有銘柄間の平均絶対相関係数
        n_symbols: 保有銘柄数
        symbols: 保有銘柄コードのリスト

    Returns:
        送信成功時 True
    """
    now = format_jst(fmt=DISCORD_MINUTE_FORMAT)
    lines = [
        f"**[相関リスク警告] {now}**",
        f"ENC={enc:.2f} < 閾値={enc_threshold:.2f}（新規エントリーをブロック）",
        f"平均相関係数: {avg_correlation:.2f}",
        f"保有銘柄数: {n_symbols}",
    ]
    if symbols:
        syms = ", ".join(f"`{s}`" for s in symbols[:10])
        lines.append(f"保有銘柄: {syms}")

    return send_webhook_notification(
        "相関リスク警告 — 分散度低下",
        "\n".join(lines),
        color=0xFF6600,
    )


def send_rule_daily_signals(
    signals: list[dict],
    market: str,
    buy_orders: int,
    sell_orders: int,
) -> bool:
    """ルールベース日次シグナル通知を Discord に送信する。"""
    now = format_jst(fmt="%Y/%m/%d %H:%M JST")
    buy_signals = [s for s in signals if s["signal"] == 1]
    sell_signals = [s for s in signals if s["signal"] == -1]

    lines = [
        f"**[ルールシグナル] {now}  ({market})**",
        f"BUY候補: {len(buy_signals)}銘柄  |  SELL候補: {len(sell_signals)}銘柄",
        f"ペーパー発注: BUY={buy_orders}  SELL={sell_orders}",
        "",
    ]

    if buy_signals:
        lines.append("**BUY シグナル:**")
        for s in buy_signals:
            price_str = f"{s['price']:,.0f}円" if s.get("price") else "---"
            lines.append(
                f"  • `{s['symbol']}` [{s['rule']}]  " f"価格={price_str}  勝率={s['win_rate']:.1%}"
            )

    if sell_signals:
        lines.append("")
        lines.append("**SELL シグナル:**")
        for s in sell_signals:
            price_str = f"{s['price']:,.0f}円" if s.get("price") else "---"
            lines.append(
                f"  • `{s['symbol']}` [{s['rule']}]  " f"価格={price_str}  勝率={s['win_rate']:.1%}"
            )

    if not buy_signals and not sell_signals:
        lines.append("本日はシグナルなし（全銘柄 HOLD）")

    return send_webhook_text("\n".join(lines))


def send_pre_close_alert(alerts: list) -> bool:
    """
    引け前ポジション再評価アラートを Discord に送信する。

    Args:
        alerts: evaluate_positions() の戻り値（PositionAlert のリスト）

    Returns:
        送信成功時 True
    """
    from src.trading.pre_close_alert_service import AlertType

    now = format_jst(fmt=DISCORD_MINUTE_FORMAT)

    if not alerts:
        return send_status_notification(PRE_CLOSE_ALERT, [f"時刻: {now}", "保有ポジションなし"])

    stop_loss = [a for a in alerts if a.alert_type == AlertType.STOP_LOSS]
    take_profit = [a for a in alerts if a.alert_type == AlertType.TAKE_PROFIT]
    hold = [a for a in alerts if a.alert_type == AlertType.HOLD]

    lines = [
        f"**[引け前ポジション再評価] {now}**",
        f"保有: {len(alerts)}銘柄  |  損切り: {len(stop_loss)}  利確: {len(take_profit)}  保有継続: {len(hold)}",
        "",
    ]

    if stop_loss:
        lines.append("🔴 **損切り推奨（SL接近）**")
        for a in stop_loss:
            lines.append(
                f"  • `{a.symbol}` "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み損益={a.unrealized_pnl:+,.0f}円"
            )

    if take_profit:
        lines.append("")
        lines.append("🟡 **利確推奨（目標値超）**")
        for a in take_profit:
            lines.append(
                f"  • `{a.symbol}` "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み益={a.unrealized_pnl:+,.0f}円({a.hold_pnl_ratio:+.1%})"
            )

    if hold:
        lines.append("")
        lines.append("🟢 **保有継続**")
        for a in hold:
            lines.append(
                f"  • `{a.symbol}` "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み損益={a.unrealized_pnl:+,.0f}円"
            )

    return send_webhook_text_chunked("\n".join(lines), preserve_lines=False)


def send_shadow_evaluation_notification(result: dict) -> bool:
    """
    A/B テスト（シャドーモード）評価結果を Discord Webhook に送信する。

    challenger_wins=True のとき昇格候補として通知する。

    Args:
        result: evaluate_shadow_models() の戻り値

    Returns:
        成功時 True、失敗時 False
    """
    challenger_wins = result.get("challenger_wins", False)
    spec = SHADOW_EVALUATION_CHALLENGER_WINS if challenger_wins else SHADOW_EVALUATION_NO_WINNER

    def _fmt(v) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    lines = [
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
        f"Production  — Hit Rate: {_fmt(result.get('production_hit_rate'))} / Sharpe: {_fmt(result.get('production_sharpe'))} (n={result.get('n_production', 0)})",
        f"Challenger  — Hit Rate: {_fmt(result.get('challenger_hit_rate'))} / Sharpe: {_fmt(result.get('challenger_sharpe'))} (n={result.get('n_challenger', 0)})",
    ]
    if challenger_wins:
        lines.append("→ Challenger が上回りました。手動承認後に promote_challenger_to_production() を実行してください。")

    return send_status_notification(spec, lines)
