"""
Discord通知ユーティリティ

Webhookを使用したDiscord通知機能
"""

import logging
import os
from typing import Optional

import requests

from src.domain.types import PredictionResult
from src.reporting.discord.discord_formatters import convert_df_for_discord, get_market_emoji
from src.reporting.discord.discord_notification_specs import (
    DAILY_PIPELINE_COMPLETION,
    DAILY_PIPELINE_ERROR,
    DAILY_SETTLE_COMPLETION,
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
from src.services.discord_query_service import get_latest_market_prediction_snapshots
from src.utils.japan_time import format_jst, format_jst_from_iso, isoformat_jst

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
        embed_data = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": isoformat_jst(),
                }
            ]
        }

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


def send_weekly_report(
    accuracy_df=None, horizon: int = 1, diff_summary: Optional[dict] = None
) -> bool:
    """
    週次パフォーマンスレポートを Discord Webhook に送信する。

    直近の方向正解率・平均誤差を銘柄ごとに集計してレポートする。

    Args:
        accuracy_df: load_drift_summary() の戻り値 DataFrame（None の場合はDB から取得）
        horizon: 対象ホライズン

    Returns:
        成功時 True、失敗時 False
    """
    import pandas as pd

    from src.utils.db import load_drift_summary, load_paper_real_diff_summary

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
