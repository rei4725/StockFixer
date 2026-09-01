"""
Discord通知ユーティリティ

Webhookを使用したDiscord通知機能
"""

import logging
from typing import Optional

from src.reporting.discord.discord_formatters import build_prediction_list, get_market_emoji
from src.reporting.discord.discord_notification_specs import (
    COLOR_INFO,
    DAILY_PIPELINE_COMPLETION,
    DAILY_PIPELINE_ERROR,
    DAILY_SETTLE_COMPLETION,
    DB_BACKUP_COMPLETION,
    DB_BACKUP_ERROR,
    DB_MAINTENANCE_COMPLETION,
    DB_MAINTENANCE_ERROR,
    get_daily_order_spec,
)
from src.reporting.discord.discord_text import (  # noqa: F401  # re-export（後方互換）
    DISCORD_DATE_FORMAT,
    DISCORD_DATETIME_FORMAT,
    DISCORD_MINUTE_FORMAT,
)
from src.reporting.discord.notifications_drift import (  # noqa: F401  # re-export（#497 第3弾）
    send_accuracy_summary,
    send_correlation_alert,
    send_drift_alert,
    send_drift_retrain_notification,
    send_hit_rate_drift_alert,
    send_miss_analysis_summary,
)
from src.reporting.discord.notifications_model import (  # noqa: F401  # re-export（#497 第4弾）
    send_allocation_rebalance_report,
    send_factory_completion,
    send_feature_suggestion_notification,
    send_optimization_completion,
    send_promotion_result,
    send_shadow_evaluation_notification,
    send_shap_batch_summary,
    send_shap_notification,
    send_strategy_promotion_detected,
)
from src.reporting.discord.notifications_report import (  # noqa: F401  # re-export（#497 第2弾）
    send_monthly_report_notification,
    send_walk_forward_report_completion,
    send_weekly_report,
    send_weekly_training_completion,
)
from src.reporting.discord.webhook_sender import (  # noqa: F401  # re-export + ドメイン関数の送信基盤
    _get_webhook_url,
    _post_webhook,
    send_status_fields,
    send_status_notification,
    send_text_file_chunked,
    send_webhook_file,
    send_webhook_notification,
    send_webhook_text,
    send_webhook_text_chunked,
)
from src.reporting.query_service import get_latest_market_prediction_snapshots
from src.utils.japan_time import format_jst, format_jst_from_iso

logger = logging.getLogger(__name__)


def _append_llm_reasons(parts: list[str], results: list, max_count: int = 3) -> None:
    """上位銘柄の LLM 推薦理由を parts リストに追記する。

    Ollama が未接続の場合は何もしない。
    """
    try:
        from src.reporting.llm_reason import generate_reasons_for_top  # noqa: PLC0415

        reasons = generate_reasons_for_top(results, max_count=max_count)
        if not reasons:
            return
        lines = ["💡 推薦理由（上位3銘柄）"]
        for symbol, reason in reasons.items():
            lines.append(f"・{symbol}: {reason}")
        parts.append("\n".join(lines))
    except Exception as e:
        logger.debug("LLM 推薦理由生成をスキップ: %s", e)


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
    # 1. 完了メッセージを embed fields（2カラムグリッド）で送信
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
    ]
    if data_count is not None:
        fields.append({"name": "📊 取得データ", "value": f"{data_count:,} 件", "inline": True})
    if prediction_markets:
        markets_str = " ".join(f"{get_market_emoji(m)}{m.upper()}" for m in prediction_markets)
        fields.append({"name": "🌐 予測市場", "value": markets_str, "inline": True})

    success = send_status_fields(DAILY_PIPELINE_COMPLETION, fields)

    # 2. 予測結果をマーケット単位の「リスト型」embed で送信
    if include_forecast:
        try:
            latest_ts, snapshots = get_latest_market_prediction_snapshots()
            if latest_ts and snapshots:
                ts_label = latest_ts[:16] if len(latest_ts) >= 16 else latest_ts
                for snapshot in snapshots:
                    _send_prediction_snapshot(snapshot, ts_label)
        except Exception as e:
            logger.error("予測結果送信失敗: %s", e, exc_info=True)

    return success


def _send_prediction_snapshot(snapshot, ts_label: str) -> None:
    """1マーケット分の予測（上位/下位）をリスト型 embed で送信する。"""
    parts: list[str] = []
    if snapshot.top_results:
        parts.append("📈 **上位（予想上昇）**")
        parts.extend(build_prediction_list(snapshot.top_results, max_n=10))
        _append_llm_reasons(parts, snapshot.top_results)
    if snapshot.worst_results:
        if parts:
            parts.append("")
        parts.append("📉 **下位（予想下落）**")
        parts.extend(build_prediction_list(snapshot.worst_results, max_n=10))

    if not parts:
        return

    emoji = get_market_emoji(snapshot.market)
    title = f"{emoji} {snapshot.market.upper()} 予測 — {ts_label}"
    send_webhook_notification(title, "\n".join(parts), color=COLOR_INFO)


def send_daily_pipeline_error(error_message: str) -> bool:
    """
    日次パイプラインエラー通知

    Args:
        error_message: エラーメッセージ

    Returns:
        成功時True、失敗時False
    """
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "❌ エラー", "value": error_message, "inline": False},
    ]

    return send_status_fields(DAILY_PIPELINE_ERROR, fields)


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
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "⚙️ モード", "value": mode, "inline": True},
        {"name": "🟢 買い注文", "value": f"{buy_orders:,} 件", "inline": True},
        {"name": "🔴 売り注文", "value": f"{sell_orders:,} 件", "inline": True},
    ]
    if trading_stopped:
        fields.append(
            {
                "name": "⛔ 停止理由",
                "value": stop_reason or "リスクガードにより停止",
                "inline": False,
            }
        )
        if daily_loss is not None and daily_loss_limit is not None:
            fields.append(
                {
                    "name": "💰 当日損失",
                    "value": f"{daily_loss:,.0f} 円 / 上限: {daily_loss_limit:,.0f} 円",
                    "inline": True,
                }
            )
    return send_status_fields(spec, fields)


def send_daily_settle_completion(settled_count: int) -> bool:
    """
    ペーパートレード約定処理完了通知を Discord Webhook に送信する。

    Args:
        settled_count: 約定処理した注文数

    Returns:
        成功時 True、失敗時 False
    """
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "📊 約定件数", "value": f"{settled_count:,} 件", "inline": True},
    ]
    return send_status_fields(DAILY_SETTLE_COMPLETION, fields)


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
            lines.append(
                f"⚠️ 指数除外・取引可能のため保留 ({len(diff.removed_unverified)}銘柄): {syms}"
            )

        if diff.capped:
            lines.append("🛑 安全弁発動: 削除上限（10%）に達したため一部保留")

        total_after = len(diff.kept) + len(diff.added)
        lines.append(f"📋 合計: {total_after}銘柄")

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
        fields: list[dict] = [
            {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
            {"name": "❌ エラー", "value": error, "inline": False},
        ]
    else:
        spec = DB_MAINTENANCE_COMPLETION
        diff_mb = size_after_mb - size_before_mb
        fields = [
            {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
            {"name": "⏱ 処理時間", "value": f"{elapsed_seconds:.1f} 秒", "inline": True},
            {
                "name": "💾 DBサイズ",
                "value": f"{size_before_mb:,.2f} MB → {size_after_mb:,.2f} MB ({diff_mb:+,.2f} MB)",
                "inline": False,
            },
        ]
    return send_status_fields(spec, fields)


def send_backup_completion(
    backup_path: str,
    size_mb: float,
    elapsed_seconds: float,
    pruned_count: int,
    error: Optional[str] = None,
) -> bool:
    """
    DB バックアップ完了通知を Discord に送信する。

    Args:
        backup_path: バックアップ先ファイルパス
        size_mb: バックアップファイルサイズ（MB）
        elapsed_seconds: 処理時間（秒）
        pruned_count: 削除した旧世代数
        error: エラーメッセージ（None なら成功）

    Returns:
        成功時 True、失敗時 False
    """
    if error:
        spec = DB_BACKUP_ERROR
        fields: list[dict] = [
            {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
            {"name": "❌ エラー", "value": error, "inline": False},
        ]
    else:
        spec = DB_BACKUP_COMPLETION
        fields = [
            {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
            {"name": "⏱ 処理時間", "value": f"{elapsed_seconds:.1f} 秒", "inline": True},
            {"name": "💾 サイズ", "value": f"{size_mb:,.2f} MB", "inline": True},
            {"name": "🗑 削除世代", "value": f"{pruned_count:,} 件", "inline": True},
            {"name": "📁 保存先", "value": backup_path, "inline": False},
        ]
    return send_status_fields(spec, fields)


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
