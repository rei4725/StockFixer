"""
Discord通知ユーティリティ

Webhookを使用したDiscord通知機能
"""

import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


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
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。" "Webhook通知をスキップします。")
        return False

    try:
        embed_data = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        }

        response = requests.post(webhook_url, json=embed_data, timeout=10)
        response.raise_for_status()

        logger.info(f"Discord通知送信成功: {title}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Discord通知送信失敗: {e}")
        return False


def send_webhook_text(text: str) -> bool:
    """
    プレーンテキストメッセージをWebhookで送信する

    Args:
        text: 送信テキスト（コードフェンス込み）

    Returns:
        成功時True、失敗時False
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。" "Webhook通知をスキップします。")
        return False

    try:
        payload = {"content": text}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info("Discord通知送信成功: テキストメッセージ")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Discord通知送信失敗: {e}")
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
    from src.api.discord_bot import convert_df_for_discord
    from src.utils.db import (
        load_latest_prediction_timestamp,
        load_prediction_markets,
        load_prediction_results,
    )

    # 1. 完了メッセージを送信
    title = "✅ 日次パイプライン完了"

    message_lines = [
        f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if data_count is not None:
        message_lines.append(f"取得データ: {data_count}件")

    if prediction_markets:
        markets_str = "、".join(prediction_markets)
        message_lines.append(f"予測市場: {markets_str}")

    message = "\n".join(message_lines)

    success = send_webhook_notification(title, message, color=0x00FF00)

    # 2. 予測結果テーブルを送信（マーケット単位でTop→Worst、1メッセージにまとめる）
    _MARKET_EMOJI = {"JP": "🇯🇵", "NASDAQ": "🇺🇸", "US": "🇺🇸"}
    if include_forecast:
        try:
            latest_ts = load_latest_prediction_timestamp()
            if latest_ts:
                markets = load_prediction_markets(latest_ts)

                if markets:
                    ts_label = latest_ts[:16] if len(latest_ts) >= 16 else latest_ts
                    parts = [f"📊 予測結果 — {ts_label}\n{'━' * 28}"]

                    # マーケット単位で上位10 → 下位10の順に出力
                    for market in sorted(markets):
                        emoji = _MARKET_EMOJI.get(market, "🌐")
                        parts.append(f"\n{emoji} {market}")

                        df_top = load_prediction_results(
                            predicted_at=latest_ts, market=market, top_n=10
                        )
                        if df_top is not None and not df_top.empty:
                            df_top = convert_df_for_discord(df_top)
                            parts.append(f"📈 上位10銘柄\n```\n{df_top.to_string(index=False)}\n```")

                        df_worst = load_prediction_results(
                            predicted_at=latest_ts, market=market, worst_n=10
                        )
                        if df_worst is not None and not df_worst.empty:
                            df_worst = convert_df_for_discord(df_worst)
                            parts.append(f"📉 下位10銘柄\n```\n{df_worst.to_string(index=False)}\n```")

                    if len(parts) > 1:
                        msg = "\n".join(parts)
                        # Discordメッセージ長制限対応（超過時のみ分割）
                        max_length = 3800
                        for i in range(0, len(msg), max_length):
                            send_webhook_text(msg[i : i + max_length])

        except Exception as e:
            logger.error(f"予測結果テーブル送信失敗: {e}")

    return success


def send_daily_pipeline_error(error_message: str) -> bool:
    """
    日次パイプラインエラー通知

    Args:
        error_message: エラーメッセージ

    Returns:
        成功時True、失敗時False
    """
    title = "❌ 日次パイプライン失敗"

    message = f"エラー: {error_message}\n時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return send_webhook_notification(title, message, color=0xFF0000)


def send_webhook_file(file_path: str, title: str = "") -> bool:
    """
    ファイル（PNG等）を Discord Webhook にアップロードして送信する。

    Args:
        file_path: 送信するファイルの絶対パス
        title: 添付メッセージ（省略可）

    Returns:
        成功時True、失敗時False
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。ファイル送信をスキップします。")
        return False

    if not os.path.exists(file_path):
        logger.warning(f"送信対象ファイルが存在しません: {file_path}")
        return False

    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            payload = {"content": title} if title else {}
            response = requests.post(
                webhook_url,
                data=payload,
                files={"file": (filename, f)},
                timeout=30,
            )
        response.raise_for_status()
        logger.info(f"Discordファイル送信成功: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Discordファイル送信失敗: {e}")
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
        logger.info(f"ドリフト警告なし (horizon={horizon}d, 閾値={threshold:.0%})")
        return False

    lines = [f"**[モデルドリフト警告] horizon={horizon}d (方向正解率 ≤ {threshold:.0%})**\n"]
    for _, row in drift_rows.iterrows():
        acc = row.get("direction_accuracy", 0)
        err = row.get("mean_abs_error", 0)
        n = int(row.get("n_samples", 0))
        lines.append(
            f"• `{row['market']}/{row['symbol']}` " f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}"
        )

    message = "\n".join(lines)
    return send_webhook_text(message)


def send_weekly_report(accuracy_df=None, horizon: int = 1) -> bool:
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

    from src.utils.db import load_drift_summary

    if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
        accuracy_df = load_drift_summary(horizon=horizon)

    if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
        logger.info("週次レポート: 精度データなし")
        return False

    from datetime import datetime

    now = datetime.now().strftime("%Y/%m/%d")
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

    message = "\n".join(lines)
    # Discord の 2000 文字制限に対応した分割送信
    chunks: list[str] = []
    current = ""
    for line in message.split("\n"):
        if len(current) + len(line) + 1 > 1900:
            chunks.append(current)
            current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)

    success = True
    for chunk in chunks:
        if not send_webhook_text(chunk):
            success = False
    return success


def send_weekly_training_completion(models: list) -> bool:
    """
    週次モデル学習完了通知を Discord Webhook に送信する。

    Args:
        models: 学習したモデル名のリスト

    Returns:
        成功時 True、失敗時 False
    """
    title = "✅ 週次モデル学習完了"
    models_str = "\n".join(f"• {m}" for m in models)
    message = f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" f"学習済みモデル:\n{models_str}"
    return send_webhook_notification(title, message, color=0x00FF00)


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
    title = "⚠️ 自動発注停止" if trading_stopped else "✅ 自動発注完了"
    lines = [
        f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"モード: {mode}",
        f"買い注文: {buy_orders} 件",
        f"売り注文: {sell_orders} 件",
    ]
    if trading_stopped:
        lines.append(f"停止理由: {stop_reason or 'リスクガードにより停止'}")
        if daily_loss is not None and daily_loss_limit is not None:
            lines.append(f"当日損失: {daily_loss:.0f} 円 / 上限: {daily_loss_limit:.0f} 円")
    message = "\n".join(lines)
    color = 0xFF9900 if trading_stopped else 0x00BFFF
    return send_webhook_notification(title, message, color=color)


def send_daily_settle_completion(settled_count: int) -> bool:
    """
    ペーパートレード約定処理完了通知を Discord Webhook に送信する。

    Args:
        settled_count: 約定処理した注文数

    Returns:
        成功時 True、失敗時 False
    """
    title = "✅ 約定処理完了"
    message = f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" f"約定件数: {settled_count} 件"
    return send_webhook_notification(title, message, color=0x00BFFF)


def send_optimization_completion(success: int, failed: int) -> bool:
    """
    週次バックテスト最適化完了通知を Discord Webhook に送信する。

    Args:
        success: 最適化成功銘柄数
        failed: 最適化失敗銘柄数

    Returns:
        成功時 True、失敗時 False
    """
    title = "✅ 週次バックテスト最適化完了"
    status_icon = "⚠️" if failed > 0 else "✅"
    message = (
        f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"成功: {success} 銘柄\n"
        f"失敗: {failed} 銘柄 {status_icon if failed > 0 else ''}\n"
        f"最適パラメータを `config/optimal_params.json` に保存しました"
    )
    color = 0x00FF00 if failed == 0 else 0xFFAA00
    return send_webhook_notification(title, message, color=color)


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
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        f"• 運用開始日　: `{started_at}`",
    ]
    message = "\n".join(summary_lines)

    # 2000文字制限対応で分割送信
    success = True
    max_len = 1900
    for i in range(0, len(message), max_len):
        if not send_webhook_text(message[i : i + max_len]):
            success = False
    return success


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

    status_icon = "⚠️" if failed_count > 0 else "✅"
    title = f"{status_icon} Walk-Forward 比較レポート完了"
    message_lines = [
        f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"成功: {success_count} 銘柄 / 失敗: {failed_count} 銘柄 / 合計: {total_count} 銘柄",
        f"前回比較: {previous_path if previous_path else 'なし（初回実行）'}",
    ]
    color = 0x00FF00 if failed_count == 0 else 0xFFAA00
    ok = send_webhook_notification(title, "\n".join(message_lines), color=color)

    if markdown_path:
        try:
            md_text = open(markdown_path, encoding="utf-8").read()
            max_len = 1900
            for i in range(0, len(md_text), max_len):
                if not send_webhook_text(md_text[i : i + max_len]):
                    ok = False
        except Exception as e:
            logger.error(f"Walk-Forwardレポートテキスト送信失敗: {e}")
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

    lines = [f"🔄 **ウォッチリスト更新** — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    lines.append("━" * 28)

    _MARKET_EMOJI = {"us": "🇺🇸", "jp": "🇯🇵"}

    for diff in diffs:
        emoji = _MARKET_EMOJI.get(diff.market.lower(), "🌐")
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

    message = "\n".join(lines)
    max_len = 1900
    success = True
    for i in range(0, len(message), max_len):
        if not send_webhook_text(message[i : i + max_len]):
            success = False
    return success


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

    message = "\n".join(lines)
    max_len = 1900
    success = True
    for i in range(0, len(message), max_len):
        if not send_webhook_text(message[i : i + max_len]):
            success = False
    return success


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

    from datetime import datetime

    now = datetime.now().strftime("%Y/%m/%d %H:%M")
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
