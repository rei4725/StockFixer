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
    HIT_RATE_DRIFT_ALERT,
    SHADOW_EVALUATION_CHALLENGER_WINS,
    SHADOW_EVALUATION_NO_WINNER,
    NotificationSpec,
    get_daily_order_spec,
    get_optimization_spec,
)
from src.reporting.discord.discord_text import (  # noqa: F401  # re-export（後方互換）
    DISCORD_DATE_FORMAT,
    DISCORD_DATETIME_FORMAT,
    DISCORD_MINUTE_FORMAT,
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
            f"• `{row['market']}/{row['symbol']}` " f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}"
        )

    return send_webhook_text("\n".join(lines))


def send_promotion_result(promoted: bool, reason: str, criteria: dict) -> bool:
    """
    モデル昇格評価結果を Discord Webhook に送信する。

    Args:
        promoted: 実際に昇格が実行されたかどうか
        reason: 判定理由サマリー文
        criteria: 昇格基準の詳細 dict（promotion_gate.evaluate_promotion() の戻り値と同形式）

    Returns:
        成功時 True、失敗時 False
    """
    status = "昇格実行" if promoted else "見送り"
    lines = [
        f"**[週次] モデル昇格評価: {status}**",
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
        reason,
        "```",
    ]
    criteria_labels = {
        "net_return": "Net Return",
        "mdd": "MDD",
        "sharpe": "Sharpe",
        "hit_rate": "Hit Rate",
        "slippage": "Slippage",
    }
    for key, c in criteria.items():
        label = criteria_labels.get(key, key)
        mark = "OK" if c.get("passed") else "NG"
        lines.append(f"[{mark}] {label}")
    lines.append("```")
    return send_webhook_text_chunked("\n".join(lines))


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
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "✅ 成功", "value": f"{success:,} 銘柄", "inline": True},
        {"name": f"{status_icon} 失敗", "value": f"{failed:,} 銘柄", "inline": True},
        {
            "name": "💾 保存先",
            "value": "最適パラメータを `config/optimal_params.json` に保存しました",
            "inline": False,
        },
    ]
    return send_status_fields(spec, fields)


def send_factory_completion(
    market: str,
    evaluated: int,
    passed: int,
    champion_sharpe: float,
    pbo: float,
    best_label: str,
    best_sharpe: float,
    report_hashes: list[str],
) -> bool:
    """
    戦略ファクトリー夜間バッチ（#369）の完了通知を Discord Webhook に送信する。

    Args:
        market: 対象マーケット
        evaluated: 評価した候補仮説数
        passed: ゲート合格数
        champion_sharpe: 対照（チャンピオン）の最高 Sharpe
        pbo: バッチ全体の PBO
        best_label: 最高 Sharpe 候補のラベル
        best_sharpe: 最高 Sharpe 候補の値
        report_hashes: 合格仮説のハッシュ一覧（レポートファイル名）

    Returns:
        成功時 True、失敗時 False
    """
    icon = "🏭✨" if passed > 0 else "🏭"
    spec = NotificationSpec(
        title=f"{icon} 戦略ファクトリー夜間バッチ完了 ({market})",
        color=0x00BFFF if passed > 0 else 0x808080,
    )
    passed_value = "\n".join(f"`{h}.json`" for h in report_hashes) if report_hashes else "なし"
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "🧪 評価仮説", "value": f"{evaluated} 本", "inline": True},
        {"name": "✅ ゲート合格", "value": f"{passed} 本", "inline": True},
        {"name": "👑 チャンピオン Sharpe", "value": f"{champion_sharpe:.3f}", "inline": True},
        {"name": "📉 バッチ PBO", "value": f"{pbo:.3f}", "inline": True},
        {
            "name": "🥇 ベスト候補",
            "value": f"{best_label} (Sharpe {best_sharpe:.3f})",
            "inline": False,
        },
        {"name": "📄 レポート (results/factory/reports/)", "value": passed_value, "inline": False},
    ]
    return send_status_fields(spec, fields)


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


def send_hit_rate_drift_alert(result) -> bool:
    """
    週次 Hit Rate ドリフト検知結果を Discord Webhook に送信する（R-274）。

    is_drifted=False の場合は送信しない。

    Args:
        result: check_weekly_hit_rate_drift() が返す DriftMonitorResult

    Returns:
        送信成功時 True、送信不要または失敗時 False
    """
    if not result.is_drifted:
        logger.info(
            "Hit Rate ドリフト警告なし: week=%s drop=%.2f%%",
            result.current_week,
            (result.drop_ratio or 0) * 100,
        )
        return False

    def _pct(val) -> str:
        return f"{val * 100:.1f}%" if val is not None else "N/A"

    fields: list[dict] = [
        {"name": "📅 週", "value": result.current_week or "N/A", "inline": True},
        {"name": "🎯 当週 Hit Rate", "value": _pct(result.current_hit_rate), "inline": True},
        {
            "name": f"📊 過去 {result.alert_weeks} 週平均",
            "value": _pct(result.avg_hit_rate),
            "inline": True,
        },
        {"name": "📉 低下率", "value": _pct(result.drop_ratio), "inline": True},
        {"name": "🚧 閾値", "value": _pct(result.alert_threshold), "inline": True},
    ]
    return send_status_fields(
        HIT_RATE_DRIFT_ALERT,
        fields,
        description="再学習・モデル切り替えをご検討ください。",
    )


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

    prod_value = (
        f"Hit Rate: {_fmt(result.get('production_hit_rate'))}"
        f" / Sharpe: {_fmt(result.get('production_sharpe'))}"
        f" (n={result.get('n_production', 0):,})"
    )
    chal_value = (
        f"Hit Rate: {_fmt(result.get('challenger_hit_rate'))}"
        f" / Sharpe: {_fmt(result.get('challenger_sharpe'))}"
        f" (n={result.get('n_challenger', 0):,})"
    )
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "🏭 Production", "value": prod_value, "inline": False},
        {"name": "🧪 Challenger", "value": chal_value, "inline": False},
    ]
    description = ""
    if challenger_wins:
        description = (
            "→ Challenger が上回りました。"
            "手動承認後に promote_challenger_to_production() を実行してください。"
        )

    return send_status_fields(spec, fields, description=description)
