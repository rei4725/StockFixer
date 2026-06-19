"""
Discord レポート通知

週次パフォーマンス / 週次学習完了 / Walk-Forward / 月次レポートの通知関数群。
discord_utils.py の段階的分割（#497 第2弾）で抽出。送信基盤は webhook_sender に依存する。
"""

import logging
from typing import Optional

from src.reporting.discord.discord_notification_specs import (
    MONTHLY_REPORT_COMPLETION,
    WEEKLY_TRAINING_COMPLETION,
    get_walk_forward_report_spec,
)
from src.reporting.discord.discord_text import DISCORD_DATE_FORMAT, DISCORD_DATETIME_FORMAT
from src.reporting.discord.webhook_sender import (
    send_status_fields,
    send_text_file_chunked,
    send_webhook_text_chunked,
)
from src.utils.japan_time import format_jst

logger = logging.getLogger(__name__)


def send_weekly_report(
    accuracy_df=None,
    horizon: int = 1,
    diff_summary: Optional[dict] = None,
    llm_review: Optional[str] = None,
) -> bool:
    """
    週次パフォーマンスレポートを Discord Webhook に送信する。

    直近の方向正解率・平均誤差を銘柄ごとに集計してレポートする。
    前週スナップショットがあれば先週比変化・改善/悪化上位も表示する。
    全体 Hit Rate が連続 N 週低下していた場合はアラートも送る。

    Args:
        accuracy_df: load_drift_summary() の戻り値 DataFrame（None の場合はDB から取得）
        horizon: 対象ホライズン
        diff_summary: paper/real 乖離サマリー（None の場合は DB から取得）
        llm_review: Claude が生成した講評テキスト（None の場合は講評節を省略）

    Returns:
        成功時 True、失敗時 False
    """
    import pandas as pd

    from src.utils.db import (
        load_drift_summary,
        load_paper_real_diff_summary,
        load_weekly_accuracy_snapshots,
    )

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
            f"• `{row['market']}/{row['symbol']}` "
            f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}{flag}"
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

    if llm_review:
        lines.append("\n**🧠 Claude 講評**")
        lines.append(llm_review)

    return send_webhook_text_chunked("\n".join(lines))


def send_weekly_training_completion(models: list) -> bool:
    """
    週次モデル学習完了通知を Discord Webhook に送信する。

    Args:
        models: 学習したモデル名のリスト

    Returns:
        成功時 True、失敗時 False
    """
    models_str = "\n".join(f"• {m}" for m in models) or "なし"
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "📦 学習モデル数", "value": f"{len(models):,} 件", "inline": True},
        {"name": "🏷 学習済みモデル", "value": models_str, "inline": False},
    ]
    return send_status_fields(WEEKLY_TRAINING_COMPLETION, fields)


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
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "✅ 成功", "value": f"{success_count:,} 銘柄", "inline": True},
        {"name": "❌ 失敗", "value": f"{failed_count:,} 銘柄", "inline": True},
        {"name": "📊 合計", "value": f"{total_count:,} 銘柄", "inline": True},
        {
            "name": "📄 前回比較",
            "value": previous_path if previous_path else "なし（初回実行）",
            "inline": False,
        },
    ]
    ok = send_status_fields(spec, fields)

    if markdown_path:
        try:
            if not send_text_file_chunked(markdown_path, preserve_lines=False):
                ok = False
        except Exception as e:
            logger.error("Walk-Forwardレポートテキスト送信失敗: %s", e)
            ok = False

    return ok


def send_monthly_report_notification(
    target_month: str,
    net_return: Optional[float],
    max_drawdown: Optional[float],
    sharpe_ratio: Optional[float],
    hit_rate: Optional[float],
    avg_slippage: Optional[float],
    symbol_count: Optional[int],
    report_path: Optional[str] = None,
) -> bool:
    """
    月次レポート生成完了通知を Discord に送信する（R-203）。

    Args:
        target_month: 対象年月 "YYYY-MM"
        net_return:   Net Return（WF fold 平均）
        max_drawdown: Max Drawdown（WF fold 平均）
        sharpe_ratio: Sharpe Ratio（WF fold 平均）
        hit_rate:     方向一致率（直近30日）
        avg_slippage: 平均スリッページ
        symbol_count: 集計銘柄数
        report_path:  保存先 Markdown ファイルパス（任意）

    Returns:
        送信成功時 True、失敗時 False
    """

    def _fmt(val: Optional[float], pct: bool = False, decimals: int = 2) -> str:
        if val is None:
            return "N/A"
        if pct:
            return f"{val * 100:.{decimals}f}%"
        return f"{val:.{decimals}f}"

    fields: list[dict] = [
        {"name": "📅 対象月", "value": target_month, "inline": True},
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "💹 Net Return", "value": _fmt(net_return, pct=True), "inline": True},
        {"name": "📉 Max Drawdown", "value": _fmt(max_drawdown, pct=True), "inline": True},
        {"name": "📈 Sharpe Ratio", "value": _fmt(sharpe_ratio), "inline": True},
        {"name": "🎯 Hit Rate", "value": _fmt(hit_rate, pct=True), "inline": True},
        {"name": "💧 Avg Slippage", "value": _fmt(avg_slippage, pct=True), "inline": True},
        {
            "name": "🏷 集計銘柄数",
            "value": f"{symbol_count:,}" if symbol_count is not None else "N/A",
            "inline": True,
        },
    ]
    if report_path:
        fields.append({"name": "📁 保存先", "value": report_path, "inline": False})
    return send_status_fields(MONTHLY_REPORT_COMPLETION, fields)
