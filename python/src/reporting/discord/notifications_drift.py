"""
Discord ドリフト / 予測精度通知

予測外れ分析・モデルドリフト警告・精度サマリー・再学習トリガー・Hit Rate ドリフト・
相関リスクの通知関数群。discord_utils.py の段階的分割（#497 第3弾）で抽出。
送信基盤は webhook_sender に依存する。
"""

import logging

from src.reporting.discord.discord_notification_specs import HIT_RATE_DRIFT_ALERT
from src.reporting.discord.discord_text import DISCORD_DATE_FORMAT, DISCORD_MINUTE_FORMAT
from src.reporting.discord.webhook_sender import (
    send_status_fields,
    send_webhook_notification,
    send_webhook_text,
)
from src.utils.japan_time import format_jst

logger = logging.getLogger(__name__)


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
