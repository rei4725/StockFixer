"""
Discord 特徴量系通知

学習の SHAP 寄与と、ドリフト銘柄の特徴量除外提案の通知関数群。
notifications_model.py が 600 行に近いため、そこから動かした（#1932）。
送信基盤は webhook_sender に依存する。
"""

import logging

from src.reporting.discord.discord_text import DISCORD_DATE_FORMAT
from src.reporting.discord.webhook_sender import (
    send_webhook_notification,
    send_webhook_text_chunked,
)
from src.utils.db.symbol_master import symbol_name_suffix
from src.utils.japan_time import format_jst

logger = logging.getLogger(__name__)


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
            lines.append(
                f"• `{market}/{symbol}`{symbol_name_suffix(market, symbol)} — 除外候補なし"
            )
            continue
        lines.append(
            f"• `{market}/{symbol}`"
            f"{symbol_name_suffix(market, symbol)} 除外候補 {len(candidates_df)} 件:"
        )
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
