"""
Discord モデル/学習系通知

昇格評価・最適化完了・戦略ファクトリー・SHAP 寄与・特徴量提案・シャドー評価の
通知関数群。discord_utils.py の段階的分割（#497 第4弾）で抽出。
送信基盤は webhook_sender に依存する。
"""

import logging

from src.reporting.discord.discord_notification_specs import (
    SHADOW_EVALUATION_CHALLENGER_WINS,
    SHADOW_EVALUATION_NO_WINNER,
    NotificationSpec,
    get_optimization_spec,
)
from src.reporting.discord.discord_text import DISCORD_DATE_FORMAT, DISCORD_DATETIME_FORMAT
from src.reporting.discord.webhook_sender import (
    send_status_fields,
    send_webhook_notification,
    send_webhook_text_chunked,
)
from src.utils.japan_time import format_jst

logger = logging.getLogger(__name__)


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
