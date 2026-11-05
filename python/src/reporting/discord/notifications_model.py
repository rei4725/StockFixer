"""
Discord モデル/学習系通知

昇格評価・最適化完了・戦略ファクトリー・SHAP 寄与・特徴量提案・シャドー評価の
通知関数群。バックテストによる仮の入れ替えの通知もここに置く。discord_utils.py の段階的分割（#497 第4弾）で抽出。
送信基盤は webhook_sender に依存する。
"""

import logging

from src.reporting.discord.discord_notification_specs import (
    BACKTEST_PROMOTION_FAILED,
    BACKTEST_PROMOTION_LOST,
    BACKTEST_PROMOTION_NOT_COMPARABLE,
    BACKTEST_PROMOTION_ROLLBACK,
    BACKTEST_PROMOTION_SKIPPED,
    BACKTEST_PROMOTION_SWAP,
    BACKTEST_PROMOTION_WATCH_KEPT,
    BACKTEST_PROMOTION_WON,
    FACTORY_ERROR,
    OPTIMIZATION_ERROR,
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
from src.utils.db.symbol_master import symbol_name_suffix
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
        "prediction_spread": "Prediction Spread",
    }
    for key, c in criteria.items():
        label = criteria_labels.get(key, key)
        mark = "OK" if c.get("passed") else "NG"
        lines.append(f"[{mark}] {label}")
    lines.append("```")
    return send_webhook_text_chunked("\n".join(lines))


def send_optimization_completion(success: int, failed: int, skipped: int = 0) -> bool:
    """
    週次バックテスト最適化完了通知を Discord Webhook に送信する。

    Args:
        success: 最適化成功銘柄数
        failed: 最適化失敗銘柄数
        skipped: スキップした銘柄数（成功には数えない。#973）

    Returns:
        成功時 True、失敗時 False
    """
    spec = get_optimization_spec(failed=failed)
    status_icon = "⚠️" if failed > 0 else "✅"
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "✅ 成功", "value": f"{success:,} 銘柄", "inline": True},
        {"name": f"{status_icon} 失敗", "value": f"{failed:,} 銘柄", "inline": True},
    ]
    if skipped > 0:
        fields.append({"name": "⏭️ スキップ", "value": f"{skipped:,} 銘柄", "inline": True})
    fields.append(
        {
            "name": "💾 保存先",
            "value": "最適パラメータを `config/optimal_params.json` に保存しました",
            "inline": False,
        }
    )
    return send_status_fields(spec, fields)


def send_optimization_failure(error: str) -> bool:
    """
    週次バックテスト最適化が全体で失敗したことを Discord Webhook に送信する（#973）。

    最適パラメータは更新されていないため、保存先の案内は出さない。

    Args:
        error: 失敗の内容

    Returns:
        成功時 True、失敗時 False
    """
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "❌ エラー", "value": error[:1000] or "不明", "inline": False},
        {
            "name": "💾 最適パラメータ",
            "value": "更新されていません（前回の値のまま）",
            "inline": False,
        },
    ]
    return send_status_fields(OPTIMIZATION_ERROR, fields)


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


def send_factory_failure(market: str, error: str) -> bool:
    """
    戦略ファクトリー夜間バッチが全体で失敗したことを Discord Webhook に送信する（#1284）。

    その夜の市場の評価は行われていないため、やり直しの案内を添える。

    Args:
        market: 対象マーケット
        error: 失敗の内容

    Returns:
        成功時 True、失敗時 False
    """
    fields: list[dict] = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "🌐 市場", "value": market, "inline": True},
        {"name": "❌ エラー", "value": error[:1000] or "不明", "inline": False},
        {
            "name": "🧪 評価",
            "value": "この回の評価は行われていません（run_strategy_factory.py でやり直せます）",
            "inline": False,
        },
    ]
    return send_status_fields(FACTORY_ERROR, fields)


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


def send_strategy_promotion_detected(
    pr_number: int, rule_or_feature_id: str, pre_promotion_baseline: float
) -> bool:
    """
    戦略ファクトリー由来 PR のマージ検出（＝昇格）を Discord Webhook に通知する。

    Args:
        pr_number: マージされた PR 番号
        rule_or_feature_id: 対象仮説/アイデアの識別子（factory hash）
        pre_promotion_baseline: 昇格直前のチャンピオン Sharpe

    Returns:
        成功時 True、失敗時 False
    """
    lines = [
        "**🏭⬆️ 戦略ファクトリー: 新規昇格を検出**",
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
        f"PR: #{pr_number}",
        f"識別子: `{rule_or_feature_id}`",
        f"昇格直前ベースライン Sharpe: {pre_promotion_baseline:.3f}",
    ]
    return send_webhook_text_chunked("\n".join(lines))


def send_allocation_rebalance_report(
    action: str,
    tqqq_price: float,
    shy_price: float,
    tqqq_qty_before: float,
    shy_qty_before: float,
    cash_before: float,
    tqqq_qty_after: float,
    shy_qty_after: float,
    cash_after: float,
) -> bool:
    """配分戦略(TQQQ/短期債)の初期建玉作成・リバランス結果を Discord に通知する。

    Args:
        action: "initial"（初期建玉作成）または "rebalance"（リバランス）
    """
    from src.reporting.discord.discord_notification_specs import ALLOCATION_REBALANCE_COMPLETION

    action_label = "初期建玉作成" if action == "initial" else "リバランス"
    fields = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "📌 種別", "value": action_label, "inline": True},
        {
            "name": "TQQQ",
            "value": f"{tqqq_qty_before:.4f}株 → {tqqq_qty_after:.4f}株 @ ${tqqq_price:,.2f}",
            "inline": False,
        },
        {
            "name": "SHY",
            "value": f"{shy_qty_before:.4f}株 → {shy_qty_after:.4f}株 @ ${shy_price:,.2f}",
            "inline": False,
        },
        {
            "name": "現金",
            "value": f"${cash_before:,.2f} → ${cash_after:,.2f}",
            "inline": False,
        },
    ]
    return send_status_fields(ALLOCATION_REBALANCE_COMPLETION, fields)


_BACKTEST_PROMOTION_SPECS = {
    "won": BACKTEST_PROMOTION_WON,
    "lost": BACKTEST_PROMOTION_LOST,
    "not_comparable": BACKTEST_PROMOTION_NOT_COMPARABLE,
    "failed": BACKTEST_PROMOTION_FAILED,
}
_FIELD_LIMIT = 1000  # Discord の field の値は 1024 字まで


def _bp_num(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"


def _bp_field(name: str, value: str, inline: bool = False) -> dict:
    return {"name": name, "value": (value or "-")[:_FIELD_LIMIT], "inline": inline}


def _bp_metrics(m: dict | None) -> str:
    m = m or {}
    return (
        f"基準差 {_bp_num(m.get('hit_rate_minus_majority'))} / MAE {_bp_num(m.get('mae'))}"
        f" / 順位相関 {_bp_num(m.get('rank_corr'))} / 上げ率 {_bp_num(m.get('predicted_up_rate'))}"
        f" (n={m.get('n_baseline', 0)})"
    )


def send_backtest_promotion_result(result: dict) -> bool:
    """バックテストの評価の結果（勝ち／負け／比べが成り立たない／失敗／見送り）を送る。

    Args:
        result: run_backtest_promotion() の戻り値
    """
    outcome = str(result.get("outcome") or "")
    spec = _BACKTEST_PROMOTION_SPECS.get(outcome, BACKTEST_PROMOTION_SKIPPED)
    fields = [
        _bp_field("🕐 時刻", format_jst(fmt=DISCORD_DATETIME_FORMAT), True),
        _bp_field("🔖 版", f"`{result.get('version') or '-'}`", True),
    ]
    if result.get("dry_run"):
        fields.append(_bp_field("🧪 dry-run", "入れ替えない回", True))
    evaluation = result.get("evaluation") or {}
    dates = result.get("dates") or []
    if dates:
        n_symbols = evaluation.get("n_symbols", 0)
        fields.append(
            _bp_field("📅 期間", f"{dates[0]}〜{dates[-1]}（{len(dates)} 日・{n_symbols} 銘柄）")
        )
    if evaluation:
        fields.append(_bp_field("🧪 候補", _bp_metrics(evaluation.get("candidate"))))
        fields.append(_bp_field("🏭 Production", _bp_metrics(evaluation.get("production"))))
        daily = evaluation.get("daily") or []
        wins = sum(1 for d in daily if isinstance(d.get("diff"), (int, float)) and d["diff"] > 0)
        path = evaluation.get("path_check") or {}
        fields.append(_bp_field("📈 日ごとの勝ち越し", f"{wins}/{len(daily)} 日", True))
        fields.append(
            _bp_field(
                "🛣️ 道の一致",
                f"{'OK' if path.get('passed') else 'NG'}"
                f"（{path.get('compared', 0)} 銘柄・最大差 {_bp_num(path.get('max_abs_diff'))}）",
                True,
            )
        )
    reasons = (evaluation.get("decision") or {}).get("reasons") or []
    detail = "\n".join(reasons) or result.get("reason") or ""
    if result.get("error"):
        detail = f"{result.get('reason') or ''}\n{result['error']}"
    if detail:
        fields.append(_bp_field("📝 理由", detail))
    if result.get("restore_error"):
        backups = "\n".join(result.get("backups") or []) or "-"
        fields.append(_bp_field("⚠️ production が候補のままの恐れ（戻す写し）", backups))
    watch = result.get("watch") or {}
    if watch.get("rolled_back"):
        restored = "\n".join(watch.get("backups") or []) or "-"
        fields.append(_bp_field("⏪ production は旧モデルに戻した（戻した写し）", restored))
    fields.append(_bp_field("📄 JSON", result.get("result_path") or "-"))
    return send_status_fields(spec, fields)


def send_backtest_promotion_swap(result: dict) -> bool:
    """バックテストで勝った候補を本番の名前で置いたことを送る。"""
    fields = [
        _bp_field("🕐 時刻", format_jst(fmt=DISCORD_DATETIME_FORMAT), True),
        _bp_field("🔖 版", f"`{result.get('version') or '-'}`", True),
        _bp_field("💾 写し（戻す先）", "\n".join(result.get("backups") or []) or "-"),
    ]
    description = (
        "次の予想の回から新しい production を使う。採点済みの日が5日たまったら見張りで判定する。"
    )
    return send_status_fields(BACKTEST_PROMOTION_SWAP, fields, description=description)


def send_backtest_promotion_rollback(result: dict) -> bool:
    """入れ替えの後の見張りの判定（戻した／戻さない）を送る。"""
    watch = result.get("watch") or {}
    spec = (
        BACKTEST_PROMOTION_ROLLBACK if watch.get("rolled_back") else BACKTEST_PROMOTION_WATCH_KEPT
    )
    triggered = "\n".join(watch.get("triggered") or []) or "当たらなかった"
    if watch.get("triggered") and not watch.get("rolled_back"):
        triggered += "\n→ 戻す先のバックテストも良くないため戻さない"
    fields = [
        _bp_field("🕐 時刻", format_jst(fmt=DISCORD_DATETIME_FORMAT), True),
        _bp_field("🔖 入れ替えた版", f"`{watch.get('version') or '-'}`", True),
        _bp_field("📊 実測（最初の5日）", _bp_metrics(watch.get("live"))),
        _bp_field("🚨 見張りの条件", triggered),
    ]
    if watch.get("rolled_back"):
        restored = "\n".join(watch.get("backups") or []) or "-"
        fields.append(_bp_field("💾 戻した写し", restored))
    if watch.get("error"):
        fields.append(_bp_field("⚠️ 状態を保存できなかった（見張りが残る）", watch["error"]))
    fields.append(_bp_field("📄 JSON", result.get("result_path") or "-"))
    return send_status_fields(spec, fields)
