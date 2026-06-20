"""
Claude 週次レビュー講評生成モジュール

週次の予測精度（ドリフト）と paper/real 乖離の指標を Claude に渡し、
「今週の総評・懸念点・推奨アクション」を日本語で生成する。

バックエンドは LLM_BACKEND で選択する（sdk=API 課金 / cli=サブスク認証）。
LLM_REVIEW_ENABLED=False（既定）または生成失敗時は None を返し、呼び出し元は
講評なしで週次レポートを送信する（graceful degradation）。
実弾の発注判断には一切関与しない（読み取り専用の講評のみ）。
"""

from typing import Optional

import pandas as pd

from config.settings import LLM_REVIEW_ENABLED, LLM_REVIEW_MAX_TOKENS, LLM_REVIEW_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "あなたは経験豊富な株式トレーディングシステムのアナリストです。"
    "週次の予測精度とpaper/real約定乖離の指標が与えられます。"
    "投資判断ではなく、システム運用者向けの観察コメントを述べてください。"
    "次の3点を簡潔な日本語の箇条書きで答えてください: "
    "(1)今週の総評 (2)懸念点 (3)推奨アクション。"
    "与えられた数値のみを用い、推測の数値を作らないこと。"
    "前置きや締めの挨拶は不要です。"
)

# Claude に渡す要注意銘柄の最大件数（トークン節約）
_MAX_SYMBOLS = 10


def _build_metrics_digest(
    accuracy_df: pd.DataFrame,
    diff_summary: Optional[dict],
    horizon: int,
) -> str:
    """指標を Claude 用のコンパクトなテキストに整形する。"""
    lines = [f"対象ホライズン: {horizon}日"]

    mean_acc = float(accuracy_df["direction_accuracy"].mean())
    lines.append(f"全体平均方向正解率: {mean_acc:.1%}（{len(accuracy_df)}銘柄）")

    worst = accuracy_df.sort_values("direction_accuracy", ascending=True).head(_MAX_SYMBOLS)
    lines.append("方向正解率の低い銘柄（要注意・低い順）:")
    for _, row in worst.iterrows():
        acc = float(row.get("direction_accuracy", 0.0))
        err = float(row.get("mean_abs_error", 0.0))
        n = int(row.get("n_samples", 0))
        lines.append(
            f"- {row['market']}/{row['symbol']}: 正解率{acc:.1%}, 平均誤差{err:.4f}, N={n}"
        )

    if diff_summary and diff_summary.get("tracked_count", 0) > 0:
        lines.append("paper/real 乖離（直近7日）:")
        lines.append(
            f"- tracked={diff_summary['tracked_count']}件, "
            f"comparable={diff_summary['comparable_count']}件, "
            f"平均paper slippage={diff_summary['avg_paper_slippage']:.3%}, "
            f"平均real slippage={diff_summary['avg_real_slippage']:.3%}, "
            f"平均乖離率={diff_summary['avg_abs_diff_ratio']:.3%}"
        )

    return "\n".join(lines)


def generate_weekly_review(
    accuracy_df: Optional[pd.DataFrame],
    diff_summary: Optional[dict] = None,
    horizon: int = 1,
) -> Optional[str]:
    """週次指標から Claude の講評を生成する。

    Args:
        accuracy_df: load_drift_summary() の戻り値 DataFrame
        diff_summary: load_paper_real_diff_summary() の戻り値 dict（任意）
        horizon: 対象ホライズン

    Returns:
        講評テキスト。無効・データなし・API 失敗時は None。
    """
    if not LLM_REVIEW_ENABLED:
        return None
    if accuracy_df is None or accuracy_df.empty:
        logger.info("[llm_review] 精度データなしのため講評をスキップ")
        return None

    digest = _build_metrics_digest(accuracy_df, diff_summary, horizon)

    try:
        from src.infrastructure.llm.factory import get_text_review_port  # noqa: PLC0415

        port = get_text_review_port()
        review = port.complete(
            system=_SYSTEM_PROMPT,
            user=digest,
            model=LLM_REVIEW_MODEL,
            max_tokens=LLM_REVIEW_MAX_TOKENS,
        ).strip()
        if review:
            logger.info("[llm_review] 講評生成完了（%d文字）", len(review))
            return review
        logger.warning("[llm_review] 空の講評が返却された")
        return None
    except Exception:
        logger.error("[llm_review] 講評生成でエラー", exc_info=True)
        return None
