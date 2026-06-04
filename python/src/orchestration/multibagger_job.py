"""月次ライブ・スクリーン + Discord アラート（#434）

月次で multibagger スクリーン（#433: トレンド + 質ゲート）を実行し、

1. 新規候補（上位N・スコア・主要指標）
2. 既存保有ポジション（#430 保有/撤退エンジン）の HOLD / SCALE_OUT / EXIT シグナル

を Discord に通知する。**売買は人間が最終判断（半自動）** のため、本ジョブは
発注を行わず、保有状態の更新と通知のみを担う。

レイヤー: orchestration（最上位）から screening / reporting / utils（下位）を参照する
のは合法（上位 → 下位）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.screening.hold_engine import simulate_position
from src.screening.positions import close_position, list_positions, update_position
from src.screening.quality_gate import apply_quality_gate
from src.screening.trend_screener import screen_trend_candidates
from src.screening.types import MultibaggerCandidate, Position, PositionEvent
from src.utils.db.stock_features import load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 撤退を意味する PositionEvent.reason（end_of_data は「未撤退で系列末尾」を表す）
_EXIT_REASONS = {"ma_break", "trail_stop", "thesis_break"}


def run_monthly_multibagger_screen(market: str = "us", top_n: int = 10) -> dict:
    """月次 multibagger スクリーン + 保有判定を実行し Discord 通知する。

    Args:
        market: 対象マーケット（既定 "us"）。
        top_n: 新規候補の通知件数。

    Returns:
        通知ペイロード（新規候補・保有アクション・本文）を含む dict。
    """
    logger.info("=== 月次 multibagger スクリーン開始 (market=%s) ===", market)

    new_candidates = _screen_new_candidates(market, top_n)
    position_actions = _evaluate_positions(market)

    message = _format_message(market, new_candidates, position_actions)
    _notify(message)

    logger.info(
        "=== 月次 multibagger スクリーン完了: 新規候補=%d 件, 保有判定=%d 件 ===",
        len(new_candidates),
        len(position_actions),
    )
    return {
        "market": market,
        "new_candidates": new_candidates,
        "position_actions": position_actions,
        "message": message,
    }


def _screen_new_candidates(market: str, top_n: int) -> list[MultibaggerCandidate]:
    """#433 の最終スクリーン（トレンド + 質ゲート）で新規候補を取得する。"""
    try:
        trend = screen_trend_candidates(market=market)
        final = apply_quality_gate(trend)
        return final[:top_n]
    except Exception as e:
        logger.error("新規候補スクリーン失敗: %s", e, exc_info=True)
        return []


def _evaluate_positions(market: str) -> list[dict]:
    """open ポジションを #430 保有エンジンに通し、アクションを判定・永続化する。"""
    actions: list[dict] = []
    for pos in list_positions(status="open"):
        if pos.market != market:
            continue
        try:
            action = _evaluate_one_position(pos)
        except Exception as e:
            logger.error("保有判定失敗 (%s/%s): %s", pos.market, pos.symbol, e, exc_info=True)
            continue
        if action is not None:
            actions.append(action)
    return actions


def _evaluate_one_position(pos: Position) -> Optional[dict]:
    """1 ポジションを最新価格まで保有シミュレートし、アクションを確定・永続化する。"""
    prices = load_stock_features(pos.market, pos.symbol)
    if prices is None or "date" not in prices.columns or "Close" not in prices.columns:
        logger.warning("価格データなし（判定スキップ）: %s/%s", pos.market, pos.symbol)
        return None

    events = simulate_position(prices, pos.entry_date)
    if not events:
        logger.warning("保有イベントなし（判定スキップ）: %s/%s", pos.market, pos.symbol)
        return None

    action, reason, held_fraction, multiple, price = _interpret_events(events)

    if action == "EXIT":
        close_position(pos.market, pos.symbol, close_price=price, reason=reason)
    else:
        update_position(
            pos.market,
            pos.symbol,
            held_fraction=held_fraction,
            last_action=action.lower(),
            last_reason=reason,
            last_multiple=multiple,
        )

    return {
        "market": pos.market,
        "symbol": pos.symbol,
        "action": action,
        "reason": reason,
        "held_fraction": held_fraction,
        "multiple": multiple,
        "price": price,
        "entry_price": pos.entry_price,
    }


def _interpret_events(events: list[PositionEvent]) -> tuple[str, str, float, float, float]:
    """保有イベント列から月次アクションを導出する。

    Returns:
        (action, reason, held_fraction, multiple, price)
        action は "EXIT" | "SCALE_OUT" | "HOLD"。
    """
    last = events[-1]
    if last.action == "exit" and last.reason in _EXIT_REASONS:
        return ("EXIT", last.reason, 0.0, last.multiple, last.price)

    scale_outs = [e for e in events if e.action == "scale_out"]
    if scale_outs:
        latest = scale_outs[-1]
        return ("SCALE_OUT", latest.reason, latest.held_fraction, last.multiple, last.price)

    return ("HOLD", "hold", 1.0, last.multiple, last.price)


def _format_message(
    market: str,
    candidates: list[MultibaggerCandidate],
    actions: list[dict],
) -> str:
    """新規候補 + 保有アクションを Discord 向けテキスト（コードフェンス）に整形する。"""
    lines: list[str] = [f"📈 月次 multibagger スクリーン ({market.upper()})", ""]

    lines.append("【新規候補】")
    if candidates:
        lines.append("```")
        lines.append(f"{'symbol':<8}{'score':>8}{'close':>10}{'CAGR':>8}{'ROE':>8}")
        for c in candidates:
            lines.append(
                f"{c.symbol:<8}{c.multibagger_score:>8.3f}{c.close:>10.2f}"
                f"{_pct(c.revenue_cagr_3y):>8}{_pct(c.roe):>8}"
            )
        lines.append("```")
    else:
        lines.append("該当なし")

    lines.append("")
    lines.append("【保有ポジション】")
    if actions:
        lines.append("```")
        lines.append(f"{'symbol':<8}{'action':<10}{'mult':>7}{'reason':>14}")
        for a in actions:
            lines.append(
                f"{a['symbol']:<8}{a['action']:<10}{a['multiple']:>6.2f}x{a['reason']:>14}"
            )
        lines.append("```")
    else:
        lines.append("保有なし")

    return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    """比率を百分率表記（None/NaN は "-"）に整形する。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value * 100:.1f}%"


def _notify(message: str) -> None:
    """Discord に通知する。送信失敗してもジョブ全体は落とさない（ログのみ）。"""
    try:
        from src.reporting.discord.discord_utils import send_webhook_text_chunked

        send_webhook_text_chunked(message)
    except Exception as e:
        logger.error("月次 multibagger 通知失敗: %s", e, exc_info=True)
