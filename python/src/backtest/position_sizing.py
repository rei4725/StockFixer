"""ポジションサイジングロジック。

Backtester から抽出（File Size Guard 対応・#374）。ロジック自体の変更はなし。
"""

from __future__ import annotations

from typing import Any, Optional


def calc_position_details(
    cash: float,
    price: float,
    position_sizing: str,
    position_fraction: float,
    atr_risk_pct: float,
    atr_multiplier: float,
    atr_min_fraction: float,
    atr_max_fraction: float,
    fee_rate: float,
    slippage: float,
    pred_value: Optional[float] = None,
    atr_value: Optional[float] = None,
) -> dict[str, Any]:
    """
    ポジションサイジングに基づいて購入数量を算出する。

    Args:
        cash: 利用可能な現金
        price: 現在の株価
        position_sizing: サイジング種別 ("full", "fixed", "confidence", "atr")
        position_fraction: 固定ポジション比率（fixed モード用）
        atr_risk_pct: ATRモード: 1トレードあたりのリスク割合
        atr_multiplier: ATRモード: ストップ幅とするATRの倍数
        atr_min_fraction: ATRモード: 建玉下限比率
        atr_max_fraction: ATRモード: 建玉上限比率
        fee_rate: 取引手数料率
        slippage: 片道スリッページ率
        pred_value: 予測値（confidence モードで使用）
        atr_value: ATR値（atr モードで使用）

    Returns:
        購入数量と補助情報
    """
    unit_cost = price * (1 + fee_rate + slippage)
    fallback_used: bool = False
    if unit_cost <= 0 or cash <= 0:
        return {
            "qty": 0,
            "position_fraction": 0.0,
            "sizing_mode": position_sizing,
            "atr_value": atr_value,
            "atr_stop_distance": None,
            "atr_risk_amount": None,
            "atr_fallback_used": False,
        }

    if position_sizing == "fixed":
        available = cash * position_fraction
    elif position_sizing == "confidence" and pred_value is not None:
        # 予測確信度（|pred|）に比例して資金を配分
        # |pred| = 0.01 (1%) → fraction ~0.5, |pred| = 0.02+ → fraction ~1.0
        confidence = min(abs(pred_value) * 50, 1.0)
        min_frac = 0.2
        max_frac = 1.0
        fraction = min_frac + confidence * (max_frac - min_frac)
        available = cash * fraction
    elif position_sizing == "atr" and atr_value is not None and atr_value > 0:
        # ATR連動ポジションサイジング
        # リスク額 = equity × atr_risk_pct
        # ストップ幅 = ATR × atr_multiplier（1ATR分の価格変動をリスク上限とみなす）
        # 購入株数 = リスク額 / (ATR × atr_multiplier)
        risk_amount = cash * atr_risk_pct
        stop_distance = atr_value * atr_multiplier
        qty_by_risk = risk_amount / stop_distance
        min_fraction = min(atr_min_fraction, atr_max_fraction)
        max_fraction = max(atr_min_fraction, atr_max_fraction)
        min_qty = int((cash * min_fraction) // unit_cost)
        max_qty_by_fraction = int((cash * max_fraction) // unit_cost)
        max_affordable_qty = int(cash // unit_cost)
        effective_max_qty = min(max_affordable_qty, max_qty_by_fraction)
        if effective_max_qty <= 0:
            qty = 0
        else:
            qty = int(min(qty_by_risk, effective_max_qty))
            if min_qty > effective_max_qty:
                min_qty = effective_max_qty
            qty = max(min_qty, qty)
        return build_position_details(
            qty=qty,
            cash=cash,
            unit_cost=unit_cost,
            sizing_mode="atr",
            atr_value=atr_value,
            atr_stop_distance=round(stop_distance, 6),
            atr_risk_amount=round(risk_amount, 6),
        )
    else:
        # "full" モード（デフォルト: 全額投入）
        available = cash
        fallback_used = position_sizing == "atr"

    qty = int(available // unit_cost)
    return build_position_details(
        qty=qty,
        cash=cash,
        unit_cost=unit_cost,
        sizing_mode="full" if position_sizing == "atr" else position_sizing,
        atr_value=atr_value,
        atr_fallback_used=fallback_used if position_sizing == "atr" else False,
    )


def build_position_details(
    qty: int,
    cash: float,
    unit_cost: float,
    sizing_mode: str,
    atr_value: Optional[float],
    atr_stop_distance: Optional[float] = None,
    atr_risk_amount: Optional[float] = None,
    atr_fallback_used: bool = False,
) -> dict[str, Any]:
    position_value = max(0, qty) * unit_cost
    position_fraction = position_value / cash if cash > 0 else 0.0
    return {
        "qty": max(0, qty),
        "position_fraction": round(position_fraction, 6),
        "sizing_mode": sizing_mode,
        "atr_value": atr_value,
        "atr_stop_distance": atr_stop_distance,
        "atr_risk_amount": atr_risk_amount,
        "atr_fallback_used": atr_fallback_used,
    }
