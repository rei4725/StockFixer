"""流動性から成行・指値と寄付・引けを決め、指値が約定するかを判定する純粋なルール関数。

本番の発注（trading の _choose_order_params）とバックテスト（backtest の Backtester）の
両方から呼ぶ。しきい値と余裕の幅は呼び出し元が config.settings の同じ定数を渡す（#1418）。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.tick_size import round_limit_price, tick_size


@dataclass(frozen=True)
class LimitOrderPlan:
    """注文の種別の判定。is_limit が False なら成行（limit_price は 0）。

    session: "open"（寄付）または "close"（引け）
    """

    is_limit: bool
    limit_price: float
    reasons: tuple[str, ...]
    session: str


def plan_limit_order(
    market: str,
    is_buy_side: bool,
    current_price: float,
    avg_volume: float | None,
    spread_proxy: float | None,
    volume_threshold: float,
    spread_threshold: float,
    price_buffer: float,
) -> LimitOrderPlan:
    """平均出来高が少ないか値幅が広ければ指値にする（R-103 拡張: R-405）。"""
    reasons: list[str] = []
    if avg_volume is not None and avg_volume < volume_threshold:
        reasons.append(f"low_volume={avg_volume:.0f}")
    if spread_proxy is not None and spread_proxy > spread_threshold:
        reasons.append(f"wide_range={spread_proxy:.3%}")

    if not reasons:
        return LimitOrderPlan(False, 0.0, (), "open")

    # 低流動性 → 引け指値（引け時点で流動性が集まりやすく価格優位性が高い）
    # 広スプレッドのみ → 寄付指値（寄付時の価格優位性を活用）
    session = "close" if any("low_volume" in r for r in reasons) else "open"
    # 余裕の幅は少なくとも現在値の帯の単位1つ分にとり、丸めの後も買いは現在値以上・
    # 売りは現在値以下に残す（#1403）
    buffer = max(current_price * price_buffer, tick_size(market, current_price))
    # 買い（BUY・SHORT_COVER）は現在値より上、売り（SELL・SHORT）は下に指値を置く
    limit_price = current_price + buffer if is_buy_side else max(0.0, current_price - buffer)
    # 呼値の単位に丸める。買いは切り下げ、売りは切り上げて、余裕の幅より現在値から離さない（#1396）
    limit_price = round_limit_price(market, limit_price, is_buy_side)
    return LimitOrderPlan(True, limit_price, tuple(reasons), session)


def limit_order_fills(
    plan: LimitOrderPlan, is_buy_side: bool, high: float, low: float, close: float
) -> bool:
    """約定の日のバーが指値に届いたか。引けの指値は終値、寄付の指値は高値・安値で判定する。"""
    if not plan.is_limit:
        return True
    if plan.session == "close":
        high = low = close
    return low <= plan.limit_price if is_buy_side else high >= plan.limit_price
