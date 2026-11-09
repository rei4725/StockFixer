"""発注株数が 0 になった理由の判定（#1734）。

RiskManager.calc_position_size が 0 を返すとき、呼び手が理由を分けられるように
理由の文を作る。株数の計算そのものはしない（計算は risk_manager.py に残す）。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

REASON_PRICE = "price<=0"
REASON_KELLY = "kelly<=0"
REASON_UNDER_LOT = "単元未満"
REASON_SCALED = "縮小で単元割れ"


def zero_size_reason(
    kelly: float,
    invest_amount: float,
    price: float,
    base_qty: int,
    lot: int,
    scales: dict[str, float],
) -> str:
    """株数が 0 になった理由の文を返す。

    Args:
        kelly: クリップ後の Kelly 比率
        invest_amount: レジーム補正と1銘柄上限を掛けた後の投資額
        price: 発注価格
        base_qty: 縮小を掛ける前の株数（投資額 / 価格）
        lot: 最低売買単位
        scales: 縮小の名前と係数（confidence / dd / vix）。1 未満のものを縮小とみなす
    """
    if price <= 0:
        return REASON_PRICE
    if kelly <= 0:
        return f"{REASON_KELLY} kelly={kelly:.3f}"
    if base_qty < lot:
        return (
            f"{REASON_UNDER_LOT} 投資額={invest_amount:.0f} "
            f"1単元の代金={price * lot:.0f} lot={lot}"
        )
    shrunk = " ".join(f"{name}={scale:.2f}" for name, scale in scales.items() if scale < 1.0)
    return f"{REASON_SCALED} base_qty={base_qty} lot={lot} {shrunk}".rstrip()


def note_zero_size(symbol: str, reason: str) -> str:
    """理由を INFO で出して、そのまま返す。"""
    logger.info(f"[risk] {symbol}: 発注株数 0 の理由 {reason}")
    return reason
