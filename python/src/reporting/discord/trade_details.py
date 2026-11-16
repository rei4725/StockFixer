"""約定・発注の知らせの1件ごとの明細（#1876）

銘柄（コードと名前）・売買の向き・単価・数量を1行ずつ組み立てる。embed の1欄の文字数の上限を
超えないよう、入らない分は「ほか N 件」にまとめる。明細が作れないときは None を返し、
知らせは件数だけの形で出す。
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from src.utils.db.symbol_master import symbol_label

logger = logging.getLogger(__name__)

TRADE_DETAIL_FIELD_NAME = "📋 明細"
# Discord の embed の1欄の value の上限（文字数）
FIELD_VALUE_LIMIT = 1024
DEFAULT_MAX_LINES = 15

# trading の OrderSide の値（BUY=1, SELL=2, SHORT=3, SHORT_COVER=4）。reporting は trading を読まない
_SIDE_LABELS = {1: "買い", 2: "売り", 3: "空売り", 4: "買い戻し"}


@dataclass(frozen=True)
class TradeLine:
    """明細の1件。price は約定の単価か指値。成行の注文は is_market を立てる"""

    symbol: str
    side: int
    qty: int
    price: float
    is_market: bool = False


def _price_text(line: TradeLine) -> str:
    if line.is_market or line.price <= 0:
        return "成行"
    price = float(line.price)
    return f"{price:,.0f}円" if price.is_integer() else f"{price:,.1f}円"


def format_trade_line(line: TradeLine, market: str = "jp") -> str:
    """明細の1行（例: • 7203 トヨタ自動車 買い 100株 @ 2,850円）"""
    side = _SIDE_LABELS.get(int(line.side), str(line.side))
    return f"• {symbol_label(market, line.symbol)} {side} {int(line.qty):,}株 @ {_price_text(line)}"


def _rest_text(rest: int) -> str:
    return f"…ほか {rest} 件"


def build_trade_detail_value(
    lines: Sequence[TradeLine],
    market: str = "jp",
    max_lines: int = DEFAULT_MAX_LINES,
    limit: int = FIELD_VALUE_LIMIT,
) -> Optional[str]:
    """明細の欄の値を作る。無ければ None。上限に入らない分は「ほか N 件」の行にまとめる"""
    texts = [format_trade_line(line, market) for line in lines]
    if not texts:
        return None
    shown: list[str] = []
    for i, text in enumerate(texts):
        remaining = len(texts) - i - 1
        candidate = shown + [text]
        if remaining:
            candidate = candidate + [_rest_text(remaining)]
        if len(shown) >= max_lines or len("\n".join(candidate)) > limit:
            break
        shown.append(text)
    rest = len(texts) - len(shown)
    if rest:
        shown.append(_rest_text(rest))
    return "\n".join(shown)


def safe_trade_detail_field(
    lines: Optional[Sequence[TradeLine]], market: str = "jp"
) -> Optional[dict]:
    """明細の embed の欄を返す。明細が無い・作れないときは None（知らせは件数だけで出す）"""
    if not lines:
        return None
    try:
        value = build_trade_detail_value(lines, market)
    except Exception as e:
        logger.error("約定・発注の明細を作れない。件数だけで知らせる: %s", e, exc_info=True)
        return None
    if value is None:
        return None
    return {"name": TRADE_DETAIL_FIELD_NAME, "value": value, "inline": False}
