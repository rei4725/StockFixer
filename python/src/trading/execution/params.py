"""発注パラメータ算出: 動的閾値・Kelly 入力・分割比率・注文種別判定。"""

import pandas as pd

from config.settings import (
    BUY_THRESHOLD,
    LIMIT_ORDER_AVG_VOLUME_THRESHOLD,
    LIMIT_ORDER_LOOKBACK_DAYS,
    LIMIT_ORDER_PRICE_BUFFER,
    LIMIT_ORDER_SPREAD_PROXY_THRESHOLD,
)
from src.domain.ports import MarketDataPort
from src.domain.trading_rules import THRESHOLD_SCALE_MAX as _THRESHOLD_SCALE_MAX
from src.domain.trading_rules import THRESHOLD_SCALE_MIN as _THRESHOLD_SCALE_MIN
from src.domain.trading_rules import THRESHOLD_SCALE_MIN_ROWS as _THRESHOLD_SCALE_MIN_ROWS
from src.trading.brokers.base import OrderSide, OrderType
from src.utils.data_path_utils import get_ticker
from src.utils.optimal_params_loader import get_optimal_params


def _compute_market_threshold_scale(predictions: pd.DataFrame) -> float:
    """当日の予測分布から実運用の閾値スケールを決定する。"""
    if predictions.empty or "diff_ratio" not in predictions.columns:
        return 1.0

    abs_diff = pd.to_numeric(predictions["diff_ratio"], errors="coerce").abs().dropna()
    if len(abs_diff) < _THRESHOLD_SCALE_MIN_ROWS:
        return 1.0

    median_abs = float(abs_diff.median())
    dispersion = float(abs_diff.std())
    if median_abs <= 0 or pd.isna(dispersion) or dispersion <= 0:
        return 1.0

    raw_scale = dispersion / median_abs
    return float(min(_THRESHOLD_SCALE_MAX, max(_THRESHOLD_SCALE_MIN, raw_scale)))


def _resolve_base_threshold(market: str, symbol: str) -> float:
    """銘柄別の最適閾値を取得し、未設定時は既定値にフォールバックする。"""
    params = get_optimal_params(market, symbol)
    raw_threshold = params.get("threshold") if params else None
    try:
        threshold = abs(float(raw_threshold)) if raw_threshold is not None else 0.0
    except (TypeError, ValueError):
        threshold = 0.0

    if threshold <= 0:
        threshold = abs(BUY_THRESHOLD)
    return threshold


def _attach_dynamic_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    """予測結果に実運用用の動的閾値列を付与する。"""
    if predictions.empty:
        return predictions.copy()

    scale = _compute_market_threshold_scale(predictions)
    threshold_cache: dict[tuple[str, str], float] = {}

    def _get_threshold(row: pd.Series) -> float:
        key = (str(row["market"]), str(row["symbol"]))
        if key not in threshold_cache:
            threshold_cache[key] = _resolve_base_threshold(*key)
        return threshold_cache[key]

    enriched = predictions.copy()
    enriched["base_threshold"] = enriched.apply(_get_threshold, axis=1)
    enriched["threshold_scale"] = scale
    enriched["effective_buy_threshold"] = enriched["base_threshold"] * scale
    enriched["effective_sell_threshold"] = -enriched["effective_buy_threshold"]
    return enriched


def _load_execution_metrics(
    market: str, symbol: str, market_data: MarketDataPort | None
) -> tuple[float | None, float | None]:
    if market_data is None:
        return None, None
    ticker = get_ticker(market, symbol)
    hist = market_data.get_ohlcv(ticker, period=f"{LIMIT_ORDER_LOOKBACK_DAYS + 2}d")
    if hist.empty:
        return None, None

    recent = hist.tail(LIMIT_ORDER_LOOKBACK_DAYS).copy()
    if recent.empty:
        return None, None

    avg_volume = None
    if "Volume" in recent.columns:
        volume_series = pd.to_numeric(recent["Volume"], errors="coerce").dropna()
        if not volume_series.empty:
            avg_volume = float(volume_series.mean())

    spread_proxy = None
    if {"High", "Low", "Close"}.issubset(recent.columns):
        close = pd.to_numeric(recent["Close"], errors="coerce")
        high = pd.to_numeric(recent["High"], errors="coerce")
        low = pd.to_numeric(recent["Low"], errors="coerce")
        range_ratio = ((high - low) / close.replace(0, pd.NA)).dropna()
        if not range_ratio.empty:
            spread_proxy = float(range_ratio.mean())

    return avg_volume, spread_proxy


def _resolve_kelly_params(
    market: str, symbol: str
) -> tuple[float | None, float | None, float | None]:
    """BT実績の Kelly 入力値を optimal_params.json から取得する。
    未登録・値が 0 以下・NaN の場合は None を返し、RiskManager のデフォルト値を使用させる。

    Returns:
        (win_rate, avg_win, avg_loss) — 各値が使用不能な場合は None
    """
    params = get_optimal_params(market, symbol)
    metrics = params.get("metrics", {}) if params else {}

    def _extract(key: str) -> float | None:
        v = metrics.get(key)
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return fv if fv > 0.0 else None

    return _extract("win_rate"), _extract("avg_win"), _extract("avg_loss")


# R-308: 分割エントリー/エグジット（確信度連動）
_SPLIT_HIGH_CONFIDENCE = 0.80
_SPLIT_LOW_CONFIDENCE = 0.50


def _calc_split_ratio(confidence_ratio: float) -> float:
    """確信度に応じた発注比率を返す（R-308）。

    Returns:
        1.0: confidence_ratio >= 0.80 → 全量発注
        0.5: 0.50 <= confidence_ratio < 0.80 → 1/2 発注
        0.0: confidence_ratio < 0.50 → 見送り
    """
    if confidence_ratio >= _SPLIT_HIGH_CONFIDENCE:
        return 1.0
    elif confidence_ratio >= _SPLIT_LOW_CONFIDENCE:
        return 0.5
    else:
        return 0.0


def _apply_split_qty(qty: int, split_ratio: float, lot: int = 100) -> int:
    """分割比率を株数に適用し、市場別の最低売買単位（lot）に丸める。"""
    return max(lot, int(qty * split_ratio // lot) * lot)


def _choose_order_params(
    market: str,
    symbol: str,
    side: OrderSide,
    current_price: float,
    market_data: MarketDataPort | None = None,
) -> tuple[OrderType, float, str, str]:
    """流動性指標から成行/指値・寄付/引けを自動判定する（R-103 拡張: R-405）。

    Returns:
        (OrderType, limit_price, reason_string, order_session)
        order_session: "open"（寄付）または "close"（引け）
    """
    avg_volume, spread_proxy = _load_execution_metrics(market, symbol, market_data)

    reasons: list[str] = []
    if avg_volume is not None and avg_volume < LIMIT_ORDER_AVG_VOLUME_THRESHOLD:
        reasons.append(f"low_volume={avg_volume:.0f}")
    if spread_proxy is not None and spread_proxy > LIMIT_ORDER_SPREAD_PROXY_THRESHOLD:
        reasons.append(f"wide_range={spread_proxy:.3%}")

    if not reasons:
        return OrderType.MARKET, 0.0, "market", "open"

    # 低流動性 → 引け指値（引け時点で流動性が集まりやすく価格優位性が高い）
    # 広スプレッドのみ → 寄付指値（寄付時の価格優位性を活用）
    order_session = "close" if any("low_volume" in r for r in reasons) else "open"
    buffer = current_price * LIMIT_ORDER_PRICE_BUFFER
    limit_price = (
        current_price + buffer if side == OrderSide.BUY else max(0.0, current_price - buffer)
    )
    return OrderType.LIMIT, limit_price, ", ".join(reasons), order_session
