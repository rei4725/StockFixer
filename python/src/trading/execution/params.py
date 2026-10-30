"""発注パラメータ算出: 動的閾値・Kelly 入力・分割比率・注文種別判定。"""

from dataclasses import replace

import pandas as pd

from config.settings import (
    BUY_THRESHOLD,
    LIMIT_ORDER_AVG_VOLUME_THRESHOLD,
    LIMIT_ORDER_LOOKBACK_DAYS,
    LIMIT_ORDER_PRICE_BUFFER,
    LIMIT_ORDER_SPREAD_PROXY_THRESHOLD,
)
from src.domain.limit_order_rules import plan_limit_order
from src.domain.ports import MarketDataPort
from src.domain.trading_rules import THRESHOLD_SCALE_MAX as _THRESHOLD_SCALE_MAX
from src.domain.trading_rules import THRESHOLD_SCALE_MIN as _THRESHOLD_SCALE_MIN
from src.domain.trading_rules import THRESHOLD_SCALE_MIN_ROWS as _THRESHOLD_SCALE_MIN_ROWS
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.types import TradingGateStatus
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger
from src.utils.optimal_params_loader import OptimalParamsUnreadableError, get_optimal_params

from .stats import OrderExecutionStats

logger = get_logger(__name__)

# 閾値の既定値を使った印。runner はこの印で新規を止め、エラーに数える（#932・#1214）
_PARAMS_UNREADABLE_COL = "optimal_params_unreadable"


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


def _read_optimal_params(market: str, symbol: str, what: str) -> tuple[dict | None, bool]:
    """optimal_params.json が読めない日は ERROR を出し、未登録と同じく None で続ける（#1214）。

    Returns:
        (params, readable) — 読めなかったときは (None, False)
    """
    try:
        return get_optimal_params(market, symbol), True
    except OptimalParamsUnreadableError:
        logger.error(
            "[exec] %s: optimal_params.json が読めない。既定の%sで続ける",
            symbol,
            what,
            exc_info=True,
        )
        return None, False


def _base_threshold_with_readable(market: str, symbol: str) -> tuple[float, bool]:
    params, readable = _read_optimal_params(market, symbol, "閾値")
    raw_threshold = params.get("threshold") if params else None
    try:
        threshold = abs(float(raw_threshold)) if raw_threshold is not None else 0.0
    except (TypeError, ValueError):
        threshold = 0.0

    if threshold <= 0:
        threshold = abs(BUY_THRESHOLD)
    return threshold, readable


def _resolve_base_threshold(market: str, symbol: str) -> float:
    """銘柄別の最適閾値を取得し、未設定時・読めない日は既定値にフォールバックする。"""
    return _base_threshold_with_readable(market, symbol)[0]


def _block_new_if_params_unreadable(
    predictions: pd.DataFrame, gate_status: TradingGateStatus, stats: OrderExecutionStats
) -> TradingGateStatus:
    """閾値を既定値で決めた日は、新規（買い・ショート）を止めてエラーに数える（#932・#1214）。

    決済（予想の売り・SHORT_COVER）はゲートによらず出るので、保有は手放せる。
    """
    if _PARAMS_UNREADABLE_COL not in predictions.columns:
        return gate_status
    if not predictions[_PARAMS_UNREADABLE_COL].astype(bool).any():
        return gate_status
    reason = "optimal_params.json が読めないため既定の閾値で決済だけ行う"
    logger.error("[exec] %s（新規は止める）", reason)
    stats["errors"] += 1
    return replace(gate_status, is_allowed=False, reason=gate_status.reason or reason)


def _attach_dynamic_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    """予測結果に実運用用の動的閾値列を付与する。"""
    if predictions.empty:
        return predictions.copy()

    scale = _compute_market_threshold_scale(predictions)
    threshold_cache: dict[tuple[str, str], tuple[float, bool]] = {}

    def _cached(row: pd.Series) -> tuple[float, bool]:
        key = (str(row["market"]), str(row["symbol"]))
        if key not in threshold_cache:
            threshold_cache[key] = _base_threshold_with_readable(*key)
        return threshold_cache[key]

    enriched = predictions.copy()
    enriched["base_threshold"] = enriched.apply(lambda row: _cached(row)[0], axis=1)
    if not all(readable for _, readable in threshold_cache.values()):
        enriched[_PARAMS_UNREADABLE_COL] = enriched.apply(lambda row: not _cached(row)[1], axis=1)
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
    ただし win_rate が 0 で avg_loss が正（取引があり全部負けた）の場合は win_rate を 0.0 で
    返し、Kelly を 0 にして新規を見送らせる（#1259）。

    Returns:
        (win_rate, avg_win, avg_loss) — 各値が使用不能な場合は None
    """
    params, _ = _read_optimal_params(market, symbol, "Kelly の入力")
    metrics = params.get("metrics", {}) if params else {}

    def _extract(key: str) -> float | None:
        v = metrics.get(key)
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return fv if fv > 0.0 else None

    win_rate, avg_win, avg_loss = _extract("win_rate"), _extract("avg_win"), _extract("avg_loss")
    if win_rate is None and avg_loss is not None:
        raw_win_rate: float | None
        try:
            raw_win_rate = float(metrics.get("win_rate"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raw_win_rate = None
        if raw_win_rate == 0.0:
            win_rate = 0.0
    return win_rate, avg_win, avg_loss


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

    # 判定はバックテストと同じ domain の関数で行う（#1418）
    plan = plan_limit_order(
        market,
        side in (OrderSide.BUY, OrderSide.SHORT_COVER),
        current_price,
        avg_volume,
        spread_proxy,
        LIMIT_ORDER_AVG_VOLUME_THRESHOLD,
        LIMIT_ORDER_SPREAD_PROXY_THRESHOLD,
        LIMIT_ORDER_PRICE_BUFFER,
    )
    if not plan.is_limit:
        return OrderType.MARKET, 0.0, "market", "open"
    return OrderType.LIMIT, plan.limit_price, ", ".join(plan.reasons), plan.session
