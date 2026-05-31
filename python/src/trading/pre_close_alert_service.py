"""引け前ポジション再評価アラートサービス（R-409）

15:00（JST）に保有ポジションを直近予測と照合し、
利確/損切り推奨度を分類して返す。
"""

from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 損切り推奨: モデル予測下落率がこの値以下
STOP_LOSS_PRED_THRESHOLD = -0.01
# 利確推奨: 保有含み益率がこの値以上 かつ予測が下落方向
TAKE_PROFIT_HOLD_THRESHOLD = 0.02


class AlertType(str, Enum):
    TAKE_PROFIT = "利確推奨"
    STOP_LOSS = "損切り推奨"
    HOLD = "保有継続"


@dataclass
class PositionAlert:
    symbol: str
    qty: int
    avg_price: float
    current_price: float
    avg_pred_price: float
    diff_ratio: float
    unrealized_pnl: float
    hold_pnl_ratio: float
    alert_type: AlertType


def evaluate_positions(market_data_port=None) -> list[PositionAlert]:
    """
    保有ポジション × 直近予測の差分を計算し推奨アクションを分類する。

    分類基準:
        損切り推奨: モデルの予測下落率 diff_ratio <= STOP_LOSS_PRED_THRESHOLD
        利確推奨: 保有含み益率 >= TAKE_PROFIT_HOLD_THRESHOLD かつ diff_ratio <= 0
        保有継続: 上記以外

    Args:
        market_data_port: MarketDataPort 実装（呼び出し元から注入）

    Returns:
        PositionAlert リスト（ポジションなし・予測データなしの場合は空リスト）
    """
    import os

    from src.trading.brokers.paper.paper_broker import PaperBroker
    from src.utils.db import load_latest_prediction_timestamp, load_prediction_results

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため引け前アラートをスキップ（kabuポジション未対応）")
        return []

    broker = PaperBroker(market_data_port=market_data_port)
    positions = broker.get_positions()
    if not positions:
        logger.info("保有ポジションなし — 引け前アラートをスキップ")
        return []

    ts = load_latest_prediction_timestamp()
    if ts is None:
        logger.warning("予測データなし — 引け前アラートをスキップ")
        return []

    pred_df = load_prediction_results(predicted_at=ts)
    pred_map: dict[str, dict] = {}
    if not pred_df.empty:
        for _, row in pred_df.iterrows():
            pred_map[str(row["symbol"])] = row.to_dict()

    alerts: list[PositionAlert] = []
    for pos in positions:
        symbol = str(pos["symbol"])
        current_price = float(pos["current_price"])
        avg_price = float(pos["avg_price"])
        qty = int(pos["qty"])
        unrealized_pnl = float(pos["unrealized_pnl"])
        hold_pnl_ratio = (current_price - avg_price) / avg_price if avg_price else 0.0

        pred = pred_map.get(symbol)
        if pred is not None:
            avg_pred_price = float(pred.get("avg_pred_price") or current_price)
            diff_ratio = float(pred.get("diff_ratio") or 0.0)
        else:
            avg_pred_price = current_price
            diff_ratio = 0.0

        if diff_ratio <= STOP_LOSS_PRED_THRESHOLD:
            alert_type = AlertType.STOP_LOSS
        elif hold_pnl_ratio >= TAKE_PROFIT_HOLD_THRESHOLD and diff_ratio < 0.0:
            alert_type = AlertType.TAKE_PROFIT
        else:
            alert_type = AlertType.HOLD

        alerts.append(
            PositionAlert(
                symbol=symbol,
                qty=qty,
                avg_price=avg_price,
                current_price=current_price,
                avg_pred_price=avg_pred_price,
                diff_ratio=diff_ratio,
                unrealized_pnl=unrealized_pnl,
                hold_pnl_ratio=hold_pnl_ratio,
                alert_type=alert_type,
            )
        )

    logger.info(
        "引け前アラート評価完了: %d件 (損切り=%d 利確=%d 保有継続=%d)",
        len(alerts),
        sum(1 for a in alerts if a.alert_type == AlertType.STOP_LOSS),
        sum(1 for a in alerts if a.alert_type == AlertType.TAKE_PROFIT),
        sum(1 for a in alerts if a.alert_type == AlertType.HOLD),
    )
    return alerts


def get_pre_close_alerts(market_data_port=None) -> list[str]:
    """evaluate_positions() の結果をフォーマット済み文字列リストで返す。"""
    from src.utils.japan_time import format_jst

    MINUTE_FORMAT = "%Y-%m-%d %H:%M JST"
    now = format_jst(fmt=MINUTE_FORMAT)

    alerts = evaluate_positions(market_data_port=market_data_port)

    if not alerts:
        return [f"時刻: {now}", "保有ポジションなし"]

    stop_loss = [a for a in alerts if a.alert_type == AlertType.STOP_LOSS]
    take_profit = [a for a in alerts if a.alert_type == AlertType.TAKE_PROFIT]
    hold = [a for a in alerts if a.alert_type == AlertType.HOLD]

    lines = [
        f"**[引け前ポジション再評価] {now}**",
        f"保有: {len(alerts)}銘柄  |  損切り: {len(stop_loss)}  利確: {len(take_profit)}  保有継続: {len(hold)}",
        "",
    ]

    if stop_loss:
        lines.append("🔴 **損切り推奨（SL接近）**")
        for a in stop_loss:
            lines.append(
                f"  • `{a.symbol}` "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み損益={a.unrealized_pnl:+,.0f}円"
            )

    if take_profit:
        lines.append("")
        lines.append("🟡 **利確推奨（目標値超）**")
        for a in take_profit:
            lines.append(
                f"  • `{a.symbol}` "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み益={a.unrealized_pnl:+,.0f}円({a.hold_pnl_ratio:+.1%})"
            )

    if hold:
        lines.append("")
        lines.append("🟢 **保有継続**")
        for a in hold:
            lines.append(
                f"  • `{a.symbol}` "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み損益={a.unrealized_pnl:+,.0f}円"
            )

    return lines
