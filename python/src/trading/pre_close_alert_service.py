"""引け前ポジション再評価アラートサービス（R-409）

15:00（JST）に保有ポジションを直近予測と照合し、
利確/損切り推奨度を分類して返す。
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from src.domain.jp_market_holidays import is_jp_business_day
from src.domain.ports import PredictionResultRepository, TradeDiffSink
from src.utils.db.symbol_master import symbol_name_suffix
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
    NO_PREDICTION = "予想なし"


class SkipReason(str, Enum):
    """アラートを出さなかった理由。"""

    MODE_UNSUPPORTED = "対象外の発注のモード"
    NO_POSITIONS = "保有ポジションなし"
    NO_PREDICTION_RUN = "予想の回が無い"
    STALE_PREDICTION_RUN = "予想の回が古い"


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


@dataclass
class PreCloseEvaluation:
    """引け前の評価の結果。alerts が空のときは skip_reason に理由を持つ。"""

    alerts: list[PositionAlert]
    skip_reason: SkipReason | None = None
    predicted_at: str | None = None


def _today_jst() -> date:
    """日本時間の今日（テストで差し替える）。"""
    return datetime.now(ZoneInfo("Asia/Tokyo")).date()


def _previous_business_day(today: date) -> date:
    """today の前の東証の営業日（土日と表の休場日を飛ばす）。"""
    previous = today - timedelta(days=1)
    while not is_jp_business_day(previous):
        previous -= timedelta(days=1)
    return previous


def _run_date(predicted_at: str) -> date | None:
    """予想の回の時刻（20261021_073000 か ISO 形式）から日付を読む。読めなければ None。"""
    try:
        return datetime.strptime(predicted_at[:15], "%Y%m%d_%H%M%S").date()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(predicted_at).date()
    except (ValueError, TypeError):
        return None


def evaluate_positions(
    market_data_port=None,
    *,
    trade_diff_sink: TradeDiffSink,
    prediction_repo: PredictionResultRepository,
) -> list[PositionAlert]:
    """
    保有ポジション × 直近予測の差分を計算し推奨アクションを分類する。

    分類基準:
        損切り推奨: モデルの予測下落率 diff_ratio <= STOP_LOSS_PRED_THRESHOLD
        利確推奨: 保有含み益率 >= TAKE_PROFIT_HOLD_THRESHOLD かつ diff_ratio <= 0
        予想なし: 予想の回にその銘柄が載っていない
        保有継続: 上記以外

    Args:
        market_data_port: MarketDataPort 実装（呼び出し元から注入）
        trade_diff_sink: 約定乖離の記録先（TradeDiffSink 実装。合成ルートが必ず渡す）
        prediction_repo: 予測結果の読み取り元（合成ルートが必ず渡す）

    Returns:
        PositionAlert リスト（ポジションなし・予測データなしの場合は空リスト。
        空の理由は evaluate_pre_close の skip_reason で読む）
    """
    return evaluate_pre_close(
        market_data_port=market_data_port,
        trade_diff_sink=trade_diff_sink,
        prediction_repo=prediction_repo,
    ).alerts


def evaluate_pre_close(
    market_data_port=None,
    *,
    trade_diff_sink: TradeDiffSink,
    prediction_repo: PredictionResultRepository,
) -> PreCloseEvaluation:
    """evaluate_positions と同じ評価をし、アラートを出さなかった理由も返す。"""
    import os

    from src.trading.brokers.paper.paper_broker import PaperBroker

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため引け前アラートをスキップ（kabuポジション未対応）")
        return PreCloseEvaluation(alerts=[], skip_reason=SkipReason.MODE_UNSUPPORTED)

    broker = PaperBroker(
        market_data_port=market_data_port,
        trade_diff_sink=trade_diff_sink,
    )
    positions = broker.get_positions()
    if not positions:
        logger.info("保有ポジションなし — 引け前アラートをスキップ")
        return PreCloseEvaluation(alerts=[], skip_reason=SkipReason.NO_POSITIONS)

    # production の後に保存される challenger の回を掴まないよう、版を絞る
    ts = prediction_repo.latest_timestamp(model_version="production")
    if ts is None:
        logger.warning("予測データなし — 引け前アラートをスキップ")
        return PreCloseEvaluation(alerts=[], skip_reason=SkipReason.NO_PREDICTION_RUN)

    # 朝の予想が止まった日に、前の日より古い回を今日の予想として使わない
    run_date = _run_date(ts)
    previous_day = _previous_business_day(_today_jst())
    if run_date is not None and run_date < previous_day:
        logger.warning(
            "production の予想の回 %s が前の営業日 %s より古い — 引け前アラートをスキップ",
            ts,
            previous_day.isoformat(),
        )
        return PreCloseEvaluation(
            alerts=[], skip_reason=SkipReason.STALE_PREDICTION_RUN, predicted_at=ts
        )

    pred_df = prediction_repo.results_at(predicted_at=ts)
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

        if pred is None:
            # 予想の回に載らない保有を、モデルが「持っていてよい」と言った保有継続に混ぜない
            alert_type = AlertType.NO_PREDICTION
        elif diff_ratio <= STOP_LOSS_PRED_THRESHOLD:
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

    no_prediction = [a.symbol for a in alerts if a.alert_type == AlertType.NO_PREDICTION]
    if no_prediction:
        logger.warning(
            "予想の回 %s に載らない保有（予想なし）が %d件: %s",
            ts,
            len(no_prediction),
            ", ".join(no_prediction),
        )

    logger.info(
        "引け前アラート評価完了: %d件 (損切り=%d 利確=%d 保有継続=%d 予想なし=%d)",
        len(alerts),
        sum(1 for a in alerts if a.alert_type == AlertType.STOP_LOSS),
        sum(1 for a in alerts if a.alert_type == AlertType.TAKE_PROFIT),
        sum(1 for a in alerts if a.alert_type == AlertType.HOLD),
        len(no_prediction),
    )
    return PreCloseEvaluation(alerts=alerts, predicted_at=ts)


def _skip_lines(now: str, evaluation: PreCloseEvaluation) -> list[str]:
    """アラートを出さなかったときの文を、理由ごとに分けて返す。"""
    reason = evaluation.skip_reason
    if reason == SkipReason.NO_PREDICTION_RUN:
        return [
            f"時刻: {now}",
            "⚠️ production の予想の回が無いため、引け前アラートを出せない（朝の予想が保存されていない）",
        ]
    if reason == SkipReason.STALE_PREDICTION_RUN:
        return [
            f"時刻: {now}",
            f"⚠️ production の予想の回が古い（{evaluation.predicted_at}）ため、"
            "引け前アラートを出せない（朝の予想が止まっている）",
        ]
    if reason == SkipReason.MODE_UNSUPPORTED:
        return [f"時刻: {now}", "この発注のモードでは引け前アラートを出さない"]
    return [f"時刻: {now}", "保有ポジションなし"]


def get_pre_close_alerts(
    market_data_port=None,
    *,
    trade_diff_sink: TradeDiffSink,
    prediction_repo: PredictionResultRepository,
) -> list[str]:
    """evaluate_positions() の結果をフォーマット済み文字列リストで返す。"""
    from src.utils.japan_time import format_jst

    MINUTE_FORMAT = "%Y-%m-%d %H:%M JST"
    now = format_jst(fmt=MINUTE_FORMAT)

    evaluation = evaluate_pre_close(
        market_data_port=market_data_port,
        trade_diff_sink=trade_diff_sink,
        prediction_repo=prediction_repo,
    )
    alerts = evaluation.alerts

    if not alerts:
        return _skip_lines(now, evaluation)

    stop_loss = [a for a in alerts if a.alert_type == AlertType.STOP_LOSS]
    take_profit = [a for a in alerts if a.alert_type == AlertType.TAKE_PROFIT]
    hold = [a for a in alerts if a.alert_type == AlertType.HOLD]
    no_prediction = [a for a in alerts if a.alert_type == AlertType.NO_PREDICTION]

    lines = [
        f"**[引け前ポジション再評価] {now}**",
        f"保有: {len(alerts)}銘柄  |  損切り: {len(stop_loss)}  利確: {len(take_profit)}  "
        f"保有継続: {len(hold)}  予想なし: {len(no_prediction)}",
        "",
    ]

    if stop_loss:
        lines.append("🔴 **損切り推奨（SL接近）**")
        for a in stop_loss:
            lines.append(
                f"  • `{a.symbol}`{symbol_name_suffix(None, a.symbol)} "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み損益={a.unrealized_pnl:+,.0f}円"
            )

    if take_profit:
        lines.append("")
        lines.append("🟡 **利確推奨（目標値超）**")
        for a in take_profit:
            lines.append(
                f"  • `{a.symbol}`{symbol_name_suffix(None, a.symbol)} "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み益={a.unrealized_pnl:+,.0f}円({a.hold_pnl_ratio:+.1%})"
            )

    if hold:
        lines.append("")
        lines.append("🟢 **保有継続**")
        for a in hold:
            lines.append(
                f"  • `{a.symbol}`{symbol_name_suffix(None, a.symbol)} "
                f"現在値={a.current_price:,.0f}円 "
                f"予測={a.avg_pred_price:,.0f}円({a.diff_ratio:+.1%}) "
                f"含み損益={a.unrealized_pnl:+,.0f}円"
            )

    if no_prediction:
        lines.append("")
        lines.append("⚪ **予想なし（予想の回に載らない銘柄。モデルの判断ではない）**")
        for a in no_prediction:
            lines.append(
                f"  • `{a.symbol}`{symbol_name_suffix(None, a.symbol)} "
                f"現在値={a.current_price:,.0f}円 "
                f"含み損益={a.unrealized_pnl:+,.0f}円({a.hold_pnl_ratio:+.1%})"
            )

    return lines
