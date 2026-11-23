"""
Discord通知: ルールベースの日次シグナル

discord_utils.py から分けた（#1914）。discord_utils から re-export する。
"""

from typing import Optional

from src.reporting.discord.webhook_sender import send_webhook_text
from src.utils.db.symbol_master import symbol_name_suffix
from src.utils.japan_time import format_jst

RULE_TRACK_RECORD_HORIZON = "5"
RULE_TRACK_RECORD_MIN_N = 30


def _rule_track_record_text(track_record: dict, rule: str, direction: str) -> str:
    """ルールの実績（rule_accuracy の全期間の5日先）を1行に入る字に整える。"""
    stats = track_record.get(rule, {}).get(direction, {}).get(RULE_TRACK_RECORD_HORIZON, {})
    n = stats.get("n") or 0
    edge = stats.get("edge")
    if n == 0 or edge is None:
        return "実績なし"
    if n < RULE_TRACK_RECORD_MIN_N:
        return f"実績不足 n={n}"
    mean_return = stats.get("mean_signed_return")
    return_text = "" if mean_return is None else f" 平均{mean_return * 100:+.1f}%"
    return f"実績{edge * 100:+.1f}pt{return_text} n={n}"


def _rule_signal_line(signal: dict, market: str, direction: str, track_record: dict) -> str:
    price_str = f"{signal['price']:,.0f}円" if signal.get("price") else "---"
    record = _rule_track_record_text(track_record, signal["rule"], direction)
    return (
        f"  • `{signal['symbol']}`{symbol_name_suffix(market, signal['symbol'])} "
        f"[{signal['rule']}]  価格={price_str}  {record}  BT勝率={signal['win_rate']:.1%}"
    )


def _skip_reasons_line(skip_reasons: Optional[dict[str, int]]) -> str:
    """no_signal を除いた発注しなかった理由を、数の多い順の1行にする。無ければ空の字。"""
    items = [(k, n) for k, n in (skip_reasons or {}).items() if k != "no_signal" and n > 0]
    if not items:
        return ""
    items.sort(key=lambda kv: -kv[1])
    return "発注しなかった: " + " / ".join(f"{k}={n}" for k, n in items)


def send_rule_daily_signals(
    signals: list[dict],
    market: str,
    buy_orders: int,
    sell_orders: int,
    total: int,
    track_record: Optional[dict] = None,
    skip_reasons: Optional[dict[str, int]] = None,
) -> bool:
    """ルールベース日次シグナル通知を Discord に送信する。total は対象の銘柄数（#1571）。

    track_record はルールごとの実績（load_rule_track_record の返り値）。
    skip_reasons は発注しなかった理由ごとの数（execute_rule_paper_trades の返り値。#2114）。
    """
    track_record = track_record or {}
    now = format_jst(fmt="%Y/%m/%d %H:%M JST")
    buy_signals = [s for s in signals if s["signal"] == 1]
    sell_signals = [s for s in signals if s["signal"] == -1]

    lines = [
        f"**[ルールシグナル] {now}  ({market})**",
        f"BUY候補: {len(buy_signals)}銘柄  |  SELL候補: {len(sell_signals)}銘柄",
        f"ペーパー発注: BUY={buy_orders}  SELL={sell_orders}",
    ]
    skip_line = _skip_reasons_line(skip_reasons)
    if skip_line:
        lines.append(skip_line)
    lines.append("")

    if buy_signals:
        lines.append("**BUY シグナル:**")
        for s in buy_signals:
            lines.append(_rule_signal_line(s, market, "buy", track_record))

    if sell_signals:
        lines.append("")
        lines.append("**SELL シグナル:**")
        for s in sell_signals:
            lines.append(_rule_signal_line(s, market, "sell", track_record))

    if buy_signals or sell_signals:
        lines.append("")
        lines.append(
            "実績: 本番のシグナルの5日後の的中率と基準の差（pt）と平均の損益。BT: バックテスト"
        )

    if total <= 0:
        lines.append("❌ 有効ルールが0銘柄のため判定できず（保有の SELL も判定していない）")
    elif not buy_signals and not sell_signals:
        lines.append("本日はシグナルなし（全銘柄 HOLD）")

    return send_webhook_text("\n".join(lines))
