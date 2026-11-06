"""月次ライブ・スクリーン + Discord アラート（#434）

月次で multibagger スクリーン（#433: トレンド + 質ゲート）を実行し、

1. 新規候補（上位N・スコア・主要指標）
2. 既存保有ポジション（#430 保有/撤退エンジン）の HOLD / SCALE_OUT / EXIT シグナル

を Discord に通知する。**売買は人間が最終判断（半自動）** のため、本ジョブは
発注を行わず、候補の財務と保有状態の更新、通知のみを担う。

レイヤー: orchestration（最上位）から screening / reporting / utils（下位）を参照する
のは合法（上位 → 下位）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config.settings import STALE_BARS_MAX_WEEKDAYS
from src.domain.types import PositionEvent, TrendCandidate
from src.market_data.fundamentals import fetch_fundamentals
from src.orchestration.job_errors import NonRetryableJobError
from src.screening.hold_engine import simulate_position
from src.screening.positions import close_position, list_positions, update_position
from src.screening.quality_gate import apply_quality_gate
from src.screening.trend_screener import (
    business_days_behind,
    last_valid_bar_day,
    screen_trend_candidates,
)
from src.screening.types import MultibaggerCandidate, Position
from src.utils.db.market_data import load_raw_ohlcv
from src.utils.db.stock_fundamentals import load_fundamentals, upsert_fundamentals
from src.utils.db.symbol_master import symbol_name_suffix
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 撤退を意味する PositionEvent.reason（end_of_data は「未撤退で系列末尾」を表す）
_EXIT_REASONS = {"ma_break", "trail_stop", "thesis_break"}

# 今回の取得に失敗した銘柄の前の行を、質ゲートに使ってよい長さ（as_of からの暦日）。
# 決算は四半期ごとに出るので、最も長い四半期の 92 日までの行なら、取りこぼす決算は
# おおむね1回分で済む。残りの 8 日は補完の回のずれの余裕である。
# 毎月の回で取得が3回続けて落ちた行までが線の内に入り、4回目で外れる
_FUNDAMENTALS_MAX_AGE_DAYS = 100

# 今回の財務を取れず、前の行で判定した銘柄に候補の表で付ける印
_PRIOR_ROW_MARK = " ※前の行で判定"


@dataclass(frozen=True)
class _FundamentalsRefresh:
    """質ゲートの前に行ったトレンド候補の財務の更新の結果。

    Attributes:
        total: トレンド候補の数。
        failed: 今回の取得・保存ができなかった銘柄の数。
        missing: failed のうち、前の行も無く質ゲートが判定できない銘柄の数。
        stale_symbols: failed のうち、前の行の as_of が線より古く質ゲートに渡さない銘柄。
        prior_symbols: failed のうち、線の内の前の行で質ゲートに渡す銘柄。
    """

    total: int = 0
    failed: int = 0
    missing: int = 0
    stale_symbols: frozenset[str] = frozenset()
    prior_symbols: frozenset[str] = frozenset()

    @property
    def all_missing(self) -> bool:
        """候補の全部で財務が無く、質ゲートが1銘柄も判定できないとき真。"""
        return self.total > 0 and self.missing == self.total

    @property
    def all_failed(self) -> bool:
        """候補の全部で今回の取得が落ちたとき真。前の行の有無は問わない。"""
        return self.total > 0 and self.failed == self.total

    @property
    def all_unjudged(self) -> bool:
        """行が無いか線より古いかで、候補の全部を質ゲートが判定できないとき真。"""
        return self.total > 0 and self.missing + len(self.stale_symbols) == self.total


def run_monthly_multibagger_screen(market: str = "us", top_n: int = 10) -> dict:
    """月次 multibagger スクリーン + 保有判定を実行し Discord 通知する。

    Args:
        market: 対象マーケット（既定 "us"）。
        top_n: 新規候補の通知件数。

    Returns:
        通知ペイロード（新規候補・保有アクション・本文）を含む dict。
        回の成否は raise_if_multibagger_failed で確かめる。
    """
    logger.info("=== 月次 multibagger スクリーン開始 (market=%s) ===", market)

    new_candidates, screen_failed, refresh = _screen_new_candidates(market, top_n)
    position_actions, skipped_positions = _evaluate_positions(market)

    message = _format_message(
        market,
        new_candidates,
        position_actions,
        skipped=skipped_positions,
        screen_failed=screen_failed,
        refresh=refresh,
    )
    notified = _notify(message)
    # 知らせが届かなかった EXIT は開いたまま残し、補完の回か次の月にもう一度判定して知らせる
    close_failed = _close_exited_positions(position_actions) if notified else []

    logger.info(
        "=== 月次 multibagger スクリーン完了: 新規候補=%d 件, 保有判定=%d 件, 判定できず=%d 件 ===",
        len(new_candidates),
        len(position_actions),
        len(skipped_positions),
    )
    return {
        "market": market,
        "new_candidates": new_candidates,
        "screen_failed": screen_failed,
        "fundamentals_failed": refresh.failed,
        "fundamentals_missing": refresh.missing,
        "fundamentals_stale": len(refresh.stale_symbols),
        "fundamentals_all_missing": refresh.all_missing,
        "fundamentals_all_failed": refresh.all_failed,
        "position_actions": position_actions,
        "skipped_positions": skipped_positions,
        "close_failed": close_failed,
        "notified": notified,
        "message": message,
    }


def raise_if_multibagger_failed(result: dict) -> None:
    """run_monthly_multibagger_screen の結果に欠けがあれば、ジョブを失敗にする。

    欠けのある回を success と記録すると、その月の判定が補完されない。すぐ呼び直しても
    同じ失敗と同じ知らせを重ねるだけなので NonRetryableJobError を投げ、補完の回に任せる。

    Raises:
        NonRetryableJobError: スクリーンの失敗・候補の全部で財務が無い・候補の全部で財務の
            取得が落ちた・判定できない保有がある・閉じられなかった EXIT がある・知らせの
            送信の失敗のどれかがあるとき。
    """
    reasons = _failure_reasons(result)
    if reasons:
        detail = "、".join(reasons)
        logger.error("月次 multibagger の回に欠けがある: %s", detail)
        raise NonRetryableJobError(f"月次 multibagger の回に欠けがある: {detail}")


def _failure_reasons(result: dict) -> list[str]:
    """結果の欠けを、ジョブの失敗の文に載せる語にして返す。"""
    reasons: list[str] = []
    if result["screen_failed"]:
        reasons.append("スクリーンの失敗")
    if result["fundamentals_all_missing"]:
        reasons.append(f"候補の全部で財務が無い {result['fundamentals_missing']} 件")
    elif result["fundamentals_all_failed"]:
        reasons.append(f"候補の全部で財務の取得が落ちた {result['fundamentals_failed']} 件")
    if result["skipped_positions"]:
        reasons.append(f"判定できない保有 {len(result['skipped_positions'])} 件")
    if result["close_failed"]:
        reasons.append(f"閉じられなかった EXIT {len(result['close_failed'])} 件")
    if not result["notified"]:
        reasons.append("知らせの送信の失敗")
    return reasons


class _PositionSkipped(Exception):
    """保有の判定ができなかったこと（価格データなし・足が古い・イベントなし）を表す。"""


def _screen_new_candidates(
    market: str, top_n: int
) -> tuple[list[MultibaggerCandidate], bool, _FundamentalsRefresh]:
    """#433 の最終スクリーン（トレンド + 質ゲート）で新規候補を取得する。

    質ゲートは ``stock_fundamentals`` を読むだけなので、その前にトレンド候補の財務を更新する。
    今回の取得に失敗し、前の行が線より古い銘柄は質ゲートに渡さない。

    Returns:
        (新規候補, スクリーンが例外で失敗したか, 財務の更新の結果)
    """
    try:
        trend = screen_trend_candidates(market=market)
        refresh = _refresh_fundamentals(trend)
        final = apply_quality_gate([c for c in trend if c.symbol not in refresh.stale_symbols])
        return final[:top_n], False, refresh
    except Exception as e:
        logger.error("新規候補スクリーン失敗: %s", e, exc_info=True)
        return [], True, _FundamentalsRefresh()


def _refresh_fundamentals(trend: list[TrendCandidate]) -> _FundamentalsRefresh:
    """トレンド候補の財務を取得・保存する。1銘柄の失敗で残りの銘柄を止めない。"""
    failed = [c for c in trend if not _refresh_one_fundamental(c)]
    rows = {c.symbol: load_fundamentals(c.market, c.symbol) for c in failed}
    missing = sum(1 for row in rows.values() if row is None)
    now = datetime.now(timezone.utc)
    stale = frozenset(s for s, row in rows.items() if row is not None and _is_stale(row, now))
    prior = frozenset(s for s, row in rows.items() if row is not None) - stale
    logger.info(
        "財務更新完了: 成功=%d/%d 件, 判定できず=%d 件 (行なし=%d, 線より古い=%s), 前の行で判定=%s",
        len(trend) - len(failed),
        len(trend),
        missing + len(stale),
        missing,
        sorted(stale),
        sorted(prior),
    )
    return _FundamentalsRefresh(
        total=len(trend),
        failed=len(failed),
        missing=missing,
        stale_symbols=stale,
        prior_symbols=prior,
    )


def _is_stale(row: dict, now: datetime) -> bool:
    """前の行の as_of が線より古いかを返す。

    as_of が読めない行は線の内と確かめられないので、古い行として扱う。列は時差なしの
    timestamp なので、時差の無い値は UTC として読む（線の長さに比べて時差は無視できる）。
    """
    as_of = pd.to_datetime(row.get("as_of"), utc=True, errors="coerce")
    if pd.isna(as_of):
        return True
    return now - as_of > timedelta(days=_FUNDAMENTALS_MAX_AGE_DAYS)


def _refresh_one_fundamental(candidate: TrendCandidate) -> bool:
    """1銘柄の財務を取得・保存し、できたかを返す。"""
    try:
        record = fetch_fundamentals(candidate.market, candidate.symbol)
        if record is None:
            logger.warning("[スキップ] 財務取得失敗: %s/%s", candidate.market, candidate.symbol)
            return False
        upsert_fundamentals(record)
        return True
    except Exception as e:
        logger.error(
            "財務更新失敗 (%s/%s): %s", candidate.market, candidate.symbol, e, exc_info=True
        )
        return False


def _evaluate_positions(market: str) -> tuple[list[dict], list[dict]]:
    """open ポジションを #430 保有エンジンに通し、アクションを判定・永続化する。

    Returns:
        (判定できたアクション, 判定できなかった保有の symbol と理由)
    """
    actions: list[dict] = []
    skipped: list[dict] = []
    for pos in list_positions(status="open"):
        if pos.market != market:
            continue
        try:
            actions.append(_evaluate_one_position(pos))
        except _PositionSkipped as e:
            skipped.append({"market": pos.market, "symbol": pos.symbol, "reason": str(e)})
        except Exception as e:
            logger.error("保有判定失敗 (%s/%s): %s", pos.market, pos.symbol, e, exc_info=True)
            skipped.append(
                {"market": pos.market, "symbol": pos.symbol, "reason": f"error:{type(e).__name__}"}
            )
    return actions, skipped


def _today() -> date:
    """足の鮮度の基準の日（UTC の今日）を返す。"""
    return datetime.now(timezone.utc).date()


def _load_position_prices(pos: Position) -> pd.DataFrame:
    """保有の判定に使う足を market_data_raw から読み、date と Close の表にして返す。

    stock_features は生の Close を持たない。date は entry_date と比べるので同じ
    YYYY-MM-DD の文字列にする。最後の有効な足がトレンド・スクリーナーと同じ線より古ければ、
    止まった日の終値で判定しない。

    Raises:
        _PositionSkipped: 有効な足が無いか、最後の足が線より古いとき。
    """
    raw = load_raw_ohlcv(pos.market, pos.symbol)
    last_bar = last_valid_bar_day(raw)
    if raw is None or last_bar is None:
        logger.warning("価格データなし（判定スキップ）: %s/%s", pos.market, pos.symbol)
        raise _PositionSkipped("no_prices")

    behind = business_days_behind(last_bar, _today(), pos.market)
    if behind > STALE_BARS_MAX_WEEKDAYS:
        logger.warning(
            "足が止まった保有（判定スキップ）: %s/%s 最後の足 %s、営業日で %d日 > %d日 古い",
            pos.market,
            pos.symbol,
            last_bar,
            behind,
            STALE_BARS_MAX_WEEKDAYS,
        )
        raise _PositionSkipped("stale_prices")

    dates = pd.DatetimeIndex(raw.index).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "Close": raw["Close"].to_numpy()})


def _evaluate_one_position(pos: Position) -> dict:
    """1 ポジションを最新価格まで保有シミュレートし、アクションを確定する。

    HOLD / SCALE_OUT はここで保存する。EXIT は知らせが届いた後に
    _close_exited_positions が閉じる（閉じた保有は次の回の判定に出ないため）。

    Raises:
        _PositionSkipped: 価格データが無いか古いか、保有イベントが無く判定できないとき。
    """
    prices = _load_position_prices(pos)

    events = simulate_position(prices, pos.entry_date)
    if not events:
        logger.warning("保有イベントなし（判定スキップ）: %s/%s", pos.market, pos.symbol)
        raise _PositionSkipped("no_events")

    action, reason, held_fraction, multiple, price = _interpret_events(events)

    if action != "EXIT":
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


def _close_exited_positions(actions: list[dict]) -> list[str]:
    """知らせを送った EXIT の保有を閉じ、閉じられなかった symbol を返す。"""
    failed: list[str] = []
    for a in actions:
        if a["action"] != "EXIT":
            continue
        try:
            close_position(a["market"], a["symbol"], close_price=a["price"], reason=a["reason"])
        except Exception as e:
            logger.error("保有を閉じられず (%s/%s): %s", a["market"], a["symbol"], e, exc_info=True)
            failed.append(a["symbol"])
    return failed


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
    skipped: Optional[list[dict]] = None,
    screen_failed: bool = False,
    refresh: Optional[_FundamentalsRefresh] = None,
) -> str:
    """新規候補 + 保有アクションを Discord 向けテキスト（コードフェンス）に整形する。

    判定できなかった保有は「判定できず」として載せ、開いた保有があるときは
    「保有なし」と書かない。スクリーンの失敗は「該当なし」と分けて書く。
    候補の全部で財務が無いか線より古いときも「該当なし」と書かず、判定できなかった数を書く。
    財務を取れなかった銘柄があるときは、その数を添え、前の行で判定した銘柄に表で印を付ける。
    """
    skipped = skipped or []
    refresh = refresh or _FundamentalsRefresh()
    lines: list[str] = [f"📈 月次 multibagger スクリーン ({market.upper()})", ""]

    lines.append("【新規候補】")
    if screen_failed:
        lines.append("⚠️ スクリーン失敗（候補を判定できず）")
    elif candidates:
        lines.append("```")
        lines.append(f"{'symbol':<8}{'score':>8}{'close':>10}{'CAGR':>8}{'ROE':>8}")
        for c in candidates:
            lines.append(
                f"{c.symbol:<8}{c.multibagger_score:>8.3f}{c.close:>10.2f}"
                f"{_pct(c.revenue_cagr_3y):>8}{_pct(c.roe):>8}"
                f"{symbol_name_suffix(market, c.symbol)}"
                f"{_PRIOR_ROW_MARK if c.symbol in refresh.prior_symbols else ''}"
            )
        lines.append("```")
    elif refresh.all_missing:
        lines.append(f"⚠️ 財務が無く判定できず: {refresh.missing}/{refresh.total} 件")
    elif not refresh.all_unjudged:
        lines.append("該当なし")
    if refresh.failed and not refresh.all_missing:
        lines.append(f"⚠️ 財務を取れず: {refresh.failed}/{refresh.total} 件")
    if refresh.stale_symbols:
        lines.append(f"⚠️ 財務が古く判定できず: {len(refresh.stale_symbols)}/{refresh.total} 件")

    lines.append("")
    lines.append("【保有ポジション】")
    if actions:
        lines.append("```")
        lines.append(f"{'symbol':<8}{'action':<10}{'mult':>7}{'reason':>14}")
        for a in actions:
            lines.append(
                f"{a['symbol']:<8}{a['action']:<10}{a['multiple']:>6.2f}x{a['reason']:>14}"
                f"{symbol_name_suffix(market, a['symbol'])}"
            )
        lines.append("```")
    if skipped:
        lines.append(f"⚠️ 判定できず: {len(skipped)} 件")
        lines.append("```")
        for s in skipped:
            lines.append(f"{s['symbol']:<8}{s['reason']}{symbol_name_suffix(market, s['symbol'])}")
        lines.append("```")
    if not actions and not skipped:
        lines.append("保有なし")

    return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    """比率を百分率表記（None/NaN は "-"）に整形する。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value * 100:.1f}%"


def _notify(message: str) -> bool:
    """Discord に通知し、送れたかを返す。

    送信の失敗はここでは投げず、残りの処理を走らせる。ジョブの失敗にするのは
    raise_if_multibagger_failed である。
    """
    try:
        from src.reporting.discord.discord_utils import send_webhook_text_chunked

        if send_webhook_text_chunked(message):
            return True
        logger.error("月次 multibagger 通知失敗: 送信が False を返した")
        return False
    except Exception as e:
        logger.error("月次 multibagger 通知失敗: %s", e, exc_info=True)
        return False
