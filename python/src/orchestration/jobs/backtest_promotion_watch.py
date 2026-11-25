"""バックテストによる仮の入れ替えの、入れ替えの後の見張り。

入れ替えの後に採点が済んだ日が5日たまれば1回だけ判定し、悪ければ同じ5日の行で
旧 production（写し）のほうが良いときだけ写しから戻す。
backtest_promotion_job._run_locked から _check_watch を呼ぶ（#2046 で本体から分けた）。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from src.orchestration.jobs.common import _local_now
from src.prediction.backtest_promotion import (
    MIN_COVERAGE,
    WATCH_TRADING_DAYS,
    judge_rollback,
    measure_old_production,
    restore_backups,
    save_state,
)
from src.prediction.backtest_promotion.old_production import with_base_market_day
from src.prediction.baseline_metrics import baseline_metrics
from src.prediction.db import load_prediction_accuracy
from src.prediction.types import BacktestPromotionState
from src.utils.japan_time import JAPAN_TIMEZONE, now_jst
from src.utils.logger import get_logger
from src.watchlist.batch_runner import load_target_symbols

logger = get_logger(__name__)

_WATCH_MARKET = "jp"  # 見張りの採点済みの日を決める市場（発注する市場）


def _predicted_at_jst(predicted_at: Any) -> Optional[datetime]:
    """predicted_at（プロセスの時計の YYYYMMDD_HHMMSS）を JST の時刻にする。読めなければ None。"""
    try:
        naive = datetime.strptime(str(predicted_at)[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=_local_now().tzinfo).astimezone(JAPAN_TIMEZONE)


def _scored_rows_since_swap(watch: dict) -> pd.DataFrame:
    """入れ替えの後に予想され、採点が済んだ production（horizon=1）の行。_day に JST の日付。

    predicted_at はプロセスの時計（本番は UTC）の時刻なので、日付でなく時刻で入れ替えの後かを見る。
    22:00 JST に入れ替えた翌朝 07:30 JST の回は、UTC では入れ替えと同じ日付（前日 22:30）になり、
    「predicted_at の日付 > swap_date」で比べるとその回を落とすためである。日数は JST の日付で数える。
    """
    swapped_at = datetime.fromisoformat(watch["swapped_at"])
    since_days = (now_jst().date() - date.fromisoformat(watch["swap_date"])).days + 2
    rows = load_prediction_accuracy(
        horizon=1, limit=0, model_name="production", days=max(since_days, 1), raise_on_error=True
    )
    if rows is None or rows.empty or "actual_ratio" not in rows.columns:
        return pd.DataFrame()
    rows = rows[pd.to_numeric(rows["actual_ratio"], errors="coerce").notna()]
    times = rows["predicted_at"].map(_predicted_at_jst)
    keep = times.map(lambda t: t is not None and t > swapped_at).astype(bool)
    rows = rows[keep]
    return rows.assign(_day=times[keep].map(lambda t: t.strftime("%Y-%m-%d")))


def _split_watch_days(rows: pd.DataFrame, need_symbols: float) -> tuple[list[str], list[str]]:
    """行のある日を、jp の銘柄の数が線に届く日と届かない日に分ける（どちらも日付の順）。

    同じ日の同じ銘柄の行は1銘柄に数える。
    """
    if rows.empty:
        return [], []
    jp_rows = rows[rows["market"] == _WATCH_MARKET]
    symbols_by_day = jp_rows.groupby("_day")["symbol"].nunique()
    counted: list[str] = []
    skipped: list[str] = []
    for day in sorted(rows["_day"].unique()):
        (counted if symbols_by_day.get(day, 0) >= need_symbols else skipped).append(day)
    return counted, skipped


def _check_watch(state: BacktestPromotionState, result: dict) -> None:
    """採点済みの日が5日たまっていれば1回だけ判定し、見張りを消す。たまっていなければ何もしない。

    採点済みの日は、jp の採点の行のある銘柄がウォッチリストの jp の MIN_COVERAGE 以上の日だけである。
    jp の採点がそろわない日で見張りを終わらせないため、足りない日は実測にも入れない（#1706）。
    分母は行から作らない。ウォッチリストが読めない回は例外のまま _run_locked の失敗の道に乗る。
    """
    watch = state.watch
    assert watch is not None
    verdict: dict = {"version": watch.get("version"), "scored_days": 0, "judged": False}
    result["watch"] = verdict
    n_universe = len({t.symbol for t in load_target_symbols() if t.market == _WATCH_MARKET})
    if n_universe == 0:
        verdict["note"] = f"ウォッチリストに {_WATCH_MARKET} の銘柄が無いため判定しない"
        logger.error("仮の入れ替えの見張り: %s", verdict["note"])
        return

    need_symbols = n_universe * MIN_COVERAGE
    rows = _scored_rows_since_swap(watch)
    days, skipped_days = _split_watch_days(rows, need_symbols)
    verdict.update(
        scored_days=len(days),
        skipped_days=skipped_days,
        n_universe_symbols=n_universe,
        need_symbols=need_symbols,
        need_days=WATCH_TRADING_DAYS,
    )
    if len(days) < WATCH_TRADING_DAYS:
        logger.info(
            "仮の入れ替えの見張り: 採点済み %d 日（%d 日で判定）、飛ばした日 %s（jp の銘柄が %g 未満）",
            len(days),
            WATCH_TRADING_DAYS,
            skipped_days,
            need_symbols,
        )
        _notify_skipped_days(state, result)
        return

    # 入れ替えの後の、最初の5日だけで測る
    first_days = set(days[:WATCH_TRADING_DAYS])
    first_rows = rows[rows["_day"].isin(first_days)]
    live = baseline_metrics(with_base_market_day(first_rows))
    # 旧 production は取っておいた20日でなく、同じ5日・同じ行で予想させて比べる（#1729）
    old = measure_old_production(watch.get("backups", []), first_rows)
    decision = judge_rollback(live, watch["backtest_candidate"], old["old"], old["new"])
    verdict.update(
        judged=True,
        days=sorted(first_days),
        live=live,
        old_production=old,
        compared_rows=old["n_compared"],
        triggered=decision.triggered,
        old_is_better=decision.old_is_better,
        rolled_back=False,
        backups=watch.get("backups", []),
    )
    if old["reason"] is not None:
        logger.warning("仮の入れ替えの見張り: 同じ5日の旧 production を測れない: %s", old["reason"])
    if decision.rollback:
        restore_backups(watch["backups"])
        verdict["rolled_back"] = True
        if watch["version"] not in state.rolled_back_versions:
            state.rolled_back_versions.append(watch["version"])
    elif decision.triggered and old["reason"] is not None:
        verdict["note"] = (
            f"見張りに当たったが、同じ5日の旧 production を測れないため戻さない: {old['reason']}"
        )
    elif decision.triggered:
        verdict["note"] = "見張りに当たったが、同じ5日の旧 production も良くないため戻さない"
    state.watch = None
    save_state(state)
    logger.info(
        "仮の入れ替えの見張りを終えた: rolled_back=%s triggered=%s 採点済み %d 日 飛ばした日 %s",
        verdict["rolled_back"],
        decision.triggered,
        len(days),
        skipped_days,
    )


def _notify_skipped_days(state: BacktestPromotionState, result: dict) -> None:
    """飛ばした日が WATCH_TRADING_DAYS に届いたら、見張りごとに1回だけ Discord に知らせる（#1713）。

    送れた回だけ見張りに印（skipped_days_notified）を付けて保存し、次の夜から重ねて出さない。
    送れない回・印を保存できない回はログに残すだけで、見張りと判定には触らない（次の夜にもう一度送る）。
    """
    from src.reporting.discord import notifications_backtest_watch as nw

    watch = state.watch
    verdict = result["watch"]
    if watch is None or watch.get("skipped_days_notified"):
        return
    if len(verdict["skipped_days"]) < WATCH_TRADING_DAYS:
        return
    try:
        sent = nw.send_backtest_promotion_watch_waiting(result)
    except Exception:
        logger.error("仮の入れ替えの見張りの、飛ばした日の通知に失敗", exc_info=True)
        return
    if not sent:
        logger.error("仮の入れ替えの見張りの、飛ばした日の通知を送れなかった")
        return
    watch["skipped_days_notified"] = True
    verdict["skipped_days_notified"] = True
    try:
        save_state(state)
    except Exception:
        logger.error(
            "飛ばした日の通知の印を保存できなかった（次の夜にもう一度送る）", exc_info=True
        )
