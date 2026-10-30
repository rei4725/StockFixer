"""日次ジョブの「取れなかった銘柄が多すぎる」判定。

daily.py からの責務分離（ファイル行数ガード対策）。
"""

from typing import TYPE_CHECKING, Optional

from src.orchestration.jobs.common import _notify_stage_failure
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.domain.types import BatchResult
    from src.rule_engine.types import RuleSignalBatch

logger = get_logger(__name__)

# 取得のエラーの知らせに並べる銘柄の上限（#1256）
_FETCH_ERROR_NOTICE_SYMBOL_LIMIT = 5

# 8:50 の自動発注が注文を出す市場（#1432）
_ORDER_MARKET = "jp"


def _raise_if_fetch_mostly_unavailable(
    total: int,
    batch_result: "BatchResult",
    market_totals: Optional[dict[str, int]] = None,
) -> None:
    """取得のスキップ+エラーが対象の線を超えたら RuntimeError を投げる（#811）。

    run_batch_pipeline は銘柄ごとのスキップ・エラーを例外にしないため、全銘柄が
    スキップされても [1/6] が「完了」になり、古い特徴量で予測を保存し続けた。
    全体の割合に加えて、8:50 が発注する市場（jp）の銘柄だけの割合も同じ線で見る。
    jp は全体の約3割のため、jp が大きく欠けても全体の線に届かない（#1432）。
    線の下でも取得のエラーが1件以上あれば、件数と先頭の銘柄を1回知らせる（#1256）。
    毎日あるスキップ（足の止まった銘柄）は知らせに数えない。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    if total <= 0:
        return
    unavailable = len(batch_result.skipped) + len(batch_result.failed)
    if unavailable / total > DAILY_FETCH_MAX_UNAVAILABLE_RATIO:
        raise RuntimeError(
            f"取得できなかった銘柄が多すぎる: スキップ {len(batch_result.skipped)}"
            f" + エラー {len(batch_result.failed)} / 対象 {total}"
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）"
        )
    _raise_if_order_market_mostly_unavailable(market_totals or {}, batch_result)
    _notify_fetch_errors(total, batch_result)


def _raise_if_order_market_mostly_unavailable(
    market_totals: dict[str, int], batch_result: "BatchResult"
) -> None:
    """発注する市場の銘柄だけのスキップ+エラーが線を超えたら RuntimeError を投げる（#1432）。

    一部の jp 銘柄の予想だけで 8:50 の新規が出て、予想の無い保有の売りが欠けるのを止める。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    market = _ORDER_MARKET
    market_total = market_totals.get(market, 0)
    if market_total <= 0:
        return
    n_skipped = sum(1 for s in batch_result.skipped if _skipped_market(s) == market)
    n_failed = sum(1 for f in batch_result.failed if f.market == market)
    if (n_skipped + n_failed) / market_total > DAILY_FETCH_MAX_UNAVAILABLE_RATIO:
        raise RuntimeError(
            f"{market} の取得できなかった銘柄が多すぎる: スキップ {n_skipped}"
            f" + エラー {n_failed} / {market} の対象 {market_total}"
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）"
        )


def _skipped_market(skipped: object) -> Optional[str]:
    """スキップの結果（market の鍵を持つ dict）から市場を返す。"""
    if isinstance(skipped, dict):
        return skipped.get("market")
    return getattr(skipped, "market", None)


def _notify_fetch_errors(total: int, batch_result: "BatchResult") -> None:
    """取得のエラーが1件以上なら、件数と先頭の銘柄（市場つき）を知らせる（#1256）。

    取れなかった銘柄は予想・採点・保有の売りの材料が前の日のままになるため、
    後の段は止めずに、手で取り直せるようその日のうちに知らせる。
    """
    if not batch_result.failed:
        return
    names = [f"{f.market}_{f.symbol}" for f in batch_result.failed]
    head = ", ".join(names[:_FETCH_ERROR_NOTICE_SYMBOL_LIMIT])
    if len(names) > _FETCH_ERROR_NOTICE_SYMBOL_LIMIT:
        head += " ほか"
    message = f"取得のエラーが {len(names)} 件あった（対象 {total}）: {head}"
    logger.warning("[1/6] データ取得: %s", message)
    _notify_stage_failure("[1/6] データ取得の一部の銘柄", RuntimeError(message))


def _rule_signals_mostly_unavailable(batch: "RuleSignalBatch") -> bool:
    """シグナルを出せなかった銘柄が対象の線を超えたかを返す（#822・#1191）。

    線は日次の取得（#811）と同じ DAILY_FETCH_MAX_UNAVAILABLE_RATIO を使う。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    if batch.total <= 0:
        return False
    return len(batch.unavailable) / batch.total > DAILY_FETCH_MAX_UNAVAILABLE_RATIO


def _raise_if_rule_signals_mostly_unavailable(batch: "RuleSignalBatch") -> None:
    """シグナルを出せなかった銘柄が対象の線を超えたら RuntimeError を投げる（#822）。

    データが取れない日に保有銘柄の SELL が出ないまま「完了」と記録されるのを止める。
    呼び出し元は、取れた銘柄の保有の SELL を出した後に呼ぶ（#1191）。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    unavailable = len(batch.unavailable)
    if _rule_signals_mostly_unavailable(batch):
        raise RuntimeError(
            f"ルールシグナルを出せなかった銘柄が多すぎる: {unavailable} / 対象 {batch.total}"
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）"
        )
