"""ジョブの「取れなかった銘柄・失敗した銘柄が多すぎる」判定。

daily.py からの責務分離（ファイル行数ガード対策）。
"""

from collections import Counter
from typing import TYPE_CHECKING, Optional

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs.common import _notify_stage_failure
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.domain.types import BatchResult
    from src.rule_engine.types import RuleSignalBatch

logger = get_logger(__name__)

# 取得のエラーの知らせに並べる銘柄の上限（#1256）
_FETCH_ERROR_NOTICE_SYMBOL_LIMIT = 5

# 止まった相場の指標の知らせに並べる「列と最後の日」の組の上限（#1634）。[1/6] の知らせは
# Discord の欄に入るので長くしない。ふつうは止まった列ごとに1組で、列は6つである
_MACRO_STALE_NOTICE_GROUP_LIMIT = 6

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
    相場の指標の列が止まった銘柄は、線の上なら止める文に原因を足し、線の下なら1回知らせる（#1634）。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    if total <= 0:
        return
    unavailable = len(batch_result.skipped) + len(batch_result.failed)
    if unavailable / total > DAILY_FETCH_MAX_UNAVAILABLE_RATIO:
        raise RuntimeError(
            f"取得できなかった銘柄が多すぎる: スキップ {len(batch_result.skipped)}"
            f" + エラー {len(batch_result.failed)} / 対象 {total}"
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）{_macro_stale_cause(batch_result)}"
        )
    _raise_if_order_market_mostly_unavailable(market_totals or {}, batch_result)
    _notify_fetch_errors(total, batch_result)
    _notify_macro_stale(total, batch_result)


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
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）{_macro_stale_cause(batch_result)}"
        )


def order_market_predictions_mostly_missing(
    market_totals: dict[str, int], output_rows: list
) -> bool:
    """発注する市場の対象のうち予想の出なかった割合が線を超えたら知らせて True を返す（#1438）。

    [3/6] は基準の値段が取れない銘柄などを黙って飛ばすため、[1/6] の線の後でも jp の
    予想が大きく欠けうる。呼び出し元は True のとき保存の印を立てず、8:50 の新規を止める。
    """
    gap = _order_market_prediction_gap(market_totals, output_rows)
    if gap is None:
        return False
    message = f"{gap}。予想は保存したが、今日の新規は止める"
    logger.warning("[3/6] 予測 (production): %s", message)
    _notify_stage_failure(
        f"[3/6] 予測 (production) の {_ORDER_MARKET} の欠け", RuntimeError(message)
    )
    return True


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


def _macro_stale_summary(batch_result: "BatchResult") -> str:
    """止まった列と最後の日と銘柄の数の文を返す。止まった銘柄が無ければ空（#1634）。"""
    skips = batch_result.macro_stale
    groups = Counter((col, day) for skip in skips for col, day in skip.columns.items())
    if not groups:
        return ""
    limit = _MACRO_STALE_NOTICE_GROUP_LIMIT
    ranked = sorted(groups.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{col}（最後の日 {day}、{n} 件）" for (col, day), n in ranked[:limit]]
    more = " ほか" if len(ranked) > limit else ""
    return (
        f"相場の指標の列が止まって特徴量を作らなかった銘柄 {len(skips)} 件: "
        f"{', '.join(parts)}{more}"
    )


def _macro_stale_cause(batch_result: "BatchResult") -> str:
    """線を超えた文の後ろに足す、原因の列の文を返す。止まった銘柄が無ければ空（#1634）。"""
    summary = _macro_stale_summary(batch_result)
    return f"。{summary}" if summary else ""


def _notify_macro_stale(total: int, batch_result: "BatchResult") -> None:
    """線の下でも列が止まった銘柄があれば、回に1通にまとめて知らせる（#1634）。

    銘柄ごとの WARNING のログにしか出ないと、古い指標で銘柄が予想から落ちた朝に
    ログを読むまで原因が分からない。
    """
    summary = _macro_stale_summary(batch_result)
    if not summary:
        return
    message = f"{summary}（対象 {total}）"
    logger.warning("[1/6] データ取得: %s", message)
    _notify_stage_failure("[1/6] データ取得の相場の指標", RuntimeError(message))


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


def raise_if_optimization_mostly_failed(total: int, failed: int) -> None:
    """週の最適化の結果が空か、失敗が線を超えたら NonRetryableJobError を投げる（#1581）。

    銘柄ごとの失敗は例外にならないため、大半の銘柄が失敗した週も success と記録され、
    同じ週の補完が止まり、発注が古い閾値と SL/TP のまま動いた。total は結果の件数で、
    skipped（その週に済んだ銘柄）も分母に残す。呼び出し元は完了の知らせの後に呼ぶ。
    知らせは済んでいるのでやり直さず、同じ週の補完に任せる。線は日次の取得（#811）と同じ。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    if total <= 0:
        raise NonRetryableJobError("週次バックテスト最適化の結果が空だった")
    if failed / total > DAILY_FETCH_MAX_UNAVAILABLE_RATIO:
        raise NonRetryableJobError(
            f"週次バックテスト最適化で失敗した銘柄が多すぎる: {failed} / 結果 {total}"
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）"
        )


def _raise_if_no_effective_rules(batch: "RuleSignalBatch") -> None:
    """対象が0銘柄の日に NonRetryableJobError を投げる（#1571）。

    有効ルールの表が空の日は保有の SELL が1件も判定されないのに、取れなかった銘柄も
    注文の失敗も0件のため成功で終わる。呼び出し元は記録と知らせの後に呼ぶ。やり直しても
    表は空のままで同じ知らせが重なるだけなので、リトライはさせない。
    """
    if batch.total > 0:
        return
    message = "有効ルールが0銘柄のため、保有の SELL も含めて判定できなかった"
    logger.error("日次ルールシグナル: %s", message)
    raise NonRetryableJobError(message)


def _order_market_prediction_gap(market_totals: dict[str, int], output_rows: list) -> Optional[str]:
    """発注する市場の予想の欠けが線を超えていれば件数の文を返す。線の内なら None（#1438）。"""
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    market = _ORDER_MARKET
    market_total = market_totals.get(market, 0)
    if market_total <= 0:
        return None
    n_predicted = sum(1 for r in output_rows if getattr(r, "market", None) == market)
    n_missing = max(market_total - n_predicted, 0)
    if n_missing / market_total <= DAILY_FETCH_MAX_UNAVAILABLE_RATIO:
        return None
    return (
        f"{market} の予想の出なかった銘柄が多すぎる: {n_missing} / {market} の対象 {market_total}"
        f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）"
    )


def _raise_if_order_market_predictions_missing(
    market_totals: dict[str, int], output_rows: list
) -> None:
    """発注する市場の予想の欠けが線を超えた回を NonRetryableJobError で失敗にする（#1595）。

    success で終わると同じ朝の補完が動かず、印が立たないまま 8:50 の新規が止まる。
    呼び出し元は後の段と知らせの後に呼ぶ。その回の中では道を頭から流し直さず、補完に任せる。
    """
    gap = _order_market_prediction_gap(market_totals, output_rows)
    if gap is not None:
        raise NonRetryableJobError(f"[3/6] 予測 (production): {gap}")
