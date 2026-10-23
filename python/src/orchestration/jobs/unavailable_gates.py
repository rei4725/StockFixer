"""日次ジョブの「取れなかった銘柄が多すぎる」判定。

daily.py からの責務分離（ファイル行数ガード対策）。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.types import BatchResult
    from src.rule_engine.types import RuleSignalBatch


def _raise_if_fetch_mostly_unavailable(total: int, batch_result: "BatchResult") -> None:
    """取得のスキップ+エラーが対象の線を超えたら RuntimeError を投げる（#811）。

    run_batch_pipeline は銘柄ごとのスキップ・エラーを例外にしないため、全銘柄が
    スキップされても [1/6] が「完了」になり、古い特徴量で予測を保存し続けた。
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


def _raise_if_rule_signals_mostly_unavailable(batch: "RuleSignalBatch") -> None:
    """シグナルを出せなかった銘柄が対象の線を超えたら RuntimeError を投げる（#822）。

    データが取れない日に保有銘柄の SELL が出ないまま「完了」と記録されるのを止める。
    線は日次の取得（#811）と同じ DAILY_FETCH_MAX_UNAVAILABLE_RATIO を使う。
    """
    from config.settings import DAILY_FETCH_MAX_UNAVAILABLE_RATIO

    if batch.total <= 0:
        return
    unavailable = len(batch.unavailable)
    if unavailable / batch.total > DAILY_FETCH_MAX_UNAVAILABLE_RATIO:
        raise RuntimeError(
            f"ルールシグナルを出せなかった銘柄が多すぎる: {unavailable} / 対象 {batch.total}"
            f"（線 {DAILY_FETCH_MAX_UNAVAILABLE_RATIO:.0%}）"
        )
