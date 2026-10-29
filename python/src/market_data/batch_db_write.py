"""run_batch_pipeline のフェーズ2（DB書き込み・逐次）と結果のログ。

pipeline.py からの責務分離（ファイル行数ガード対策）。
保存の関数は pipeline.py から渡す（テストの patch の先 src.market_data.pipeline.<名前> を保つため）。
"""

from typing import TYPE_CHECKING, Callable

from src.domain.types import BatchFailure, BatchResult
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.domain.ports import NotificationPort

logger = get_logger(__name__)


def _raw_save_error(e: Exception) -> str:
    return f"market_data_raw の保存に失敗: {e}"


def log_pipeline_summary(phase: str, batch_result: BatchResult) -> None:
    """バッチ処理結果をログに出力する。"""
    n_success = len(batch_result.succeeded)
    n_skip = len(batch_result.skipped)
    n_error = len(batch_result.failed)
    logger.info(f"{phase} 結果サマリー 成功: {n_success} / スキップ: {n_skip} / エラー: {n_error}")
    if batch_result.failed:
        logger.warning(f"\n{phase} エラー詳細:")
        for f in batch_result.failed:
            logger.warning(f"  - {f.market}/{f.symbol}: {f.error}")


def write_batch_to_db(
    batch_result: BatchResult,
    save_raw_ohlcv: Callable,
    save_features_to_db: Callable,
    notification_port: "NotificationPort | None" = None,
    record_quality_issues: "Callable | None" = None,
) -> BatchResult:
    """フェーズ2: DB書き込み（逐次） - DuckDB排他ロック制約のため直列実行。

    market_data_raw の保存が落ちた銘柄は、特徴量の保存は続けたうえで failed に数え、
    succeeded・skipped には入れない（#1377）。採点と相関のゲートは足の表を読むため、
    [1/6] の線と知らせに乗せて、その日のうちに手で取り直せるようにする。

    特徴量をスキップした銘柄に品質の結果（段差の足など）があれば、
    record_quality_issues(market, symbol, quality_result, notification_port) で記録する（#1386）。
    """
    failed: list = []
    if record_quality_issues is not None:
        for r in batch_result.skipped:
            quality = r.get("quality_result")
            if quality is not None and quality.issues:
                record_quality_issues(r["market"], r["symbol"], quality, notification_port)

    # 特徴量をスキップした銘柄も、取れた足だけは保存する（#840）
    skipped: list = []
    skipped_raw = [r for r in batch_result.skipped if r.get("raw_data") is not None]
    if skipped_raw:
        logger.info(f"Raw保存開始（特徴量スキップ分） 対象件数: {len(skipped_raw)}")
    for r in batch_result.skipped:
        if r.get("raw_data") is None:
            skipped.append(r)
            continue
        try:
            save_raw_ohlcv(r["market"], r["symbol"], r["raw_data"])
            skipped.append(r)
        except Exception as e:
            logger.error(f"[Raw保存エラー] {r['market']}/{r['symbol']}: {e}", exc_info=True)
            failed.append(
                BatchFailure(market=r["market"], symbol=r["symbol"], error=_raw_save_error(e))
            )

    success_data = [r for r in batch_result.succeeded if r.get("data")]
    logger.info(f"DB書き込み開始（逐次） 対象件数: {len(success_data)}")

    succeeded: list = []
    for i, r in enumerate(success_data, 1):
        market, symbol, data, raw_data, quality_result = r["data"]
        raw_error = None
        try:
            if raw_data is not None and not raw_data.empty:
                try:
                    save_raw_ohlcv(market, symbol, raw_data)
                except Exception as e:
                    logger.error(f"[Raw保存エラー] {market}/{symbol}: {e}", exc_info=True)
                    raw_error = _raw_save_error(e)
            save_features_to_db(market, symbol, data, quality_result, notification_port)
            if raw_error is None:
                succeeded.append({"market": market, "symbol": symbol, "status": "success"})
            else:
                failed.append(BatchFailure(market=market, symbol=symbol, error=raw_error))
        except Exception as e:
            logger.error(f"[DB書き込みエラー] {market}/{symbol}: {e}", exc_info=True)
            error = str(e) if raw_error is None else f"{raw_error} / {e}"
            failed.append(BatchFailure(market=market, symbol=symbol, error=error))
        if i % 50 == 0:
            logger.info(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー（取得フェーズのエラー/スキップ + DB書き込み結果を統合）
    final_batch = BatchResult(
        succeeded=succeeded,
        failed=failed + batch_result.failed,
        skipped=skipped,
    )
    log_pipeline_summary("データ更新", final_batch)
    return final_batch
