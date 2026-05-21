"""
バッチ実行ユーティリティ

並列処理、対象銘柄読み込み、サマリー出力など
バッチ実行スクリプト共通の処理を提供する
"""

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from typing import Callable, Optional

from src.utils.data_path_utils import get_watchlist_path
from src.utils.db import load_index_membership_symbols_as_of
from src.utils.logger import get_logger
from src.domain.types import SymbolTask
from src.watchlist.types import BatchFailure, BatchResult

logger = get_logger(__name__)

# 個別タスクのタイムアウト（秒）: yfinance API応答待ち + 特徴量生成を考慮
DEFAULT_TASK_TIMEOUT = 300


def load_target_symbols(as_of_date: Optional[str] = None) -> list[SymbolTask]:
    """
    config/watchlist.json から対象銘柄リストを読み込む

    フォーマット:
        {"us": ["AAPL", "MSFT", ...], "jp": ["7203", "9984", ...]}

    Args:
        as_of_date: YYYY-MM-DD。指定時は index_membership_history から
            指定日以前で最新の指数構成銘柄を読み込む。

    Returns:
        list[SymbolTask]: SymbolTask(市場, 銘柄コード) のリスト
    """
    if as_of_date:
        try:
            rows = load_index_membership_symbols_as_of(as_of_date)
            if rows:
                logger.info(f"過去構成銘柄を使用: as_of_date={as_of_date}, 件数={len(rows)}")
                return [SymbolTask(market=market, symbol=symbol) for market, symbol in rows]
            logger.warning(
                f"index_membership_history に as_of_date={as_of_date} の"
                "データがないため watchlist.json を使用します"
            )
        except Exception as e:
            logger.warning(
                f"index_membership_history 読み込み失敗（watchlistへフォールバック）: {e}",
                exc_info=True,
            )

    json_path = get_watchlist_path()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    symbols = []
    for market, symbol_list in data.items():
        for symbol in symbol_list:
            symbols.append(SymbolTask(market=market, symbol=symbol))
    return symbols


def _task_field(task, field: str, default: str = "?") -> str:
    val = getattr(task, field, None)
    if val is None and hasattr(task, "get"):
        val = task.get(field)
    return val if val is not None else default


def _result_status(result) -> str:
    if isinstance(result, dict):
        return result.get("status", "success")
    if hasattr(result, "status"):
        return result.status
    return "success"


def run_parallel(
    func: Callable,
    tasks: list,
    max_workers: int = 5,
    use_process: bool = False,
    label: str = "処理",
    task_timeout: int = DEFAULT_TASK_TIMEOUT,
) -> BatchResult:
    """
    タスクを並列実行する汎用ランナー

    Args:
        func: 各タスクに適用する関数。taskの要素を引数に取り、dictを返すこと。
        tasks: タスク引数のリスト
        max_workers: 並列数
        use_process: Trueの場合ProcessPoolExecutor、FalseならThreadPoolExecutor
        label: ログ表示用のラベル
        task_timeout: 個別タスクのタイムアウト（秒）。0で無制限。

    Returns:
        BatchResult: 成功・失敗・スキップを分類した集約結果
    """
    logger.info(f"{label}開始（並列数: {max_workers}） 対象件数: {len(tasks)}")

    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    succeeded: list = []
    failed: list = []
    skipped: list = []
    timeout_val = task_timeout if task_timeout > 0 else None
    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result(timeout=timeout_val)
                status = _result_status(result)
                if status == "skip":
                    skipped.append(result)
                elif status == "error":
                    _m = _task_field(result, "market")
                    _s = _task_field(result, "symbol")
                    _e = (
                        result.get("error", "unknown")
                        if isinstance(result, dict)
                        else getattr(result, "error", "unknown")
                    )
                    logger.error(f"[エラー結果] {_m}/{_s}: {_e}")
                    failed.append(BatchFailure(market=_m, symbol=_s, error=str(_e)))
                else:
                    succeeded.append(result)
            except TimeoutError:
                _m = _task_field(task, "market")
                _s = _task_field(task, "symbol")
                logger.error(f"[タイムアウト] {_m}/{_s}: {task_timeout}秒超過")
                failed.append(
                    BatchFailure(market=_m, symbol=_s, error=f"タイムアウト（{task_timeout}秒）")
                )
            except Exception as e:
                _m = _task_field(task, "market")
                _s = _task_field(task, "symbol")
                logger.error(f"[未処理エラー] {_m}/{_s}: {e}", exc_info=True)
                failed.append(BatchFailure(market=_m, symbol=_s, error=str(e)))

    return BatchResult(succeeded=succeeded, failed=failed, skipped=skipped)


def print_summary(phase: str, batch_result: BatchResult) -> None:
    """
    バッチ処理の結果サマリーを出力し、失敗があれば Discord に通知する。

    Args:
        phase: 処理フェーズ名（例: "データ取得", "モデル作成"）
        batch_result: run_parallel() の返す BatchResult
    """
    n_success = len(batch_result.succeeded)
    n_skip = len(batch_result.skipped)
    n_error = len(batch_result.failed)

    logger.info(f"{phase} 結果サマリー 成功: {n_success} / スキップ: {n_skip} / エラー: {n_error}")

    if batch_result.failed:
        logger.warning(f"\n{phase} エラー詳細:")
        for f in batch_result.failed:
            logger.warning(f"  - {f.market}/{f.symbol}: {f.error}")

        try:
            from src.reporting.discord.discord_utils import send_webhook_notification

            lines = [f"**[{phase} バッチエラー]** {n_error} 銘柄失敗"]
            for f in batch_result.failed:
                lines.append(f"• `{f.market}/{f.symbol}`: {f.error}")
            send_webhook_notification(
                title=f"{phase} バッチエラー",
                message="\n".join(lines),
                color=0xFF0000,
            )
        except Exception as e:
            logger.error("バッチエラー Discord 通知失敗: %s", e, exc_info=True)
