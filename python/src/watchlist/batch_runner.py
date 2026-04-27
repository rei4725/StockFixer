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
from src.watchlist.types import SymbolTask

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


def run_parallel(
    func: Callable,
    tasks: list,
    max_workers: int = 5,
    use_process: bool = False,
    label: str = "処理",
    task_timeout: int = DEFAULT_TASK_TIMEOUT,
) -> list:
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
        list of dict: 各タスクの結果
    """
    logger.info(f"{label}開始（並列数: {max_workers}） 対象件数: {len(tasks)}")

    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    results = []
    timeout_val = task_timeout if task_timeout > 0 else None
    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result(timeout=timeout_val)
                results.append(result)
            except TimeoutError:
                _m = getattr(task, "market", None) or task.get("market", "?")
                _s = getattr(task, "symbol", None) or task.get("symbol", "?")
                task_label = f"{_m}/{_s}"
                logger.error(f"[タイムアウト] {task_label}: {task_timeout}秒超過")
                results.append(
                    {
                        "market": getattr(task, "market", None) or task.get("market", "?"),
                        "symbol": getattr(task, "symbol", None) or task.get("symbol", "?"),
                        "status": "error",
                        "error": f"タイムアウト（{task_timeout}秒）",
                    }
                )
            except Exception as e:
                _m = getattr(task, "market", None) or task.get("market", "?")
                _s = getattr(task, "symbol", None) or task.get("symbol", "?")
                task_label = f"{_m}/{_s}"
                logger.error(f"[未処理エラー] {task_label}: {e}", exc_info=True)
                results.append(
                    {
                        "market": getattr(task, "market", None) or task.get("market", "?"),
                        "symbol": getattr(task, "symbol", None) or task.get("symbol", "?"),
                        "status": "error",
                        "error": str(e),
                    }
                )

    return results


def print_summary(phase: str, results: list) -> None:
    """
    バッチ処理の結果サマリーを出力する

    Args:
        phase: 処理フェーズ名（例: "データ取得", "モデル作成"）
        results: 結果リスト（FeatureLoadResult または dict）
    """

    def _get(obj, key, default=None):
        """dataclass と dict の両方に対応して属性を取得"""
        if hasattr(obj, key):
            return getattr(obj, key)
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return default

    success = [r for r in results if _get(r, "status") == "success"]
    errors = [r for r in results if _get(r, "status") == "error"]
    skipped = [r for r in results if _get(r, "status") == "skip"]

    logger.info(f"{phase} 結果サマリー 成功: {len(success)} / スキップ: {len(skipped)} / エラー: {len(errors)}")

    if errors:
        logger.warning(f"\n{phase} エラー詳細:")
        for e in errors:
            market = _get(e, "market", "?")
            symbol = _get(e, "symbol", "?")
            error = _get(e, "error", "unknown")
            logger.warning(f"  - {market}/{symbol}: {error}")
