"""
バッチ実行ユーティリティ

並列処理、対象銘柄読み込み、サマリー出力など
バッチ実行スクリプト共通の処理を提供する
"""

import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List

from src.utils.data_path_utils import get_watchlist_path


def load_target_symbols() -> list:
    """
    CSVから対象銘柄リストを読み込む

    Returns:
        list of dict: [{"market": "us", "symbol": "AAPL"}, ...]
    """
    symbols = []
    csv_path = get_watchlist_path()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbols.append({"market": row["市場"], "symbol": row["銘柄コード"]})
    return symbols


def run_parallel(
    func: Callable,
    tasks: list,
    max_workers: int = 5,
    use_process: bool = False,
    label: str = "処理",
) -> list:
    """
    タスクを並列実行する汎用ランナー

    Args:
        func: 各タスクに適用する関数。taskの要素を引数に取り、dictを返すこと。
        tasks: タスク引数のリスト
        max_workers: 並列数
        use_process: Trueの場合ProcessPoolExecutor、FalseならThreadPoolExecutor
        label: ログ表示用のラベル

    Returns:
        list of dict: 各タスクの結果
    """
    print(f"\n{'='*50}")
    print(f"{label}開始（並列数: {max_workers}）")
    print(f"対象件数: {len(tasks)}")
    print(f"{'='*50}\n")

    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    results = []
    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


def print_summary(phase: str, results: list) -> None:
    """
    バッチ処理の結果サマリーを出力する

    Args:
        phase: 処理フェーズ名（例: "データ取得", "モデル作成"）
        results: 結果リスト（各要素は {"status": "success"|"error"|"skip", ...}）
    """
    success = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") == "error"]
    skipped = [r for r in results if r.get("status") == "skip"]

    print(f"\n{'='*50}")
    print(f"{phase} 結果サマリー")
    print(f"{'='*50}")
    print(f"成功: {len(success)}")
    print(f"スキップ: {len(skipped)}")
    print(f"エラー: {len(errors)}")

    if errors:
        print("\nエラー詳細:")
        for e in errors:
            print(f"  - {e.get('market', '?')}/{e.get('symbol', '?')}: {e.get('error', 'unknown')}")
    print()
