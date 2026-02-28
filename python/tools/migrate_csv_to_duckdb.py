"""
CSV → DuckDB 移行スクリプト

既存の python/data/*_*/*.csv と python/results/*/ のデータを
DuckDB にインポートする。
"""

import os
import re
import glob
import pandas as pd
from datetime import datetime


def migrate_stock_features(data_dir: str) -> int:
    """
    python/data/ 配下の全 CSV を stock_features テーブルにインポートする

    Args:
        data_dir: python/data/ のパス

    Returns:
        インポートした銘柄数
    """
    from src.utils.db import upsert_stock_features

    count = 0
    pattern = os.path.join(data_dir, "*_*")
    dirs = sorted(glob.glob(pattern))

    for dir_path in dirs:
        if not os.path.isdir(dir_path):
            continue

        dir_name = os.path.basename(dir_path)
        if "_" not in dir_name:
            continue

        market, symbol = dir_name.split("_", 1)

        # CSVファイルを探す
        csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
        if not csv_files:
            continue

        csv_path = os.path.join(dir_path, csv_files[0])
        try:
            df = pd.read_csv(csv_path)

            # Date列があれば文字列のまま保持
            if "Date" in df.columns:
                pass  # そのまま保持

            # CSVにmarket, symbol列がない場合のみ追加
            if "market" not in df.columns:
                df["market"] = market
            if "symbol" not in df.columns:
                df["symbol"] = symbol

            upsert_stock_features(market, symbol, df)
            count += 1
            print(f"  [{count}] {market}_{symbol}: {len(df)}行")
        except Exception as e:
            print(f"  [エラー] {market}_{symbol}: {e}")
            continue

    return count


def migrate_prediction_results(results_dir: str) -> int:
    """
    python/results/ 配下の全タイムスタンプディレクトリの CSV を
    prediction_results テーブルにインポートする

    Args:
        results_dir: python/results/ のパス

    Returns:
        インポートしたタイムスタンプ数
    """
    from src.utils.db import save_prediction_results

    count = 0
    if not os.path.exists(results_dir):
        print(f"  結果ディレクトリが存在しません: {results_dir}")
        return 0

    subdirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and re.match(r"\d{8}_\d{6}", d)
    ])

    for ts_dir in subdirs:
        ts_path = os.path.join(results_dir, ts_dir)
        csv_files = glob.glob(os.path.join(ts_path, "*.csv"))

        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            # ファイル名からrank_typeを判定
            if "_top10_diff_stocks.csv" in fname:
                rank_type = "top10"
            elif "_worst10_diff_stocks.csv" in fname:
                rank_type = "worst10"
            else:
                rank_type = None

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                save_prediction_results(ts_dir, df, rank_type=rank_type)
            except Exception as e:
                print(f"  [エラー] {ts_dir}/{fname}: {e}")
                continue

        count += 1
        print(f"  [{count}] {ts_dir}: インポート完了")

    return count


def verify_migration(data_dir: str) -> bool:
    """
    移行後の検証: CSV の銘柄数・行数が DB と一致するか確認する

    Args:
        data_dir: python/data/ のパス

    Returns:
        検証成功なら True
    """
    from src.utils.db import get_all_symbols, load_stock_features

    # CSV側の銘柄数を数える
    pattern = os.path.join(data_dir, "*_*")
    csv_symbols = set()
    csv_total_rows = 0
    for dir_path in sorted(glob.glob(pattern)):
        if not os.path.isdir(dir_path):
            continue
        dir_name = os.path.basename(dir_path)
        if "_" not in dir_name:
            continue
        market, symbol = dir_name.split("_", 1)
        csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
        if csv_files:
            csv_symbols.add((market, symbol))
            try:
                df = pd.read_csv(os.path.join(dir_path, csv_files[0]))
                csv_total_rows += len(df)
            except Exception:
                pass

    # DB側
    db_symbols = set(get_all_symbols())
    db_total_rows = 0
    for market, symbol in db_symbols:
        df = load_stock_features(market, symbol)
        if df is not None:
            db_total_rows += len(df)

    print(f"\n=== 検証結果 ===")
    print(f"CSV 銘柄数: {len(csv_symbols)}")
    print(f"DB  銘柄数: {len(db_symbols)}")
    print(f"CSV 総行数: {csv_total_rows}")
    print(f"DB  総行数: {db_total_rows}")

    match = csv_symbols == db_symbols and csv_total_rows == db_total_rows
    if match:
        print("✓ 検証OK: CSV と DB のデータが一致しています")
    else:
        missing = csv_symbols - db_symbols
        extra = db_symbols - csv_symbols
        if missing:
            print(f"✗ DB に不足: {missing}")
        if extra:
            print(f"✗ DB に余分: {extra}")
        if csv_total_rows != db_total_rows:
            print(f"✗ 行数不一致: CSV={csv_total_rows}, DB={db_total_rows}")

    return match


def main():
    """メイン処理"""
    # パス設定
    python_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(python_root, "data")
    results_dir = os.path.join(python_root, "results")

    print("=" * 60)
    print("CSV → DuckDB 移行ツール")
    print("=" * 60)

    # DB初期化
    from src.utils.db import get_connection
    con = get_connection()
    print(f"\nDB接続完了")

    # 1. 株価特徴量データの移行
    print(f"\n--- 株価特徴量データの移行 ---")
    print(f"ソース: {data_dir}")
    n_features = migrate_stock_features(data_dir)
    print(f"→ {n_features} 銘柄をインポートしました")

    # 2. 予測結果データの移行
    print(f"\n--- 予測結果データの移行 ---")
    print(f"ソース: {results_dir}")
    n_results = migrate_prediction_results(results_dir)
    print(f"→ {n_results} 回分の結果をインポートしました")

    # 3. 検証
    print(f"\n--- 移行データの検証 ---")
    verify_migration(data_dir)

    print(f"\n移行完了!\n")


if __name__ == "__main__":
    main()
