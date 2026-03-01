"""
データパイプラインサービス

データ取得 → 特徴量生成 → DB保存 の一連の処理を統合するサービス層
data層とfeatures層を組み合わせて利用する
"""

import re
from datetime import datetime
from typing import Optional

from src.data.data_loader import get_stock_data, get_raw_ohlcv_from_db
from src.data.data_saver import save_raw_stock_data, save_raw_ohlcv
from src.features.technical_analysis import create_basic_lag_features, add_technical_indicators
from src.utils.db import upsert_stock_features, delete_stock_features
from src.utils.data_path_utils import get_ticker


def fetch_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[tuple]:
    """
    株価データを取得し特徴量を生成する（DB書き込みなし）。
    並列実行しても安全な純粋なデータ処理のみを行う。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)

    Returns:
        (market, symbol, data) のタプル。失敗時はNone。
    """
    # start_date, end_date自動決定
    if start_date is None or end_date is None:
        try:
            df_all = get_stock_data(market, symbol, "1900-01-01", datetime.now().strftime("%Y-%m-%d"))
            if df_all is None or df_all.empty:
                print(f"{symbol} のデータが取得できませんでした。")
                return None
            start_date = df_all.index.min().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"{symbol} のデータ取得でエラー: {e}")
            return None
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 市場ごとにティッカーを補正
    ticker = get_ticker(market, symbol)

    # まずDBのキャッシュから取得を試みる（yfinance再取得を回避）
    df = get_raw_ohlcv_from_db(market, symbol, start_date, end_date)
    if df is not None and not df.empty:
        print(f"DBキャッシュから取得: market={market}, symbol={symbol} ({len(df)}行)")
    else:
        print(f"yfinanceから取得: market={market}, symbol={symbol}, ticker={ticker}, {start_date}～{end_date}")
        df = get_stock_data(market, ticker, start_date, end_date)
        if df is None or df.empty:
            print("データが取得できませんでした。")
            return None
        # 生データをDBに保存（次回以降はキャッシュを使用）
        try:
            save_raw_ohlcv(market, symbol, df)
        except Exception as e:
            print(f"Raw OHLCV保存エラー（処理継続）: {e}")

    # テクニカル指標を追加
    df = add_technical_indicators(df)

    print("特徴量生成（全数値列ラグ特徴量）...")
    X, y = create_basic_lag_features(df, n_lags=5, feature_cols=None)
    if X is None or X.empty or y is None:
        print("特徴量生成に失敗しました。")
        return None

    # 特徴量名の正規化
    def normalize_col(col):
        return re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
    X.columns = [normalize_col(c) for c in X.columns]

    # X, yを1つのDataFrameにまとめる
    data = X.copy()

    # market と symbol を列として追加（統合モデル用）
    data['market'] = market
    data['symbol'] = symbol
    # market をエンコード（数値化）
    market_codes = {"us": 0, "jp": 1}
    data['market_encoded'] = market_codes.get(market, -1)

    data['y'] = y

    return (market, symbol, data)


def save_features_to_db(market: str, symbol: str, data) -> None:
    """
    特徴量DataFrameをDBに保存する（逐次実行用）。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        data: 特徴量DataFrame
    """
    delete_stock_features(market, symbol)
    upsert_stock_features(market, symbol, data)
    print(f"DB保存完了: {market}_{symbol}")


def save_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = None  # 後方互換のため残置（未使用）
):
    """
    指定した市場・シンボル・期間の株価データを取得し、特徴量生成後、DBに保存する。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)
        out_dir: 後方互換のため残置（未使用）
    """
    result = fetch_stock_data_with_features(market, symbol, start_date, end_date)
    if result is None:
        return
    _, _, data = result
    save_features_to_db(market, symbol, data)


def run_data_batch():
    """
    ウォッチリストの全銘柄をバッチ取得する。
    
    フェーズ1: データ取得＋特徴量生成（並列） - I/O boundなAPI呼び出し
    フェーズ2: DB書き込み（逐次） - DuckDBの排他ロック制約を回避
    """
    from src.services.batch_runner import load_target_symbols, run_parallel, print_summary

    # バッチ取得の並列数（yfinance API制限を考慮）
    MAX_DATA_WORKERS = 5

    def _fetch_only(task: dict) -> dict:
        """バッチランナー用: データ取得＋特徴量生成のみ（DB書き込みなし）"""
        market, symbol = task["market"], task["symbol"]
        try:
            print(f"[データ取得開始] {market}/{symbol}")
            result = fetch_stock_data_with_features(market=market, symbol=symbol)
            if result is None:
                print(f"[データ取得スキップ] {market}/{symbol}")
                return {"market": market, "symbol": symbol, "status": "skip"}
            print(f"[データ取得完了] {market}/{symbol}")
            return {"market": market, "symbol": symbol, "status": "success", "data": result}
        except Exception as e:
            print(f"[データ取得エラー] {market}/{symbol}: {e}")
            return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}

    symbols = load_target_symbols()
    if not symbols:
        print("対象銘柄がありません。")
        return

    # フェーズ1: データ取得＋特徴量生成（並列）
    fetch_results = run_parallel(
        func=_fetch_only,
        tasks=symbols,
        max_workers=MAX_DATA_WORKERS,
        label="データ取得",
    )

    # フェーズ2: DB書き込み（逐次） - DuckDB排他ロック制約のため直列実行
    success_data = [r for r in fetch_results if r.get("status") == "success" and r.get("data")]
    print(f"\n{'='*50}")
    print(f"DB書き込み開始（逐次）")
    print(f"対象件数: {len(success_data)}")
    print(f"{'='*50}\n")

    db_results = []
    for i, r in enumerate(success_data, 1):
        market, symbol, data = r["data"]
        try:
            save_features_to_db(market, symbol, data)
            db_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            print(f"[DB書き込みエラー] {market}/{symbol}: {e}")
            db_results.append({"market": market, "symbol": symbol, "status": "error", "error": str(e)})
        if i % 50 == 0:
            print(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー（取得フェーズのエラー/スキップ + DB書き込み結果を統合）
    final_results = db_results.copy()
    final_results += [r for r in fetch_results if r.get("status") in ("error", "skip")]
    print_summary("データ更新", final_results)
