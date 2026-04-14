"""
データパイプラインサービス

データ取得 → 特徴量生成 → DB保存 の一連の処理を統合するサービス層
data層とfeatures層を組み合わせて利用する
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.data_loader import (
    fetch_cross_asset_features,
    get_raw_ohlcv_from_db,
    get_stock_data,
    merge_market_data,
    should_fetch_fresh_data,
)
from src.data.data_saver import save_raw_ohlcv
from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
from src.utils.data_path_utils import get_ticker, normalize_col
from src.utils.db import upsert_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    defer_raw_save: bool = False,
) -> Optional[tuple]:
    """
    株価データを取得し特徴量を生成する（DB書き込みなし）。
    並列実行しても安全な純粋なデータ処理のみを行う。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)
        defer_raw_save: Trueの場合、raw OHLCVのDB保存を行わず呼び出し元に委譲する

    Returns:
        (market, symbol, data, raw_data_to_save) のタプル。失敗時はNone。
    """
    raw_data_to_save = None

    # start_date, end_date自動決定（5年前〜現在をデフォルトとする）
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    # 市場ごとにティッカーを補正
    ticker = get_ticker(market, symbol)

    # DB内のデータを確認
    df = get_raw_ohlcv_from_db(market, symbol, start_date, end_date)

    # DB内の最新日付をチェック（営業日を考慮）
    db_latest_date = None
    if df is not None and not df.empty:
        db_latest_date = pd.to_datetime(df.index.max())

    # 新規取得が必要かどうか判定（営業日考慮）
    need_fresh = should_fetch_fresh_data(db_latest_date, end_date=end_date)

    if need_fresh:
        # 新規データを取得すべき
        if db_latest_date is not None:
            # DB内有データ：差分更新ロジック
            fetch_start = (pd.to_datetime(db_latest_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"差分取得: {market}/{symbol} ({fetch_start}～{end_date})")
            fresh_data = get_stock_data(market, ticker, fetch_start, end_date)
            if fresh_data is not None and not fresh_data.empty:
                df = merge_market_data(df, fresh_data)
                logger.info(f"差分マージ完了: {len(fresh_data)}行追加")
                if defer_raw_save:
                    raw_data_to_save = fresh_data
                else:
                    try:
                        # 差分のみ保存（全件再保存より効率的）
                        save_raw_ohlcv(market, symbol, fresh_data)
                    except Exception as e:
                        logger.error(f"Raw OHLCV保存エラー（処理継続）: {e}", exc_info=True)
            else:
                logger.warning("差分データなし（営業日外の可能性）")
        else:
            # DB内無データ：フル取得
            logger.info(
                "yfinanceから取得: "
                f"market={market}, symbol={symbol}, ticker={ticker}, "
                f"{start_date}～{end_date}"
            )
            df = get_stock_data(market, ticker, start_date, end_date)
            if df is None or df.empty:
                logger.warning("データが取得できませんでした。")
                return None
            if defer_raw_save:
                raw_data_to_save = df
            else:
                try:
                    save_raw_ohlcv(market, symbol, df)
                except Exception as e:
                    logger.error(f"Raw OHLCV保存エラー（処理継続）: {e}")
    else:
        # DBデータで十分
        if df is not None and not df.empty:
            logger.debug(f"DBキャッシュで最新: market={market}, symbol={symbol} ({len(df)}行)")
        else:
            logger.warning(f"DBデータなし: market={market}, symbol={symbol}")
            return None

    # 全行NaNな列を除去（例: auto_adjust=True時の Adj Close 等）
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    if all_nan_cols:
        logger.debug(f"全NaN列を除去: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # テクニカル指標を追加
    df = add_technical_indicators(df)

    # クロスアセット特徴量を結合（R-306：VIX / USD/JPY / 米国10年債利回り）
    cross_asset = fetch_cross_asset_features(start_date, end_date)
    if cross_asset is not None and not cross_asset.empty:
        df_idx = df.index
        if isinstance(df_idx, pd.DatetimeIndex) and df_idx.tz is not None:
            df.index = df_idx.tz_localize(None)
        df = df.join(cross_asset, how="left")
        for col in cross_asset.columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        logger.debug(f"クロスアセット特徴量を付与: {list(cross_asset.columns)} ({market}/{symbol})")

    # 部分的なNaN値を前方/後方補完（OHLCV欠損日等によるdropna行数増大を防ぐ）
    nan_before = int(df.isnull().sum().sum())
    if nan_before > 0:
        df = df.ffill().bfill()
        nan_after = int(df.isnull().sum().sum())
        logger.debug(f"NaN補完: {nan_before}個→{nan_after}個 ({market}/{symbol})")

    # 行数不足チェック（TA指標の最大ウィンドウ26 + ラグ5 + ターゲット1 = 実用最低30行）
    _MIN_ROWS = 30
    if len(df) < _MIN_ROWS:
        logger.warning(f"特徴量生成をスキップ: {market}/{symbol} " f"（{len(df)}行 < 最低{_MIN_ROWS}行）")
        return None

    logger.info(f"特徴量生成（全数値列ラグ特徴量）... {market}/{symbol} ({len(df)}行)")
    X, y = create_basic_lag_features(df, n_lags=10, feature_cols=None)
    if X is None or X.empty or y is None:
        # dropna後に空になった場合の詳細診断
        nan_cols = df.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        detail = f"NaN残存列: {nan_cols.to_dict()}" if not nan_cols.empty else "NaN列なし"
        logger.warning(f"特徴量生成に失敗しました: {market}/{symbol} " f"（元データ {len(df)}行, {detail}）")
        return None

    # 特徴量名の正規化
    X.columns = [normalize_col(c) for c in X.columns]

    # X, yを1つのDataFrameにまとめる
    data = X.copy()

    # market と symbol を列として追加（統合モデル用）
    data["market"] = market
    data["symbol"] = symbol
    # market をエンコード（数値化）
    market_codes = {"us": 0, "jp": 1}
    data["market_encoded"] = market_codes.get(market, -1)

    data["y"] = y

    return (market, symbol, data, raw_data_to_save)


def save_features_to_db(market: str, symbol: str, data) -> None:
    """
    特徴量DataFrameをDBに保存する（逐次実行用）。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        data: 特徴量DataFrame
    """
    upsert_stock_features(market, symbol, data)
    logger.info(f"DB保存完了: {market}_{symbol}")


def save_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: Optional[str] = None,  # 後方互換のため残置（未使用）
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
    _, _, data, _ = result
    save_features_to_db(market, symbol, data)


def run_data_batch(fetch_only: bool = False):
    """
    ウォッチリストの全銘柄をバッチ取得する。

    fetch_only=False（デフォルト）:
      フェーズ1: データ取得＋特徴量生成（並列） - I/O boundなAPI呼び出し
      フェーズ2: DB書き込み（逐次） - DuckDBの排他ロック制約を回避

    fetch_only=True:
      フェーズ1のみ実行（DB保存しない）
    """
    from src.services.batch_runner import load_target_symbols, print_summary, run_parallel

    # バッチ取得の並列数（yfinance API制限を考慮）
    MAX_DATA_WORKERS = 3

    def _fetch_only(task) -> dict:
        """バッチランナー用: データ取得＋特徴量生成のみ（DB書き込みなし）"""
        market = getattr(task, "market", None) or task["market"]
        symbol = getattr(task, "symbol", None) or task["symbol"]
        try:
            logger.info(f"[データ取得開始] {market}/{symbol}")
            result = fetch_stock_data_with_features(
                market=market,
                symbol=symbol,
                defer_raw_save=True,
            )
            if result is None:
                logger.info(f"[データ取得スキップ] {market}/{symbol}")
                return {"market": market, "symbol": symbol, "status": "skip"}
            logger.info(f"[データ取得完了] {market}/{symbol}")
            return {"market": market, "symbol": symbol, "status": "success", "data": result}
        except Exception as e:
            logger.error(f"[データ取得エラー] {market}/{symbol}: {e}", exc_info=True)
            return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}

    symbols = load_target_symbols()
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return

    # フェーズ1: データ取得＋特徴量生成（並列）
    fetch_results = run_parallel(
        func=_fetch_only,
        tasks=symbols,
        max_workers=MAX_DATA_WORKERS,
        label="データ取得",
    )

    # fetch_only=True の場合はここで終了
    if fetch_only:
        print_summary("データ取得（保存なし）", fetch_results)
        return

    # フェーズ2: DB書き込み（逐次） - DuckDB排他ロック制約のため直列実行
    success_data = [r for r in fetch_results if r.get("status") == "success" and r.get("data")]
    logger.info(f"DB書き込み開始（逐次） 対象件数: {len(success_data)}")

    db_results = []
    for i, r in enumerate(success_data, 1):
        market, symbol, data, raw_data = r["data"]
        try:
            if raw_data is not None and not raw_data.empty:
                try:
                    save_raw_ohlcv(market, symbol, raw_data)
                except Exception as e:
                    logger.error(f"[Raw保存エラー] {market}/{symbol}: {e}")
            save_features_to_db(market, symbol, data)
            db_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            logger.error(f"[DB書き込みエラー] {market}/{symbol}: {e}", exc_info=True)
            db_results.append(
                {"market": market, "symbol": symbol, "status": "error", "error": str(e)}
            )
        if i % 50 == 0:
            logger.info(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー（取得フェーズのエラー/スキップ + DB書き込み結果を統合）
    final_results = db_results.copy()
    final_results += [r for r in fetch_results if r.get("status") in ("error", "skip")]
    print_summary("データ更新", final_results)
