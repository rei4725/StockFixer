"""
予測パイプラインサービス

銘柄別モデル・統合モデルでの全銘柄予測、Top10/Worst10集計、DB保存を行う
"""

import glob
import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models.predict_single_stock import predict_single_stock
from src.utils.data_path_utils import get_models_dir
from src.utils.db import get_all_symbols, save_prediction_results
from src.utils.df_to_string import df_to_pretty_string
from src.utils.logger import get_logger

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logger = get_logger(__name__)

# 並列実行時のワーカー数（デフォルト=1で同期実行、スレッド間競合を回避）
# ガイドライン参照: 並列処理は競合バグの原因となるため同期集計が安定
MAX_WORKERS = 1


def get_optimal_params(market: str, symbol: str) -> dict:
    """
    保存された最適パラメータをJSONから読み込む。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル

    Returns:
        最適パラメータ辞書、見つからない場合は空辞書
    """
    # python/src/services/prediction_pipeline.py -> python/config
    config_dir = Path(__file__).parents[2] / "config"
    json_path = config_dir / "optimal_params.json"

    if not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
        key = f"{market}_{symbol}"
        params = all_params.get(key, {})
        if params:
            logger.debug(
                f"[{market}_{symbol}] 最適パラメータを読み込みました: "
                f"閉値={params.get('threshold')}, "
                f"SharpeRatio={params.get('metrics', {}).get('sharpe_ratio')}"
            )
        return params
    except Exception as e:
        logger.error(f"[{market}_{symbol}] 最適パラメータ読み込みエラー: {e}")
        return {}


def find_model_files(model_root=None, model_name="StockXGBoostModel.joblib"):
    """モデルディレクトリを再帰的に探索し、(market, symbol, model_path)のリストを返す。"""
    if model_root is None:
        model_root = get_models_dir()
    pattern = os.path.join(model_root, "*_*", model_name)
    files = glob.glob(pattern)
    result = []
    for path in files:
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 3:
            market_symbol = parts[-2]
            if "_" in market_symbol:
                market, symbol = market_symbol.split("_", 1)
                result.append((market, symbol, path))
    return result


def predict_all_individual(max_workers=MAX_WORKERS):
    """
    銘柄別モデルで全銘柄の予測を実行する

    Returns:
        list[pd.DataFrame]: 各銘柄の予測結果
    """
    model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    all_keys = set()
    for model_type in model_types:
        model_files = find_model_files(model_name=model_type)
        for market, symbol, _ in model_files:
            all_keys.add((market, symbol))

    tasks = [(market, symbol, model_types) for market, symbol in all_keys]
    output_rows = []
    logger.info(f"並列予測開始（銘柄別モデル）: {len(tasks)}銘柄, ワーカー数: {max_workers}")

    def wrapper(args):
        market, symbol, mtypes = args
        try:
            return predict_single_stock(market, symbol, model_types=mtypes)
        except Exception as e:
            logger.error(f"[{market}_{symbol}] 予測エラー: {e}", exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapper, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_rows.append(result)

    return output_rows


def predict_all_unified(max_workers=MAX_WORKERS):
    """
    統合モデルで全銘柄の予測を実行する

    Returns:
        list[pd.DataFrame]: 各銘柄の予測結果
    """
    from src.models.predict_unified import predict_with_unified_model, preload_models

    model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

    # モデルを事前ロード（並列実行前に1回だけロード）
    preload_models(model_types)

    all_keys = get_all_symbols()

    def wrapper(args):
        market, symbol = args
        try:
            return predict_with_unified_model(market, symbol, model_types=model_types)
        except Exception as e:
            logger.error(f"[{market}_{symbol}] 予測エラー: {e}", exc_info=True)
            return None

    output_rows = []
    logger.info(f"並列予測開始（統合モデル）: {len(all_keys)}銘柄, ワーカー数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapper, task): task for task in all_keys}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_rows.append(result)

    return output_rows


def predict_all_unified_multi_horizon(
    horizons: list = None, max_workers: int = MAX_WORKERS
) -> list:
    """
    複数ホライズンの統合モデルで全銘柄を予測し、コンフルエンスを付与して返す。

    Args:
        horizons: 予測ホライズンのリスト（例: [1, 3, 5, 10]）。デフォルトは [1, 3, 5, 10]。
        max_workers: 並列ワーカー数

    Returns:
        list[pd.DataFrame]: 各銘柄の多ホライズン予測結果
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    from src.models.predict_unified import predict_with_unified_model, preload_models

    # 全ホライズンのモデルを事前ロード
    model_names = []
    for h in horizons:
        suffix = f"_{h}d" if h > 1 else ""
        model_names.extend([f"UnifiedStockXGBoost{suffix}", f"UnifiedStockLightGBM{suffix}"])
    preload_models(model_names)

    all_keys = get_all_symbols()
    output_rows = []
    logger.info(f"多ホライズン予測開始（統合モデル）: {len(all_keys)}銘柄, horizons={horizons}")

    for market, symbol in all_keys:
        try:
            horizon_results = {}
            for h in horizons:
                suffix = f"_{h}d" if h > 1 else ""
                mtypes = [f"UnifiedStockXGBoost{suffix}", f"UnifiedStockLightGBM{suffix}"]
                res = predict_with_unified_model(market, symbol, model_types=mtypes, horizon=h)
                if res is not None and not res.empty:
                    horizon_results[h] = res.iloc[0]

            if not horizon_results or 1 not in horizon_results:
                continue

            # horizon=1 をベースに行を構築
            base = horizon_results[1].to_dict()
            for h in horizons:
                if h == 1 or h not in horizon_results:
                    continue
                row = horizon_results[h]
                base[f"avg_pred_price_{h}d"] = row.get("avg_pred_price")
                base[f"diff_ratio_{h}d"] = row.get("diff_ratio")

            # confluence_score: 1d と同じ方向を示すホライズン数
            base_dir = 1 if base.get("diff_ratio", 0) >= 0 else -1
            conf = sum(
                1
                for h in horizons
                if h != 1
                and h in horizon_results
                and (
                    (horizon_results[h].get("diff_ratio", 0) >= 0 and base_dir > 0)
                    or (horizon_results[h].get("diff_ratio", 0) < 0 and base_dir < 0)
                )
            )
            base["confluence_score"] = conf

            output_rows.append(pd.DataFrame([base]))
        except Exception as e:
            logger.error(f"[{market}_{symbol}] 多ホライズン予測エラー: {e}", exc_info=True)

    return output_rows


def output_top_worst_results(output_rows, mode="individual"):
    """
    予測結果からTop10/Worst10を出力し、全結果をDBに保存する

    Args:
        output_rows: predict_all_*の戻り値
        mode: "individual" or "unified"
    """
    if not output_rows:
        logger.warning("有効な結果がありませんでした。")
        return

    df_result = pd.concat(output_rows, ignore_index=True)
    display_columns = [
        "market",
        "symbol",
        "current_price",
        "avg_pred_price",
        "diff_ratio",
        "model_count",
    ]
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 全予測結果をDBに保存（Delete-Insert）
    save_prediction_results(now_str, df_result[display_columns])

    for market, df_market in df_result.groupby("market"):
        # Top10
        df_top10 = df_market.sort_values("diff_ratio", ascending=False).head(10)
        df_top10_display = df_top10[display_columns]
        logger.info(
            df_to_pretty_string(
                df_top10_display,
                header=f"=== {market} 差異割合上位10銘柄 ({mode}) === 実行日時: {now_str}",
            )
        )

        # ワースト10
        df_worst10 = df_market.sort_values("diff_ratio", ascending=True).head(10)
        df_worst10_display = df_worst10[display_columns]
        logger.info(
            df_to_pretty_string(
                df_worst10_display,
                header=f"=== {market} 差異割合ワースト10銘柄 ({mode}) === 実行日時: {now_str}",
            )
        )

    logger.info(f"結果保存完了: predicted_at={now_str}")


def run_predict_single(market: str, symbol: str):
    """
    単一銘柄の予測結果を表示・DB保存する

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
    """
    import pprint

    result = predict_single_stock(market, symbol)
    if result is None:
        logger.warning("予測に失敗しました。モデルまたはデータが存在しない可能性があります。")
    else:
        pprint.pprint(result.to_dict("records")[0], sort_dicts=False, width=120)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prediction_results(now_str, result)


def run_predict_watchlist():
    """監視リストの全銘柄を予測・DB保存する。"""
    import csv

    from src.utils.data_path_utils import get_monitor_list_path

    watchlist_path = get_monitor_list_path()
    output_rows = []
    with open(watchlist_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            market, symbol = row[0], row[1]
            company = row[2] if len(row) > 2 else ""
            result = predict_single_stock(market, symbol)
            if result is None:
                logger.warning(f"{market},{symbol} ({company}) の予測に失敗しました。")
            else:
                print(f"{market},{symbol} ({company}) の予想株価:")
                import pprint

                pprint.pprint(result.to_dict("records")[0], sort_dicts=False, width=120)
                print("-" * 40)
                output_rows.append(result)

    if output_rows:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_all = pd.concat(output_rows, ignore_index=True)
        save_prediction_results(now_str, df_all)
        print(f"\n結果保存完了: run_timestamp={now_str}")


def run_accuracy_check(horizon: int = 1) -> pd.DataFrame:
    """
    過去の予測結果と実際の株価を照合し、精度を prediction_accuracy テーブルに保存する。

    予測日当日の OHLCV が取得できた銘柄のみ採点する。予測した翌営業日（horizon 日後）の
    終値が DB に存在するかを確認し、存在すれば記録する。

    Args:
        horizon: チェック対象のホライズン（デフォルト 1 = 翌日予測を評価）

    Returns:
        pd.DataFrame: 今回採点した行のサマリー
    """
    from src.utils.db import (
        load_drift_summary,
        load_prediction_results,
        load_raw_ohlcv,
        save_prediction_accuracy,
    )

    logger.info(f"=== 予測精度チェック開始 (horizon={horizon}) ===")

    # 直近 90 日分の予測結果を取得
    df_pred = load_prediction_results(limit=5000)
    if df_pred is None or df_pred.empty:
        logger.info("予測結果が存在しません。")
        return pd.DataFrame()

    new_rows: list[dict] = []

    for _, pred in df_pred.iterrows():
        market = pred.get("market")
        symbol = pred.get("symbol")
        predicted_at_str = pred.get("predicted_at", "")
        predicted_price = pred.get("avg_pred_price")
        current_price = pred.get("current_price")

        if not market or not symbol or not predicted_at_str:
            continue

        # predicted_at は YYYYMMDD_HHMMSS 形式
        try:
            pred_date = datetime.strptime(predicted_at_str[:8], "%Y%m%d")
        except ValueError:
            continue

        if predicted_price is None or current_price is None:
            continue

        # 翌 horizon 営業日後の実績終値を取得
        ohlcv = load_raw_ohlcv(market, symbol)
        if ohlcv is None or ohlcv.empty:
            continue

        ohlcv = ohlcv.sort_index()
        future_dates = ohlcv.index[ohlcv.index > pred_date]
        if len(future_dates) < horizon:
            continue  # まだデータなし

        actual_price = float(ohlcv.loc[future_dates[horizon - 1], "Close"])
        predicted_ratio = (
            (predicted_price - current_price) / current_price if current_price else None
        )
        actual_ratio = (actual_price - current_price) / current_price if current_price else None
        direction_match = (
            (predicted_ratio >= 0) == (actual_ratio >= 0)
            if predicted_ratio is not None and actual_ratio is not None
            else None
        )

        new_rows.append(
            {
                "market": market,
                "symbol": symbol,
                "model_name": "unified",
                "predicted_at": predicted_at_str,
                "horizon": horizon,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "predicted_ratio": predicted_ratio,
                "actual_ratio": actual_ratio,
                "direction_match": direction_match,
            }
        )

    if new_rows:
        save_prediction_accuracy(new_rows)

    # サマリーを返す
    summary = load_drift_summary(horizon=horizon)
    logger.info(f"精度チェック完了: {len(new_rows)}件 採点")
    return summary
