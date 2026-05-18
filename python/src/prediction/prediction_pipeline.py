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

from config.settings import PREDICTION_MAX_WORKERS
from src.prediction.predict_single import predict_single_stock
from src.prediction.types import PredictionResult
from src.utils.data_path_utils import get_models_dir
from src.utils.db import get_all_symbols, save_prediction_results
from src.utils.df_to_string import df_to_pretty_string
from src.utils.logger import get_logger
from src.utils.run_context import new_run_context

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logger = get_logger(__name__)

# 並列実行ワーカー数: 環境変数 PREDICTION_MAX_WORKERS で上書き可能（デフォルト: CPU数の半分）
# 予測フェーズは読み取り専用のため並列実行しても DuckDB 排他ロックは不要
MAX_WORKERS = PREDICTION_MAX_WORKERS


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
        logger.error(f"[{market}_{symbol}] 最適パラメータ読み込みエラー: {e}", exc_info=True)
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
        list[PredictionResult]: 各銘柄の予測結果
    """
    model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    all_keys = set()
    for model_type in model_types:
        model_files = find_model_files(model_name=model_type)
        for market, symbol, _ in model_files:
            all_keys.add((market, symbol))

    tasks = [(market, symbol, model_types) for market, symbol in all_keys]
    output_rows: list[PredictionResult] = []
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
        list[PredictionResult]: 各銘柄の予測結果
    """
    from src.prediction.predict_unified import predict_with_unified_model, preload_models

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

    output_rows: list[PredictionResult] = []
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
) -> list[PredictionResult]:
    """
    複数ホライズンの統合モデルで全銘柄を予測し、コンフルエンスを付与して返す。

    Args:
        horizons: 予測ホライズンのリスト（例: [1, 3, 5, 10]）。デフォルトは [1, 3, 5, 10]。
        max_workers: 並列ワーカー数

    Returns:
        list[PredictionResult]: 各銘柄の多ホライズン予測結果
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    from src.prediction.predict_unified import predict_with_unified_model, preload_models

    # 全ホライズンのモデルを事前ロード
    model_names = []
    for h in horizons:
        suffix = f"_{h}d" if h > 1 else ""
        model_names.extend([f"UnifiedStockXGBoost{suffix}", f"UnifiedStockLightGBM{suffix}"])
    preload_models(model_names)

    all_keys = get_all_symbols()
    output_rows: list[PredictionResult] = []
    logger.info(f"多ホライズン予測開始（統合モデル）: {len(all_keys)}銘柄, horizons={horizons}")

    for market, symbol in all_keys:
        try:
            horizon_results: dict[int, PredictionResult] = {}
            for h in horizons:
                suffix = f"_{h}d" if h > 1 else ""
                mtypes = [
                    f"UnifiedStockXGBoost{suffix}",
                    f"UnifiedStockLightGBM{suffix}",
                ]
                res = predict_with_unified_model(market, symbol, model_types=mtypes, horizon=h)
                if res is not None:
                    horizon_results[h] = res

            if not horizon_results or 1 not in horizon_results:
                continue

            base = horizon_results[1]
            base_dir = base.diff_ratio >= 0
            conf = sum(
                1
                for h in horizons
                if h != 1
                and h in horizon_results
                and (horizon_results[h].diff_ratio >= 0) == base_dir
            )

            output_rows.append(
                PredictionResult(
                    market=base.market,
                    symbol=base.symbol,
                    current_price=base.current_price,
                    avg_pred_price=base.avg_pred_price,
                    diff_ratio=base.diff_ratio,
                    model_count=base.model_count,
                    avg_pred_price_3d=(
                        horizon_results[3].avg_pred_price if 3 in horizon_results else None
                    ),
                    avg_pred_price_5d=(
                        horizon_results[5].avg_pred_price if 5 in horizon_results else None
                    ),
                    avg_pred_price_10d=(
                        horizon_results[10].avg_pred_price if 10 in horizon_results else None
                    ),
                    diff_ratio_3d=(horizon_results[3].diff_ratio if 3 in horizon_results else None),
                    diff_ratio_5d=(horizon_results[5].diff_ratio if 5 in horizon_results else None),
                    diff_ratio_10d=(
                        horizon_results[10].diff_ratio if 10 in horizon_results else None
                    ),
                    confluence_score=conf,
                )
            )
        except Exception as e:
            logger.error(f"[{market}_{symbol}] 多ホライズン予測エラー: {e}", exc_info=True)

    return output_rows


def output_top_worst_results(
    output_rows: list[PredictionResult],
    mode="individual",
    shadow_mode: bool = False,
    model_version: str = "production",
):
    """
    予測結果からTop10/Worst10を出力し、全結果をDBに保存する

    シャドーモード時は model_version を各 PredictionResult に付与して保存する。
    production / challenger 両バージョンを同一テーブルに共存させることで、
    一定期間後の定量評価（Sharpe / Hit Rate）に使用する。

    Args:
        output_rows: predict_all_*の戻り値（list[PredictionResult]）
        mode: "individual" or "unified"
        shadow_mode: True のとき model_version を付与する
        model_version: モデルバージョンラベル（デフォルト "production"）
    """
    if not output_rows:
        logger.warning("有効な結果がありませんでした。")
        return

    run_ctx = new_run_context()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"予測パイプライン開始: run_id={run_ctx.run_id}")

    if shadow_mode:
        # model_version を付与（未設定の行のみ上書きする）
        tagged_rows = []
        for r in output_rows:
            if r.model_version is None:
                from dataclasses import replace

                tagged_rows.append(replace(r, model_version=model_version))
            else:
                tagged_rows.append(r)
        save_rows = tagged_rows
        logger.info(f"シャドーモード予測実行: model_version={model_version}, {len(save_rows)}銘柄")
    else:
        save_rows = output_rows

    # 全予測結果をDBに保存（Delete-Insert）
    save_prediction_results(now_str, save_rows)

    # 表示用 DataFrame変換（ログ出力用）
    df_result = PredictionResult.to_dataframe(output_rows)
    display_columns = [
        "market",
        "symbol",
        "current_price",
        "avg_pred_price",
        "diff_ratio",
        "model_count",
    ]

    for market, df_market in df_result.groupby("market"):
        df_top10 = df_market.sort_values("diff_ratio", ascending=False).head(10)
        logger.info(
            df_to_pretty_string(
                df_top10[display_columns],
                header=f"=== {market} 差異割合上位10銘柄 ({mode}) === 実行日時: {now_str}",
            )
        )
        df_worst10 = df_market.sort_values("diff_ratio", ascending=True).head(10)
        logger.info(
            df_to_pretty_string(
                df_worst10[display_columns],
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
        pprint.pprint(vars(result), sort_dicts=False, width=120)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prediction_results(now_str, [result])


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

                pprint.pprint(vars(result), sort_dicts=False, width=120)
                print("-" * 40)
                output_rows.append(result)

    if output_rows:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prediction_results(now_str, output_rows)
        print(f"\n結果保存完了: run_timestamp={now_str}")


def run_accuracy_check(
    horizon: int = 1,
    model_name: str = "unified",
    model_version_filter: str = None,
) -> pd.DataFrame:
    """
    過去の予測結果と実際の株価を照合し、精度を prediction_accuracy テーブルに保存する。

    予測日当日の OHLCV が取得できた銘柄のみ採点する。予測した翌営業日（horizon 日後）の
    終値が DB に存在するかを確認し、存在すれば記録する。

    Args:
        horizon: チェック対象のホライズン（デフォルト 1 = 翌日予測を評価）
        model_name: prediction_accuracy テーブルに記録するモデル名ラベル
        model_version_filter: prediction_results の model_version でフィルタ（None なら全件）

    Returns:
        pd.DataFrame: 今回採点した行のサマリー
    """
    from src.utils.db import (
        load_drift_summary,
        load_prediction_results,
        load_raw_ohlcv,
        save_prediction_accuracy,
    )

    logger.info(
        f"=== 予測精度チェック開始 (horizon={horizon}, model_name={model_name}, "
        f"model_version_filter={model_version_filter}) ==="
    )

    # 直近の予測結果を取得（model_version フィルタ付き）
    df_pred = load_prediction_results(limit=5000, model_version=model_version_filter)
    if df_pred is None or df_pred.empty:
        logger.info("予測結果が存在しません（model_version=%s）。", model_version_filter)
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
                "model_name": model_name,
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
