"""
予測パイプラインサービス

銘柄別モデル・統合モデルでの全銘柄予測、Top10/Worst10集計、DB保存を行う
"""

import glob
import json
import logging
import os
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config.data import MAX_PREDICTION_WORKERS
from src.prediction.db import save_prediction_results
from src.prediction.multi_horizon import predict_all_unified_multi_horizon  # noqa: F401
from src.prediction.predict_single import predict_single_stock
from src.prediction.prediction_run_stats import collect_prediction_run_stats
from src.prediction.types import PredictionResult
from src.prediction.unified_run_summary import log_unified_prediction_summary
from src.utils.data_path_utils import get_models_dir
from src.utils.db import get_active_symbols
from src.utils.df_to_string import df_to_pretty_string
from src.utils.logger import get_logger
from src.utils.run_context import new_run_context

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logger = get_logger(__name__)

MAX_WORKERS = MAX_PREDICTION_WORKERS
# 精度チェックで「まだデータなし」の行をログに出す件数の上限
_NO_FUTURE_SAMPLE_LIMIT = 5
# 保存した現在値と raw の基準の終値の比がこの幅に収まれば、分割による調整の違いは無いとみなす
_SPLIT_GAP_TOLERANCE = 1.3
# 比を分割の比（0.5 刻み）に丸めたときに許す、丸めとの相対の差
_SPLIT_RATIO_TOLERANCE = 0.1
# 市場ごとの (タイムゾーン, 引けの時刻)。jp は東証 15:30 JST、us は米国東部 16:00
_MARKET_CLOSE = {
    "jp": (ZoneInfo("Asia/Tokyo"), time(15, 30)),
    "us": (ZoneInfo("America/New_York"), time(16, 0)),
}


def _split_adjust_scale(current_price: float, base_close: float) -> float | None:
    """
    保存した現在値を raw の足の調整の水準に合わせる倍率を返す。

    比が _SPLIT_GAP_TOLERANCE の内なら分割は無いとみなして 1.0 を返す。
    外なら 0.5 刻みの分割の比に丸め、丸めとの差が小さいときだけその倍率を返し、
    分割の比に合わないときは None を返す。
    """
    if base_close <= 0 or current_price <= 0:
        return None
    gap = base_close / current_price
    ratio = max(gap, 1 / gap)
    if ratio <= _SPLIT_GAP_TOLERANCE:
        return 1.0
    split_ratio = round(ratio * 2) / 2
    if abs(ratio / split_ratio - 1) > _SPLIT_RATIO_TOLERANCE:
        return None
    return split_ratio if gap > 1 else 1 / split_ratio


def _base_bar_date(predicted_at_str: str, market: str) -> datetime:
    """
    予想の時刻（UTC の YYYYMMDD_HHMMSS）に引けていた最後の足の日を返す。

    市場の現地の時刻が引けより前なら、現地の前の日を返す。休日はその日以前の
    足を採るので、そのまま返してよい。時刻が読めないときや市場が不明なときは、
    UTC の日付を返す。

    Raises:
        ValueError: 先頭8字が日付として読めないとき
    """
    pred_date = datetime.strptime(predicted_at_str[:8], "%Y%m%d")
    if market not in _MARKET_CLOSE:
        return pred_date
    try:
        predicted_at = datetime.strptime(predicted_at_str[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return pred_date
    tz, close_time = _MARKET_CLOSE[market]
    local = predicted_at.replace(tzinfo=timezone.utc).astimezone(tz)
    base_day = local.date() if local.time() >= close_time else local.date() - timedelta(days=1)
    return datetime.combine(base_day, time())


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


def predict_all_unified(
    max_workers=MAX_WORKERS, on_error: Callable[[str, str], None] | None = None
):
    """
    統合モデルで全銘柄の予測を実行する

    Args:
        on_error: 例外で予想を出せなかった銘柄ごとに (market, symbol) で呼ぶ（#1118）

    Returns:
        list[PredictionResult]: 各銘柄の予測結果
    """
    from src.prediction.predict_unified import predict_with_unified_model, preload_models
    from src.prediction.types import UNIFIED_PREDICTION_MODEL_NAMES

    # daily.py の requested_models と同じ定数を参照する（I-4 対策）。
    model_types = list(UNIFIED_PREDICTION_MODEL_NAMES)

    # モデルを事前ロード（並列実行前に1回だけロード）
    preload_models(model_types)

    all_keys = get_active_symbols()
    # 特徴量が古いため予想を出さなかった銘柄（list.append はスレッドをまたいでも安全）
    stale_keys: list[tuple[str, str]] = []

    def on_stale(market: str, symbol: str) -> None:
        stale_keys.append((market, symbol))

    def wrapper(args):
        market, symbol = args
        try:
            return predict_with_unified_model(
                market, symbol, model_types=model_types, on_stale=on_stale
            )
        except Exception as e:
            logger.error(f"[{market}_{symbol}] 予測エラー: {e}", exc_info=True)
            error_keys.append((market, symbol))
            return None

    error_keys: list[tuple[str, str]] = []
    output_rows: list[PredictionResult] = []
    logger.info(f"並列予測開始（統合モデル）: {len(all_keys)}銘柄, ワーカー数: {max_workers}")

    # クリップと0埋めの数を回の終わりにログに出す（#1318）
    with collect_prediction_run_stats("unified"):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(wrapper, task): task for task in all_keys}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    output_rows.append(result)

    log_unified_prediction_summary(len(output_rows), len(all_keys), stale_keys, error_keys)
    if on_error is not None:
        for market, symbol in error_keys:
            on_error(market, symbol)
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
    skip_predicted_at: Callable[[str], bool] | None = None,
) -> pd.DataFrame:
    """
    過去の予測結果と実際の株価を照合し、精度を prediction_accuracy テーブルに保存する。

    予測日当日の OHLCV が取得できた銘柄のみ採点する。予測した翌営業日（horizon 日後）の
    終値が DB に存在するかを確認し、存在すれば記録する。

    Args:
        horizon: チェック対象のホライズン（デフォルト 1 = 翌日予測を評価）
        model_name: prediction_accuracy テーブルに記録するモデル名ラベル
        model_version_filter: prediction_results の model_version でフィルタ（None なら全件）
        skip_predicted_at: predicted_at を受けて True を返す行は採点しない（None なら外さない）。
            日次の道のやり直しで今日の回の行だけを外すのに使う（#1255）

    Returns:
        pd.DataFrame: 今回採点した行のサマリー
    """
    from src.prediction.db import (
        load_drift_summary,
        load_prediction_results,
        save_prediction_accuracy,
    )
    from src.utils.db import load_raw_ohlcv

    logger.info(
        f"=== 予測精度チェック開始 (horizon={horizon}, model_name={model_name}, "
        f"model_version_filter={model_version_filter}) ==="
    )

    # 直近の予測結果を取得（model_version フィルタ付き）。読み込みの失敗は行が無いことと分け、
    # 例外として日次の道の [2/6] の知らせに乗せる（#1282）
    df_pred = load_prediction_results(
        limit=5000, model_version=model_version_filter, raise_on_error=True
    )
    if df_pred is None or df_pred.empty:
        logger.info("予測結果が存在しません（model_version=%s）。", model_version_filter)
        return pd.DataFrame()

    new_rows: list[dict] = []
    # 採点を飛ばした件数を continue の分かれ道ごとに数え、完了ログに出す（0件の原因の切り分け用）
    skipped = {
        "必須の値なし": 0,
        "外した回": 0,
        "predicted_at形式不正": 0,
        "予想値段なし": 0,
        "足なし": 0,
        "まだデータなし": 0,
        "分割の比が合わない": 0,
    }
    # まだデータなしで飛ばした行の値を数件だけ残し、日付の比べが合わない原因の切り分けに使う
    no_future_samples: list[str] = []

    for _, pred in df_pred.iterrows():
        market = pred.get("market")
        symbol = pred.get("symbol")
        predicted_at_str = pred.get("predicted_at", "")
        predicted_price = pred.get("avg_pred_price")
        current_price = pred.get("current_price")

        # 文字列の列でも NULL は NaN になり、NaN は真と判定されるため pd.isna でも確かめる
        if any(pd.isna(v) or not v for v in (market, symbol, predicted_at_str)):
            skipped["必須の値なし"] += 1
            continue

        if skip_predicted_at is not None and skip_predicted_at(str(predicted_at_str)):
            skipped["外した回"] += 1
            continue

        # predicted_at は YYYYMMDD_HHMMSS 形式（UTC）。基準の日は予想の時刻に引けていた最後の足の日
        try:
            pred_date = _base_bar_date(predicted_at_str, market)
        except ValueError:
            skipped["predicted_at形式不正"] += 1
            continue

        # DB の NULL は DataFrame では NaN になるため、None と同じく欠けとして扱う
        if pd.isna(predicted_price) or pd.isna(current_price):
            skipped["予想値段なし"] += 1
            continue

        # 翌 horizon 営業日後の実績終値を取得
        ohlcv = load_raw_ohlcv(market, symbol)
        if ohlcv is None or ohlcv.empty:
            skipped["足なし"] += 1
            continue

        ohlcv = ohlcv.sort_index()
        future_dates = ohlcv.index[ohlcv.index > pred_date]
        if len(future_dates) < horizon:
            skipped["まだデータなし"] += 1
            if len(no_future_samples) < _NO_FUTURE_SAMPLE_LIMIT:
                no_future_samples.append(
                    f"{market}/{symbol} predicted_at={predicted_at_str} "
                    f"pred_date={pred_date.date()} 足の最後={ohlcv.index[-1]} "
                    f"足の日付の型={ohlcv.index.dtype}"
                )
            continue

        # raw の足は分割で調整し直されることがあるため、予想の日以前の最後の足の終値と
        # 保存した現在値の比から分割を見つけ、現在値と予想の値段を足と同じ水準に直す。
        # 予想の日以前の足が無いときは確かめられないので、保存した値のまま比べる
        base_dates = ohlcv.index[ohlcv.index <= pred_date]
        if len(base_dates) > 0:
            base_close = float(ohlcv.loc[base_dates[-1], "Close"])
            scale = _split_adjust_scale(float(current_price), base_close)
            if scale is None:
                skipped["分割の比が合わない"] += 1
                continue
            if scale != 1.0:
                current_price = current_price * scale
                predicted_price = predicted_price * scale

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

    if no_future_samples:
        logger.info("まだデータなしの例: %s", " | ".join(no_future_samples))

    # 書き込みに落ちた行があれば例外が上がり、日次の [2/6] の失敗の知らせに届く（#1371）
    saved = save_prediction_accuracy(new_rows) if new_rows else 0

    # サマリーを返す
    summary = load_drift_summary(horizon=horizon, model_name=model_name)
    skipped_str = ", ".join(f"{reason}={count}" for reason, count in skipped.items())
    logger.info(
        f"精度チェック完了: {len(new_rows)}件 採点, {saved}件 保存 "
        f"(読み込み={len(df_pred)}件, 飛ばした理由: {skipped_str})"
    )
    return summary
