"""
統合モデル用予測処理

統合モデルを使用して全銘柄の予測を行う
"""

import logging
import os
import warnings
from datetime import date, datetime, timedelta
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from config.settings import STALE_BARS_MAX_WEEKDAYS
from src.domain.jp_market_holidays import is_jp_business_day
from src.prediction.db import load_model_weights
from src.prediction.prediction_run_stats import (
    collect_prediction_run_stats,
    record_clip,
    record_zero_fill,
)
from src.prediction.remote_client import get_service_url, predict_via_service
from src.prediction.types import PredictionResult
from src.prediction.unified_ensemble import (  # noqa: F401 (re-export)
    _run_ensemble,
    ensemble_diff_ratio,
    feature_row,
    resolve_expected_features,
)
from src.utils.data_path_utils import get_models_dir, get_ticker
from src.utils.db import get_active_symbols, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# モデルキャッシュ（スレッドセーフ）。モデル名 → (ファイルの更新時刻, モデル)
_model_cache: Dict[str, Tuple[Optional[float], Any]] = {}
_model_cache_lock = Lock()


def load_feature_data(market: str, symbol: str) -> Optional[pd.DataFrame]:
    """
    特徴量データをDBから読み込む

    Args:
        market: 市場名
        symbol: 銘柄コード

    Returns:
        特徴量DataFrame。行が無ければ None

    Raises:
        Exception: 読み込みの失敗（DB の例外）。行が無いことと分け、呼び出し元の
            例外の数と知らせに乗せる（#1272）
    """
    return load_stock_features(market, symbol, raise_on_error=True)


def _unified_model_mtime(model_name: str) -> Optional[float]:
    """統合モデルのファイルの更新時刻。ファイルが無ければ None。"""
    path = os.path.join(get_models_dir(), "unified", f"{model_name}.joblib")
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def get_cached_model(model_name: str):
    """
    キャッシュからモデルを取得（なければロード）
    スレッドセーフな実装

    常駐のスケジューラーは同じプロセスで週の学習（昇格・challenger の学習し直し）と
    日次の予想を回すため、キャッシュはファイルの更新時刻を一緒に持ち、変わっていたら
    読み直す。見つからなかった None はキャッシュせず、次の呼び出しで再びファイルを見る。
    """
    from src.prediction.unified_model_pipeline import load_unified_model

    with _model_cache_lock:
        # 読み込みの前に時刻を取る。読み込み中に置き替わっても次の呼び出しで読み直される
        mtime = _unified_model_mtime(model_name)
        cached = _model_cache.get(model_name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            model = load_unified_model(model_name)
        except FileNotFoundError:
            _model_cache.pop(model_name, None)
            return None
        _model_cache[model_name] = (mtime, model)
        return model


def preload_models(model_types: List[str] = None) -> List[str]:
    """
    モデルを事前にロードしてキャッシュする
    並列処理の前に呼び出すことで、スレッド間でモデルを共有できる

    Returns:
        ロードに成功したモデル名のリスト（要求順）。
        出力 invariant の A-1 / A-2 が期待値としてこれを使う。
    """
    if model_types is None:
        from src.prediction.types import UNIFIED_PREDICTION_MODEL_NAMES

        model_types = list(UNIFIED_PREDICTION_MODEL_NAMES)

    logger.info("モデルを事前ロード中: %s", model_types)
    loaded: List[str] = []
    for model_name in model_types:
        model = get_cached_model(model_name)
        if model is not None:
            logger.info("  - %s: ロード完了", model_name)
            loaded.append(model_name)
        else:
            logger.info("  - %s: 見つかりません", model_name)
    logger.info("モデルの事前ロード完了: %d/%d", len(loaded), len(model_types))
    return loaded


def _is_open_day(day: date, market: Optional[str]) -> bool:
    """jp は東証の営業日（#1106）、それ以外は土日以外なら True。"""
    if market == "jp":
        return is_jp_business_day(day)
    return day.weekday() < 5


def count_weekdays_behind(
    last_date: pd.Timestamp, today: Optional[datetime] = None, market: Optional[str] = None
) -> int:
    """
    特徴量の最後の行の翌日から、あるべき最新の営業日までの営業日の数を返す。

    数え方は market_data.pipeline の同名の関数（#815）に揃える。BC 同士は
    import できないため写している。

    Args:
        last_date: 特徴量の最後の行の日付
        today: 基準日（省略時は今日）
        market: 市場名。jp は東証の休場日も除く（#1106）。省略時は土日だけを除く

    Returns:
        足りない営業日の数。
    """
    expected_latest = pd.Timestamp(today if today is not None else datetime.now()).normalize()
    while not _is_open_day(expected_latest.date(), market):
        expected_latest -= timedelta(days=1)
    if last_date.tzinfo is not None:
        last_date = last_date.tz_localize(None)
    next_day = last_date.normalize() + timedelta(days=1)
    days = pd.date_range(next_day, expected_latest)
    return sum(1 for day in days if _is_open_day(day.date(), market))


def _last_feature_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """特徴量の最後の行の日付。date 列が無いか読めなければ None。"""
    if "date" not in df.columns:
        return None
    last_date = pd.to_datetime(df["date"].iloc[-1], errors="coerce")
    if pd.isna(last_date):
        return None
    return pd.Timestamp(last_date)


def _last_bar_date(hist: pd.DataFrame) -> Optional[pd.Timestamp]:
    """yfinance の最後の足の取引所の日付。index が日時でなければ None。

    yfinance の index は取引所のタイムゾーンの 0 時である。stock_features の
    date 列も取引所の日付をタイムゾーン無しの 0 時で持つ（yf_client が
    tz_localize(None) で落とす）ため、UTC へは変換せずタイムゾーンだけを落とす。
    """
    if not isinstance(hist.index, pd.DatetimeIndex) or len(hist.index) == 0:
        return None
    bar_date = pd.Timestamp(hist.index[-1])
    if bar_date.tzinfo is not None:
        bar_date = bar_date.tz_localize(None)
    return bar_date.normalize()


def _fetch_current_price(
    market: str, symbol: str, feature_date: Optional[pd.Timestamp]
) -> Optional[float]:
    """
    基準の値段（特徴量の最後の行の日の終値）を yfinance から取る。

    取れない・最後の足が特徴量の最後の行と別の日のときは None を返す。
    Close_lag1 は前の営業日の終値、y は変化率で、どちらも基準にならない（#845）。

    Args:
        market: 市場名
        symbol: 銘柄コード
        feature_date: 特徴量の最後の行の日付（不明なら None で、日付は比べない）

    Returns:
        終値。使えなければ None。
    """
    try:
        yf_ticker = get_ticker(market, symbol)
        hist = yf.Ticker(yf_ticker).history(period="1d")
    except Exception:
        logger.warning(
            "現在価格取得失敗のため予想をスキップ: market=%s symbol=%s",
            market,
            symbol,
            exc_info=True,
        )
        return None

    if hist is None or hist.empty or "Close" not in hist.columns:
        logger.warning("現在価格が空のため予想をスキップ: market=%s symbol=%s", market, symbol)
        return None

    bar_date = _last_bar_date(hist)
    if feature_date is not None and bar_date is not None:
        if feature_date.tzinfo is not None:
            feature_date = feature_date.tz_localize(None)
        feature_day = feature_date.normalize()
        if bar_date != feature_day:
            logger.warning(
                "現在価格の足の日付が特徴量と合わないため予想をスキップ: market=%s symbol=%s "
                "足の日付=%s 特徴量の最後の行の日付=%s",
                market,
                symbol,
                bar_date.strftime("%Y-%m-%d"),
                feature_day.strftime("%Y-%m-%d"),
            )
            return None

    close = hist["Close"].iloc[-1]
    if pd.isna(close):
        logger.warning("現在価格が欠損のため予想をスキップ: market=%s symbol=%s", market, symbol)
        return None
    return float(close)


def predict_with_unified_model(
    market: str,
    symbol: str,
    model_types: List[str] = None,
    lookback_days: int = 90,
    horizon: int = 1,
    on_stale: Optional[Callable[[str, str], None]] = None,
) -> Optional[PredictionResult]:
    """
    統合モデルを使用して1銘柄の予測を行う

    Args:
        market: 市場名
        symbol: 銘柄コード
        model_types: 使用するモデル名のリスト
        lookback_days: データ取得日数（未使用、互換性のため残す）
        horizon: 予測ホライズン（営業日）。1=翌日, 3=3日後, 5=5日後, 10=10日後。
        on_stale: 特徴量が古いため予想を出さなかったときに (market, symbol) で呼ぶ

    Returns:
        予測結果のDataFrame または None
    """
    if model_types is None:
        if horizon == 1:
            model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
        else:
            model_types = [
                f"UnifiedStockXGBoost_{horizon}d",
                f"UnifiedStockLightGBM_{horizon}d",
            ]

    # 特徴量データ読み込み
    df = load_feature_data(market, symbol)
    if df is None or df.empty:
        return None

    # y列（予測ターゲット）と特徴量を分離
    if "y" not in df.columns:
        return None

    # 足が止まった銘柄は特徴量の生成でスキップされ（#815）、止まった日を最後の行と
    # する前の特徴量が残る。その行で予想を出し続けないよう、最後の行の日付を見る（#829）
    last_date = _last_feature_date(df)
    if last_date is not None:
        stale_weekdays = count_weekdays_behind(last_date, market=market)
        if stale_weekdays > STALE_BARS_MAX_WEEKDAYS:
            logger.warning(
                "特徴量が古いため予想をスキップ: market=%s symbol=%s "
                "最後の行の日付=%s（平日で %d日 > %d日 古い）",
                market,
                symbol,
                last_date.strftime("%Y-%m-%d"),
                stale_weekdays,
                STALE_BARS_MAX_WEEKDAYS,
            )
            if on_stale is not None:
                on_stale(market, symbol)
            return None

    # 最新行を取得
    latest_X = feature_row(df, -1, market)

    # 基準の値段は特徴量の最後の行の日の終値。足の日付が合わない・取れないときは
    # 1日ずれた値段で予想の値段を出さないよう、予想を出さない（#845）
    current_price = _fetch_current_price(market, symbol, last_date)
    if current_price is None:
        return None

    # 推論サービスが有効なときだけ委譲する。
    # get_service_url() で先に判定するのは、未設定の既定パスで load_model_weights()
    # の DB クエリを余計に走らせないため（銘柄数ぶん積み上がるため無視できない）。
    if get_service_url() is not None:
        try:
            service_weights = load_model_weights(market, symbol, model_types)
            numeric_X = latest_X.select_dtypes(include="number")
            service_result = predict_via_service(
                market=market,
                symbol=symbol,
                current_price=float(current_price),
                features={
                    str(col): float(numeric_X[col].iloc[0])
                    for col in numeric_X.columns
                    if pd.notna(numeric_X[col].iloc[0])
                },
                model_types=list(model_types),
                model_weights=service_weights,
            )
            # None のときは下のインプロセス推論にそのままフォールバックする
            if service_result is not None:
                return service_result
        except Exception:
            logger.warning(
                "推論サービス委譲処理で例外（インプロセス推論にフォールバック）: "
                "market=%s symbol=%s",
                market,
                symbol,
                exc_info=True,
            )

    # 各モデルで予測（キャッシュされたモデルを使用）。モデルの読み込みは今までどおり
    # 1モデルずつ予想の直前に行い、読み込みの例外も予想の例外と同じに扱う
    def _record_zero_filled(zero_filled: List[str]) -> None:
        if zero_filled:
            logger.info(
                "予想の特徴量を0で埋めた: market=%s symbol=%s horizon=%d 列=%s",
                market,
                symbol,
                horizon,
                ",".join(zero_filled),
            )
        record_zero_fill(market, horizon, zero_filled)

    outcome = _run_ensemble(
        latest_X,
        current_price,
        market,
        symbol,
        horizon,
        [(name, partial(get_cached_model, name)) for name in model_types],
        weights_fn=load_model_weights,
        on_zero_filled=_record_zero_filled,
    )
    if outcome is None:
        return None

    # 上限・下限に当たった数を回ごとのログに出す（#1318）
    record_clip(market, horizon, outcome.diff_ratio_raw, outcome.clip_max_ratio)
    diff_ratio = outcome.diff_ratio
    avg_pred_price = float(current_price * (1 + diff_ratio))

    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=float(current_price),
        avg_pred_price=float(avg_pred_price),
        diff_ratio=float(diff_ratio),
        model_count=int(outcome.model_count),
    )


def predict_all_with_unified_model(
    model_types: List[str] = None, data_dir: str = None
) -> pd.DataFrame:
    """
    全銘柄について統合モデルで予測を行う

    Args:
        model_types: 使用するモデル名のリスト
        data_dir: データディレクトリ

    Returns:
        全銘柄の予測結果DataFrame
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if model_types is None:
        model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

    # モデルを事前にロード
    preload_models(model_types)

    if data_dir is None:
        pass  # DBから直接取得するため不要

    # 全銘柄をDBから取得
    all_keys = get_active_symbols()

    print(f"予測対象: {len(all_keys)}銘柄")

    # 並列予測
    def predict_wrapper(args):
        market, symbol = args
        try:
            return predict_with_unified_model(market, symbol, model_types=model_types)
        except Exception:
            logger.warning("銘柄予測失敗: market=%s symbol=%s", market, symbol, exc_info=True)
            return None

    results = []
    with collect_prediction_run_stats("unified_all"):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(predict_wrapper, key): key for key in all_keys}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
