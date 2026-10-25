"""
統合モデル用予測処理

統合モデルを使用して全銘柄の予測を行う
"""

import logging
import os
import warnings
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from config.settings import PREDICTION_ATR_CLIP_MULTIPLIER, STALE_BARS_MAX_WEEKDAYS
from src.prediction.db import load_model_weights
from src.prediction.range_clipping import clip_diff_ratio_to_atr_range
from src.prediction.remote_client import get_service_url, predict_via_service
from src.prediction.types import PredictionResult
from src.utils.data_path_utils import get_models_dir, get_ticker
from src.utils.db import get_active_symbols, load_stock_features
from src.utils.jp_market_holidays import is_jp_business_day
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
        特徴量DataFrame または None
    """
    try:
        df = load_stock_features(market, symbol)
        return df
    except Exception:
        logger.warning(
            "特徴量データ読み込み失敗: market=%s symbol=%s", market, symbol, exc_info=True
        )
        return None


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


def _as_feature_list(names: Any) -> Optional[List[str]]:
    """特徴量名の並びを list[str] に正規化する。名前として使えなければ None。"""
    if names is None or isinstance(names, (str, bytes)):
        return None
    try:
        items = list(names)
    except TypeError:
        return None
    return [str(name) for name in items] if items else None


def resolve_expected_features(model: Any) -> Optional[List[str]]:
    """モデルが学習時に使った特徴量名を解決する。解決できなければ None。

    sklearn 互換の ``feature_names_in_`` だけを見ると LightGBM を取りこぼす。
    lightgbm がこの属性を持つようになったのは 4.5.0 からで、それ以前に pickle
    されたモデルは LightGBM 独自の ``feature_name_`` にしか名前を持たない。
    属性が無い場合は AttributeError を送出するプロパティとして実装されている
    ため、``hasattr`` ではなく getattr チェーンで順に解決する。

    引数は ModelManager のラッパー（``.model`` に推定器を持つ）でも生の推定器
    でもよい。

    ``booster_.feature_name()`` へはフォールバックしない。DataFrame で学習した
    モデルなら上の2属性で必ず解決でき、numpy 配列で学習したモデルでは booster が
    ``f0`` / ``Column_0`` といった合成名を返すため、それを本物の特徴量名として
    扱うと全列 0 埋めのフレームで predict が「成功」してしまう。解決できない
    ことを None で表明する方が安全である。
    """
    estimator = getattr(model, "model", model)

    for attr in ("feature_names_in_", "feature_name_"):
        resolved = _as_feature_list(getattr(estimator, attr, None))
        if resolved is not None:
            return resolved

    return None


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

    # 特徴量列（文字列列・日付列・ターゲットを除外）
    # 除外条件は学習側の prepare_unified_features() と揃えること。学習時に
    # 存在しない date 列を予測時だけ通すと、アラインメントを解決できなかった
    # 場合に日付が predict() へ流れ込んで失敗する（#615）。
    exclude_cols = ["y", "market", "symbol", "date"]
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    X = df[feature_cols]

    # 最新行を取得（必ずコピーを作成）
    latest_X = X.iloc[[-1]].copy()

    # 基準の値段は特徴量の最後の行の日の終値。足の日付が合わない・取れないときは
    # 1日ずれた値段で予想の値段を出さないよう、予想を出さない（#845）
    current_price = _fetch_current_price(market, symbol, last_date)
    if current_price is None:
        return None

    # market_encoded列がない場合のみ追加（後方互換性）
    if "market_encoded" not in latest_X.columns:
        market_codes = {"us": 0, "jp": 1}
        latest_X["market_encoded"] = market_codes.get(market, 0)

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

    # 各モデルで予測（キャッシュされたモデルを使用）
    pred_prices = []
    succeeded_model_names = []
    for model_name in model_types:
        try:
            model = get_cached_model(model_name)
            if model is None:
                continue

            # モデルの特徴量と入力特徴量を揃える（コピーを作成して変更）
            latest_X_aligned = latest_X.copy()
            expected_features = resolve_expected_features(model)
            if expected_features is not None:
                # 不足している特徴量は0で埋める
                for feat in expected_features:
                    if feat not in latest_X_aligned.columns:
                        latest_X_aligned[feat] = 0
                # 期待される特徴量のみ選択
                latest_X_aligned = latest_X_aligned[expected_features]
            else:
                logger.warning(
                    "モデルの期待特徴量を解決できずアラインメントをスキップ: model=%s",
                    model_name,
                )

            pred = model.predict(latest_X_aligned)
            if isinstance(pred, pd.Series):
                pred_return = float(pred.iloc[-1])
            elif isinstance(pred, (list, tuple)):
                pred_return = float(pred[-1])
            else:
                pred_return = float(pred)
            # 変化率から絶対価格を計算
            pred_price = current_price * (1 + pred_return)
            pred_prices.append(pred_price)
            succeeded_model_names.append(model_name)
        except Exception:
            # WARNING では埋もれる。1モデル落ちるとアンサンブルが単一モデルに
            # 縮退したまま予測値は出続けるため、失敗が見えないと気づけない（#615）。
            logger.error(
                "モデル予測スキップ（アンサンブルが縮退します）: model=%s market=%s symbol=%s",
                model_name,
                market,
                symbol,
                exc_info=True,
            )
            continue

    if not pred_prices:
        return None

    # model_metrics の directional_accuracy をソフトマックス重みでアンサンブル（R-202）
    weights = load_model_weights(market, symbol, succeeded_model_names)
    avg_pred_price = float(sum(p * w for p, w in zip(pred_prices, weights)))
    diff_ratio = (avg_pred_price - current_price) / current_price

    # ATR ベースの妥当なレンジへクリップ（あり得ない外れ値予測の丸め）。
    # stock_features には生の High/Low は残らずラグ特徴量しか無いため、
    # 直近1営業日分の ATR として atr_lag1 を使う（predict_single 側は
    # 生 OHLC から都度計算するのと対照的）。
    atr_value: Optional[float] = None
    if "atr_lag1" in latest_X.columns:
        raw_atr = latest_X["atr_lag1"].iloc[0]
        if pd.notna(raw_atr):
            atr_value = float(raw_atr)
    diff_ratio = clip_diff_ratio_to_atr_range(
        diff_ratio,
        atr_value,
        current_price,
        horizon=horizon,
        atr_multiplier=PREDICTION_ATR_CLIP_MULTIPLIER,
    )
    avg_pred_price = float(current_price * (1 + diff_ratio))

    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=float(current_price),
        avg_pred_price=float(avg_pred_price),
        diff_ratio=float(diff_ratio),
        model_count=int(len(pred_prices)),
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
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(predict_wrapper, key): key for key in all_keys}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
