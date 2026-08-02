"""
統合モデル用予測処理

統合モデルを使用して全銘柄の予測を行う
"""

import logging
import warnings
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.prediction.db import load_model_weights
from src.prediction.remote_client import get_service_url, predict_via_service
from src.prediction.types import PredictionResult
from src.utils.data_path_utils import get_ticker
from src.utils.db import get_all_symbols, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# モデルキャッシュ（スレッドセーフ）
_model_cache: Dict[str, Any] = {}
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


def get_cached_model(model_name: str):
    """
    キャッシュからモデルを取得（なければロード）
    スレッドセーフな実装
    """
    from src.prediction.unified_model_pipeline import load_unified_model

    with _model_cache_lock:
        if model_name not in _model_cache:
            try:
                _model_cache[model_name] = load_unified_model(model_name)
            except FileNotFoundError:
                _model_cache[model_name] = None
        return _model_cache[model_name]


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


def predict_with_unified_model(
    market: str,
    symbol: str,
    model_types: List[str] = None,
    lookback_days: int = 90,
    horizon: int = 1,
) -> Optional[PredictionResult]:
    """
    統合モデルを使用して1銘柄の予測を行う

    Args:
        market: 市場名
        symbol: 銘柄コード
        model_types: 使用するモデル名のリスト
        lookback_days: データ取得日数（未使用、互換性のため残す）
        horizon: 予測ホライズン（営業日）。1=翌日, 3=3日後, 5=5日後, 10=10日後。

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

    # 現在価格をyfinanceからリアルタイムで取得
    try:
        yf_ticker = get_ticker(market, symbol)
        ticker_obj = yf.Ticker(yf_ticker)
        hist = ticker_obj.history(period="1d")
        if not hist.empty:
            current_price = float(hist["Close"].iloc[-1])
        else:
            # フォールバック: CSVのClose_lag1を使用
            if "Close_lag1" in df.columns:
                current_price = float(df["Close_lag1"].iloc[-1])
            else:
                current_price = float(df["y"].iloc[-2]) if len(df) > 1 else float(df["y"].iloc[-1])
    except Exception:
        logger.warning(
            "現在価格取得失敗（フォールバック使用）: market=%s symbol=%s",
            market,
            symbol,
            exc_info=True,
        )
        # エラー時はフォールバック
        if "Close_lag1" in df.columns:
            current_price = float(df["Close_lag1"].iloc[-1])
        else:
            current_price = float(df["y"].iloc[-2]) if len(df) > 1 else float(df["y"].iloc[-1])

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
    all_keys = get_all_symbols()

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
