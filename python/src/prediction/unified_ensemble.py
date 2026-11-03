"""統合モデルのアンサンブルの芯（本番の予想とバックテストが同じ道を通る）。

predict_unified から切り出した。名前は predict_unified からも引ける。
"""

from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from config.settings import PREDICTION_ATR_CLIP_MULTIPLIER
from src.prediction.db import load_model_weights
from src.prediction.range_clipping import atr_clip_max_ratio, clip_diff_ratio_to_atr_range
from src.prediction.types import EnsembleOutcome
from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def ensemble_diff_ratio(
    latest_X: pd.DataFrame,
    current_price: float,
    market: str,
    symbol: str,
    horizon: int,
    models: List[Tuple[str, Any]],
    weights_fn: Callable[[str, str, List[str]], List[float]] = load_model_weights,
) -> Optional[EnsembleOutcome]:
    """
    統合モデルのアンサンブルの芯。1行の特徴量から予想の変化率を出す

    本番の predict_with_unified_model と同じ処理（0埋め・列そろえ → predict → 重み →
    ATR クリップ）で、バックテストが候補のモデルの実物を本番の重みの名前で渡せる。
    回ごとの記録（record_zero_fill・record_clip）と0埋めのログは呼ばない。

    Args:
        latest_X: 特徴量の1行
        current_price: 基準の値段
        market: 市場名
        symbol: 銘柄コード
        horizon: 予測ホライズン（営業日）
        models: （重みを引く名前, モデルの実物）の並び。実物が None のものは飛ばす
        weights_fn: (market, symbol, 予想できたモデルの名前) から重みを返す

    Returns:
        結果。使えるモデルが1つも無ければ None

    Raises:
        RuntimeError: 全モデルの予想が例外で失敗したとき（原因に最後の例外が付く）
    """
    return _run_ensemble(
        latest_X,
        current_price,
        market,
        symbol,
        horizon,
        [(name, partial(_identity, model)) for name, model in models],
        weights_fn=weights_fn,
    )


def _identity(value: Any) -> Any:
    return value


def _run_ensemble(
    latest_X: pd.DataFrame,
    current_price: float,
    market: str,
    symbol: str,
    horizon: int,
    model_loaders: List[Tuple[str, Callable[[], Any]]],
    weights_fn: Callable[[str, str, List[str]], List[float]],
    on_zero_filled: Optional[Callable[[List[str]], None]] = None,
) -> Optional[EnsembleOutcome]:
    """
    ensemble_diff_ratio の中身。モデルは（名前, 読み込む関数）で受け、予想の直前に読む

    本番の関数は読み込みの例外を予想の例外と同じに扱い、0埋めの記録を重みの読み込みより
    前に出すため、読み込む関数と on_zero_filled（モデルを回し終えたら0埋めした列で呼ぶ）を
    渡す。
    """
    pred_prices = []
    succeeded_model_names = []
    last_model_error: Optional[Exception] = None
    # モデルをまたいで0で埋めた特徴量（回ごとのログに出す、#1318）
    zero_filled: set = set()
    for model_name, load_model in model_loaders:
        try:
            model = load_model()
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
                        zero_filled.add(feat)
                # 期待される特徴量のみ選択
                latest_X_aligned = latest_X_aligned[expected_features]
                # 列はあっても値が NaN のもの（DB に NULL のまま残った列）も0で埋める（#1425）
                nan_features = [
                    feat for feat in expected_features if pd.isna(latest_X_aligned[feat].iloc[0])
                ]
                if nan_features:
                    latest_X_aligned = latest_X_aligned.fillna({f: 0 for f in nan_features})
                    zero_filled.update(nan_features)
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
        except Exception as e:
            last_model_error = e
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

    zero_filled_list = sorted(zero_filled)
    if on_zero_filled is not None:
        on_zero_filled(zero_filled_list)

    if not pred_prices:
        # 例外で全滅したときは None にせず投げ、呼び出し元の例外の数と知らせに乗せる（#1140）
        if last_model_error is not None:
            raise RuntimeError(
                f"全モデルの予測が例外で失敗: market={market} symbol={symbol}"
            ) from last_model_error
        return None

    # model_metrics の directional_accuracy をソフトマックス重みでアンサンブル（R-202）
    weights = weights_fn(market, symbol, succeeded_model_names)
    avg_pred_price = float(sum(p * w for p, w in zip(pred_prices, weights)))
    diff_ratio_raw = (avg_pred_price - current_price) / current_price

    # ATR ベースの妥当なレンジへクリップ（あり得ない外れ値予測の丸め）。
    # stock_features には生の High/Low は残らずラグ特徴量しか無いため、
    # 直近1営業日分の ATR として atr_lag1 を使う（predict_single 側は
    # 生 OHLC から都度計算するのと対照的）。
    atr_value: Optional[float] = None
    if "atr_lag1" in latest_X.columns:
        raw_atr = latest_X["atr_lag1"].iloc[0]
        if pd.notna(raw_atr):
            atr_value = float(raw_atr)
    clip_max_ratio = atr_clip_max_ratio(
        atr_value, current_price, horizon, atr_multiplier=PREDICTION_ATR_CLIP_MULTIPLIER
    )
    diff_ratio = clip_diff_ratio_to_atr_range(
        diff_ratio_raw,
        atr_value,
        current_price,
        horizon=horizon,
        atr_multiplier=PREDICTION_ATR_CLIP_MULTIPLIER,
    )
    return EnsembleOutcome(
        diff_ratio_raw=float(diff_ratio_raw),
        diff_ratio=float(diff_ratio),
        clip_max_ratio=clip_max_ratio,
        model_count=len(pred_prices),
        zero_filled=zero_filled_list,
    )


def feature_row(df: pd.DataFrame, position: int, market: str) -> pd.DataFrame:
    """
    load_stock_features の表から、予想に渡す特徴量の1行（コピー）を作る

    本番の予想（predict_with_unified_model）とバックテストが同じ行の作り方を使う。

    Args:
        df: load_stock_features の結果
        position: 行の位置（iloc。-1 で最後の行）
        market: 市場名（market_encoded 列が無いときに使う）

    Returns:
        1行の DataFrame
    """
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
    row = df[feature_cols].iloc[[position]].copy()

    # market_encoded列がない場合のみ追加（後方互換性）
    if "market_encoded" not in row.columns:
        market_codes = {"us": 0, "jp": 1}
        row["market_encoded"] = market_codes.get(market, 0)
    return row
