"""バックテスト実行ロジック（単一期間 / Walk-Forward）。

run_backtest.py 等のラッパーから呼ばれるサービス層の実行関数群。
"""

from typing import Any, Callable, Optional, Tuple

import pandas as pd

from config.settings import DEFAULT_SLIPPAGE_JP, DEFAULT_SLIPPAGE_US
from src.backtest.pipeline.features import load_features
from src.backtest.ports import get_model_manager
from src.backtest.slippage import make_slippage_fn
from src.utils.logger import get_logger

logger = get_logger(__name__)


def default_slippage_for(market: str) -> float:
    """市場別のデフォルト片道スリッページ率を返す（#494）。"""
    return DEFAULT_SLIPPAGE_JP if market == "jp" else DEFAULT_SLIPPAGE_US


def _resolve_slippage(
    market: str, slippage: Optional[float], dynamic_slippage: bool
) -> Tuple[float, Optional[Callable[[int, float, float], float]]]:
    """フラットスリッページ（未指定なら市場別デフォルト）と動的スリッページ関数を解決する。"""
    eff_slippage = slippage if slippage is not None else default_slippage_for(market)
    slippage_fn = make_slippage_fn() if dynamic_slippage else None
    return eff_slippage, slippage_fn


def run_backtest_single(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    model_name: Optional[str] = None,
    task_name: str = "return_regression",
    threshold: float = 0.0,
    source: str = "file",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    train_ratio: float = 0.8,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: Optional[float] = None,
    dynamic_slippage: bool = True,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    position_sizing: str = "full",
    position_fraction: float = 0.5,
    atr_risk_pct: float = 0.02,
    atr_multiplier: float = 1.0,
    atr_min_fraction: float = 0.1,
    atr_max_fraction: float = 1.0,
    ensemble: bool = False,
    apply_min_change_filter: bool = False,
    exclude_sentiment: bool = False,
) -> Tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    """
    単一学習/検証期間のバックテストを実行する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_type: モデルタイプ ("XGBoostModel" or "LightGBMModel")
        model_name: モデル名 (Noneなら "Backtest{model_type}")
        task_name: タスク名 ("return_regression")
        threshold: シグナル発生の変化率閾値
        source: データソース ("file" or "raw")
        start_date: バックテスト開始日
        end_date: バックテスト終了日
        train_ratio: 学習データ比率
        initial_cash: 初期資金
        fee_rate: 取引手数料率
        slippage: 片道スリッページ率（None なら市場別デフォルト US5bps/JP10bps）
        dynamic_slippage: True で出来高連動の動的スリッページ（平方根インパクト）を加算
        stop_loss_pct: ストップロス率（例: 0.05=5%下落で損切り）
        take_profit_pct: テイクプロフィット率（例: 0.10=10%上昇で利確）
        position_sizing: ポジションサイジング ("full", "fixed", "confidence", "atr")
        position_fraction: 固定ポジション比率（fixed モード用）
        atr_risk_pct: ATRモード: 1トレードあたりのリスク割合（デフォルト: 2%）
        atr_multiplier: ATRモード: ストップ幅のATR倍数（デフォルト: 1.0）
        atr_min_fraction: ATRモード: 建玉下限比率（デフォルト: 10%）
        atr_max_fraction: ATRモード: 建玉上限比率（デフォルト: 100%）
        ensemble: XGBoost+LightGBMアンサンブル予測を使用
        exclude_sentiment: True の場合、センチメント列を特徴量から除外して学習する

    Returns:
        (result_df, metrics, None) のタプル
    """
    from src.backtest.backtester import Backtester
    from src.utils.signal_generator import SignalGenerator

    task = _build_task(task_name, threshold)
    model_name = model_name or f"Backtest{model_type}"

    df = load_features(market, symbol, source)

    # 日付列をインデックスに設定（大文字・小文字どちらも対応）
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is not None:
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # インデックスが日付型でない場合、期間フィルタはスキップ
        start_date = None
        end_date = None

    # 期間フィルタ
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # train / test 分割
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    logger.info(f"学習期間: {train_df.index[0]} ～ {train_df.index[-1]} ({len(train_df)}行)")
    logger.info(f"検証期間: {test_df.index[0]} ～ {test_df.index[-1]} ({len(test_df)}行)")

    # ラベル付与
    label_col = task.label_col
    train_df = train_df.copy()
    test_df = test_df.copy()
    exclude_cols = {label_col, "market", "symbol", "market_encoded"}

    if label_col not in train_df.columns:
        train_df[label_col] = task.make_labels(
            train_df.rename(columns={"close": "Close"}, errors="ignore")
        )
        test_df[label_col] = task.make_labels(
            test_df.rename(columns={"close": "Close"}, errors="ignore")
        )

    _SENTIMENT_PREFIXES = ("sentiment_", "news_count")
    feature_cols = [
        c
        for c in train_df.columns
        if c not in exclude_cols
        and c not in ("Close", "close")
        and not (exclude_sentiment and (c.startswith(_SENTIMENT_PREFIXES)))
    ]

    # NULL を含む行を極力除去しない（90% 以上、有効な特徴量が必要）
    # thresh: 90% の列に値があることを要求
    min_valid = int(len(feature_cols) * 0.9)
    X_train = train_df[feature_cols].dropna(thresh=min_valid)
    y_train = train_df.loc[X_train.index, label_col].dropna()
    X_train = X_train.loc[y_train.index]

    X_test = test_df[feature_cols].dropna(thresh=min_valid)

    # モデル学習・予測
    mm = get_model_manager()

    if ensemble:
        pred = _ensemble_predict(mm, X_train, y_train, X_test, model_name)
    else:
        mm.create_model(model_type, model_name)
        mm.train_model(model_name, X_train, y_train)
        pred = pd.Series(mm.predict_with_model(model_name, X_test), index=X_test.index)

    signal = task.make_signal(pred)

    # スリッページ解決（未指定は市場別デフォルト、動的モデルは任意）
    eff_slippage, slippage_fn = _resolve_slippage(market, slippage, dynamic_slippage)

    # シミュレーション
    backtester = Backtester(
        model_manager=mm,
        signal_generator=SignalGenerator(market=market, symbol=symbol),
        data_loader=None,
        start_date=None,
        end_date=None,
        market=market,
        symbol=symbol,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=eff_slippage,
        slippage_fn=slippage_fn,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
        atr_min_fraction=atr_min_fraction,
        atr_max_fraction=atr_max_fraction,
    )
    result_df, metrics = backtester.simulate_trading(
        test_df.loc[X_test.index],
        signal,
        pred=pred,
    )
    close_col = "Close" if "Close" in test_df.columns else "close"
    price_series = test_df[close_col].rename("price")
    return result_df, metrics, price_series


def run_backtest_walk_forward(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    model_name: Optional[str] = None,
    task_name: str = "return_regression",
    threshold: float = 0.0,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: Optional[float] = None,
    dynamic_slippage: bool = True,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    position_sizing: str = "full",
    position_fraction: float = 0.5,
    atr_risk_pct: float = 0.02,
    atr_multiplier: float = 1.0,
    atr_min_fraction: float = 0.1,
    atr_max_fraction: float = 1.0,
    ensemble: bool = False,
    enable_short: bool = False,
) -> Tuple[None, None, pd.DataFrame]:
    """
    Walk-Forward バックテストを実行する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_type: モデルタイプ
        model_name: モデル名 (Noneなら "Backtest{model_type}")
        task_name: タスク名
        threshold: シグナル発生の変化率閾値
        source: データソース ("file", "api", "raw")
        n_splits: Walk-Forward の分割数
        initial_cash: 初期資金
        fee_rate: 取引手数料率
        slippage: 片道スリッページ率（None なら市場別デフォルト US5bps/JP10bps）
        dynamic_slippage: True で出来高連動の動的スリッページ（平方根インパクト）を加算
        stop_loss_pct: ストップロス率
        take_profit_pct: テイクプロフィット率
        position_sizing: ポジションサイジング ("full", "fixed", "confidence", "atr")
        position_fraction: 固定ポジション比率（fixed モード用）
        atr_risk_pct: ATRモード時に１トレードでリスクする資金の割合（デフォルト: 2%）
        atr_multiplier: ATRの何倒をストップ幅とするか（デフォルト: 1.0）
        atr_min_fraction: ATRモード: 建玉下限比率（デフォルト: 10%）
        atr_max_fraction: ATRモード: 建玉上限比率（デフォルト: 100%）
        ensemble: XGBoost+LightGBMアンサンブル予測を使用

    Returns:
        (None, None, wf_df) のタプル
    """
    from src.backtest.walk_forward import WalkForwardValidator
    from src.utils.signal_generator import SignalGenerator

    task = _build_task(task_name, threshold)
    model_name = model_name or f"Backtest{model_type}"

    mm = get_model_manager()
    if not ensemble:
        mm.create_model(model_type, model_name)

    eff_slippage, slippage_fn = _resolve_slippage(market, slippage, dynamic_slippage)

    wfv = WalkForwardValidator(
        market=market,
        symbol=symbol,
        model_manager=mm,
        signal_generator=SignalGenerator(market=market, symbol=symbol),
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=eff_slippage,
        slippage_fn=slippage_fn,
        n_splits=n_splits,
        source=source,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
        atr_min_fraction=atr_min_fraction,
        atr_max_fraction=atr_max_fraction,
        ensemble=ensemble,
        enable_short=enable_short,
    )

    results_df = wfv.run(model_name=model_name, task=task)
    return None, None, results_df


# --- internal helpers ---


def _build_task(task_name: str, threshold: float) -> Any:
    from src.backtest.task import ReturnRegressionTask

    if task_name == "return_regression":
        return ReturnRegressionTask(threshold=threshold)
    raise ValueError(f"未対応のタスク: {task_name}")


def _ensemble_predict(
    mm: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    base_name: str,
) -> pd.Series:
    """アンサンブル予測（平均）を返す（XGBoost + LightGBM）。

    両モデルを同一データで学習し、予測値の平均を取ることで
    バイアスを低減する。

    Args:
        mm: ModelManager インスタンス
        X_train: 学習用特徴量
        y_train: 学習用ラベル
        X_test: 検証用特徴量
        base_name: ベースモデル名

    Returns:
        アンサンブル予測値の Series
    """
    import numpy as np

    xgb_name = f"{base_name}_XGB"
    lgb_name = f"{base_name}_LGB"

    mm.create_model("XGBoostModel", xgb_name)
    mm.create_model("LightGBMModel", lgb_name)

    mm.train_model(xgb_name, X_train, y_train)
    mm.train_model(lgb_name, X_train, y_train)

    pred_xgb = mm.predict_with_model(xgb_name, X_test)
    pred_lgb = mm.predict_with_model(lgb_name, X_test)

    avg = np.mean([pred_xgb, pred_lgb], axis=0)
    print("  [Ensemble] XGBoost + LightGBM 予測値を平均")
    return pd.Series(avg, index=X_test.index)
