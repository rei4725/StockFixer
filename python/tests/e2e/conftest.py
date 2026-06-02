"""
E2E テスト共通 fixture

テスト専用の孤立環境（一時 DuckDB + 一時モデルディレクトリ）を構築し、
合成 OHLCV データを投入してパイプライン全体を通せる状態にする。

使用するテスト用銘柄:
    market = "us", symbol = "E2ETEST"
    実際の yfinance ティッカーとしては存在しないため、
    外部通信が必要なステップはモックで置き換えている。
"""

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
E2E_MARKET = "us"
E2E_SYMBOL = "E2ETEST"
N_DAYS = 200  # 特徴量生成・モデル学習に十分な行数


# ---------------------------------------------------------------------------
# 1. 合成 OHLCV fixture（モジュール全体で共有）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e2e_ohlcv() -> pd.DataFrame:
    """200 営業日分の固定 OHLCV DataFrame（yfinance 戻り値と同形式）。"""
    rng = np.random.default_rng(42)
    # 最終日を "今日" にして freshness check をパスさせる
    last_bday = pd.Timestamp.today().normalize()
    # 土日なら直前の金曜に戻す
    while last_bday.weekday() >= 5:
        last_bday -= pd.Timedelta(days=1)
    dates = pd.bdate_range(end=last_bday, periods=N_DAYS)

    close = np.cumsum(rng.normal(0, 0.5, N_DAYS)) + 100.0
    close = np.clip(close, 50.0, 300.0)

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.uniform(-0.005, 0.005, N_DAYS)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, N_DAYS))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, N_DAYS))),
            "Close": close,
            "Volume": rng.integers(500_000, 2_000_000, N_DAYS).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 2. E2E 環境 fixture（モジュール全体で共有）
#    - 一時 DuckDB・一時モデルディレクトリを構築
#    - DB / モデルパス関数をすべてパッチ
#    - 合成データを投入してモデルを学習済みにする
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e2e_db_env(e2e_ohlcv, tmp_path_factory):
    """
    E2E テスト用の孤立環境を構築して yield する。

    yield する辞書:
        db_path   : 一時 DuckDB のファイルパス
        models_dir: 一時モデルディレクトリ
        market    : E2E_MARKET
        symbol    : E2E_SYMBOL
        ohlcv     : 合成 OHLCV DataFrame
    """
    import src.utils.db as db_module

    base = tmp_path_factory.mktemp("e2e_env")
    db_path = str(base / "e2e.duckdb")
    models_dir = str(base / "models")
    os.makedirs(models_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 一時モデルサブディレクトリを返すヘルパー
    # ---------------------------------------------------------------------
    def _models_subdir(m, s):
        d = os.path.join(models_dir, f"{m}_{s}")
        os.makedirs(d, exist_ok=True)
        return d

    # ---------------------------------------------------------------------
    # DB パッチ: _connection モジュール経由の get_db_path を差し替え
    # （_DbPackageProxy.__setattr__ が _connection へ転送する）
    # ---------------------------------------------------------------------
    orig_db_path_fn = db_module.get_db_path
    db_module.close_connection()
    db_module.get_db_path = lambda: db_path
    db_module._tables_initialized = False

    # ---------------------------------------------------------------------
    # モデルディレクトリパッチ
    # 各モジュールが from ... import した名前を個別に差し替える
    # ---------------------------------------------------------------------
    patchers = [
        mock.patch(
            "src.prediction.manager.get_models_subdir",
            side_effect=_models_subdir,
        ),
        mock.patch(
            "src.prediction.manager.get_models_dir",
            return_value=models_dir,
        ),
        mock.patch(
            "src.prediction.predict_single.get_models_subdir",
            side_effect=_models_subdir,
        ),
        mock.patch(
            "src.utils.data_path_utils.get_models_subdir",
            side_effect=_models_subdir,
        ),
        mock.patch(
            "src.utils.data_path_utils.get_models_dir",
            return_value=models_dir,
        ),
        # Discord 送信を抑制（SHAP / 学習完了通知）
        mock.patch("src.reporting.discord.discord_utils.send_shap_notification", return_value=None),
        mock.patch(
            "src.reporting.discord.discord_utils.send_daily_pipeline_completion", return_value=None
        ),
        mock.patch(
            "src.reporting.discord.discord_utils.send_daily_pipeline_error", return_value=None
        ),
        # クロスアセット特徴量をモックして学習/予測で特徴量数を一致させる（R-306）
        mock.patch(
            "src.market_data.prediction_adapter"
            ".PredictionMarketDataAdapter.fetch_cross_asset_features",
            return_value=None,
        ),
        mock.patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None),
    ]
    for p in patchers:
        p.start()

    from src.backtest.ports import set_model_manager_factory
    from src.prediction.manager import ModelManager

    set_model_manager_factory(ModelManager)

    try:
        # -----------------------------------------------------------------
        # フィクスチャデータ投入
        # -----------------------------------------------------------------
        _inject_raw_ohlcv(E2E_MARKET, E2E_SYMBOL, e2e_ohlcv)
        _generate_and_save_features(E2E_MARKET, E2E_SYMBOL, e2e_ohlcv)
        _train_models(E2E_MARKET, E2E_SYMBOL)

        yield {
            "db_path": db_path,
            "models_dir": models_dir,
            "market": E2E_MARKET,
            "symbol": E2E_SYMBOL,
            "ohlcv": e2e_ohlcv,
        }
    finally:
        db_module.close_connection()
        db_module.get_db_path = orig_db_path_fn
        db_module._tables_initialized = False
        for p in patchers:
            p.stop()


# ---------------------------------------------------------------------------
# 内部ヘルパー関数（fixture から呼び出す）
# ---------------------------------------------------------------------------


def _inject_raw_ohlcv(market: str, symbol: str, df: pd.DataFrame) -> None:
    """合成 OHLCV を market_data_raw テーブルへ直接 UPSERT する。"""
    from src.market_data.saver import save_raw_ohlcv

    save_raw_ohlcv(market, symbol, df)


def _generate_and_save_features(market: str, symbol: str, df: pd.DataFrame) -> None:
    """
    OHLCV から特徴量を生成して stock_features へ保存する。
    data_pipeline.fetch_stock_data_with_features と同等の処理を直接実行する
    （yfinance 呼び出しを完全に回避するため）。
    """
    from src.market_data.technical import add_technical_indicators, create_basic_lag_features
    from src.utils.data_path_utils import normalize_col
    from src.utils.db import upsert_stock_features

    work = df.copy()
    work = add_technical_indicators(work)

    nan_count = int(work.isnull().sum().sum())
    if nan_count > 0:
        work = work.ffill().bfill()

    X, y = create_basic_lag_features(work, n_lags=10)
    if X is None or X.empty:
        raise RuntimeError("E2E テスト用の特徴量生成に失敗しました")

    # data_pipeline と同じカラム名正規化
    X.columns = [normalize_col(c) for c in X.columns]

    data = X.copy()
    data["market"] = market
    data["symbol"] = symbol
    market_codes = {"us": 0, "jp": 1}
    data["market_encoded"] = market_codes.get(market, -1)
    data["y"] = y

    upsert_stock_features(market, symbol, data)


def _train_models(market: str, symbol: str) -> None:
    """
    stock_features からデータを読み込み XGBoost / LightGBM を学習する。
    決算日データ取得（yfinance）・Discord 送信はモック不要（呼び出し側でパッチ済み）。
    """
    from src.prediction.training_pipeline import train_models_for_symbol

    # get_earnings_dates は yfinance を呼ぶが、失敗時は空配列を返すため
    # 明示的なモックなしでも安全に動作する
    result = train_models_for_symbol(market, symbol)
    if result["status"] == "error":
        raise RuntimeError(f"E2E テスト用のモデル学習に失敗しました: {result.get('error')}")
