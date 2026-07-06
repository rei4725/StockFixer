"""
pytest 共通 fixture と conftest

Unit Test / Integration Test で共有する fixture を定義。
"""

import os
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

# ============================================
# basetemp の実行毎ユニーク化（wipe 衝突防止）
# ============================================

# pytest は明示 --basetemp を起動時に無条件 rm_rf する。かつて pytest.ini で
# 固定の --basetemp=.pytest_tmp_runs（共有ルート）を指定していたところ、昇格
# タスク（auto_deploy.ps1）が残した ACL 付きディレクトリに wipe が当たり、
# 非昇格のローカル実行が全テスト WinError 5 で ERROR 化した。
# 対策としてデプロイ側と同じ「共有ルート直下に実行毎の一意サブディレクトリ」
# 方式に統一し、wipe 対象を常に自分専有の新規ディレクトリにする。
# ルート自体は誰も wipe しない。

_PYTEST_TMP_ROOT = ".pytest_tmp_runs"
_LOCAL_BASETEMP_PREFIX = "local_"
_LOCAL_BASETEMP_KEEP = 3


def _prune_old_local_basetemps(root: Path, keep: int = _LOCAL_BASETEMP_KEEP) -> None:
    """古い local_* basetemp をベストエフォートで削除する。

    削除できないもの（他プロセス使用中・権限不足の残骸）は無視して素通りする。
    デプロイ側が生成する smoke_/unit_/e2e_* には触れない（掃除の責務は
    auto_deploy.ps1 の Clear-OldPytestTmp 側にある）。
    """
    try:
        stale_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name.startswith(_LOCAL_BASETEMP_PREFIX)),
            key=lambda p: p.name,  # 名前に UTC ms を含むため辞書順 = 時系列順
            reverse=True,
        )[keep:]
    except OSError:
        return
    for stale in stale_dirs:
        shutil.rmtree(stale, ignore_errors=True)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """--basetemp 未指定時に .pytest_tmp_runs/local_<stamp>_<pid> を割り当てる。

    tryfirst で組み込み tmpdir プラグインが basetemp を読む前に差し込む。
    デプロイ側（auto_deploy.ps1）は明示 --basetemp を渡すため対象外。
    pytest 9.x は明示 basetemp の親を自動生成しないため、ルートの存在を保証する。
    """
    if config.option.basetemp is not None:
        return
    root = config.rootpath / _PYTEST_TMP_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return  # ルートを用意できない場合は pytest 既定（システム temp）に任せる
    stamp = int(time.time() * 1000)
    config.option.basetemp = root / f"{_LOCAL_BASETEMP_PREFIX}{stamp}_{os.getpid()}"
    _prune_old_local_basetemps(root)


# ============================================
# ポート注入（合成ルート相当）
# ============================================


@pytest.fixture(autouse=True)
def _wire_default_ports():
    """各テストの前に BC ポートへデフォルトの market_data アダプタを注入する。

    本番では orchestration の wire_ports() が担う注入を、テストでは autouse で
    肩代わりする。getter を個別に patch / set_*_port するテストはそのまま上書きできる。
    """
    from src.backtest.data_port import set_backtest_data_port
    from src.market_data.backtest_adapter import BacktestMarketDataAdapter
    from src.market_data.prediction_adapter import PredictionMarketDataAdapter
    from src.prediction.ports import set_market_data_port

    set_backtest_data_port(BacktestMarketDataAdapter())
    set_market_data_port(PredictionMarketDataAdapter())
    yield


# ============================================
# データ生成 Helper
# ============================================


@pytest.fixture
def sample_price_df():
    """サンプル株価 DataFrame を生成する fixture"""
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 102, 103, 104, 105],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="B"),
    )


@pytest.fixture
def sample_features_df():
    """サンプル特徴量 DataFrame を生成する fixture"""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "Close": [100 + i for i in range(10)],
            "Close_lag1": [0] + [100 + i for i in range(9)],
            "Close_lag2": [0, 0] + [100 + i for i in range(8)],
            "Volume": [1000 + i * 100 for i in range(10)],
            "SMA_5": [100 + i * 0.5 for i in range(10)],
            "RSI": [50 + i * 2 for i in range(10)],
            "symbol": ["7203"] * 10,
            "market": ["jp"] * 10,
        },
        index=dates,
    )


@pytest.fixture
def sample_signal_series(sample_price_df):
    """サンプルシグナル Series (1/0/-1) を生成する fixture"""
    return pd.Series(
        [1, 0, 0, -1, 0],  # Buy → Hold → Hold → Sell → Hold
        index=sample_price_df.index,
    )


# ============================================
# 長期データ Fixture
# ============================================


@pytest.fixture
def sample_price_df_long():
    """テクニカル指標計算に十分な行数（60行）のサンプル株価 DataFrame"""
    import numpy as np

    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    close = 100.0 + np.cumsum(rng.uniform(-1.0, 1.0, 60))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1000, 5000, 60),
        },
        index=dates,
    )


@pytest.fixture(params=["jp", "us", "nasdaq"])
def market(request):
    """マーケット種別をパラメータ化する fixture"""
    return request.param
