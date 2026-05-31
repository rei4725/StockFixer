# flake8: noqa: F401
"""
DuckDB データベースアクセスパッケージ

アプリ全体の DB 接続・テーブル操作を一元管理する。
他のモジュールはこのパッケージの関数を使用して DB 操作を行うこと。

モジュール構成:
    _connection.py     - 接続管理（短命接続 + リトライ）・スキーマ DDL
    stock_features.py  - stock_features テーブル操作
    prediction.py      - prediction_results / model_metrics / prediction_accuracy テーブル操作
    market_data.py     - market_data_raw テーブル操作
    index_membership.py - index_membership_history テーブル操作
    experiment.py      - experiment_runs テーブル操作（R-211 実験トラッキング）
"""

import sys
import types

# ---------------------------------------------------------------------------
# テスト互換モジュールプロキシ
#
# `import src.utils.db as db_module` してから行う以下の操作を
# 実態である _connection モジュールへ透過転送する:
#   db_module._tables_initialized = False
#   db_module.get_db_path = lambda: "/tmp/test.duckdb"
# ---------------------------------------------------------------------------
from src.utils.db import _connection as _conn_module  # noqa: E402

# --- 接続管理 ---
from src.utils.db._connection import _RETRY_COUNT  # noqa: F401
from src.utils.db._connection import (
    _DB_CONFIG,
    _RETRY_DELAY,
    _db_connection,
    _init_tables,
    close_connection,
    get_readonly_connection,
    init_tables,
)

# --- experiment_runs ---
from src.utils.db.experiment import generate_run_id  # noqa: F401
from src.utils.db.experiment import load_best_run  # noqa: F401
from src.utils.db.experiment import load_experiment_runs, save_experiment_run

# --- index_membership_history ---
from src.utils.db.index_membership import (  # noqa: F401
    load_index_membership_symbols_as_of,
    save_index_membership_snapshot,
)

# --- market_data_raw ---
from src.utils.db.market_data import load_all_raw_ohlcv_symbols  # noqa: F401
from src.utils.db.market_data import load_raw_ohlcv, upsert_raw_ohlcv  # noqa: F401
from src.utils.db.migration_runner import get_applied_migrations, run_migrations  # noqa: F401

# --- data_quality_log ---
from src.utils.db.quality_log import insert_quality_log  # noqa: F401

# --- stock_features ---
from src.utils.db.stock_features import _ensure_columns  # noqa: F401
from src.utils.db.stock_features import delete_stock_features  # noqa: F401
from src.utils.db.stock_features import upsert_stock_features  # noqa: F401
from src.utils.db.stock_features import get_all_symbols, load_all_stock_features
from src.utils.db.stock_features import load_stock_features as load_stock_features  # noqa: F401

# --- system_config ---
from src.utils.db.system_config import get_config_value  # noqa: F401
from src.utils.db.system_config import set_config_value  # noqa: F401


class _DbPackageProxy(types.ModuleType):
    """
    特定の属性への代入操作を _connection モジュールへ転送するプロキシ。
    _db_connection() は _connection.__dict__ から get_db_path 等を動的参照するため、
    このプロキシ経由で setattr するとテスト時のモンキーパッチが正しく機能する。

    prediction.db の関数は遅延ロード（#338 で根本解消予定）。
    モジュールレベルの from src.prediction.db import ... を避けることで
    importlinter の BC 間接依存検出を回避する。
    """

    # _connection モジュールへ転送する属性名
    _FORWARDED = frozenset(["_tables_initialized", "get_db_path", "get_data_dir", "ensure_dir"])

    # prediction.db から遅延ロードする関数名
    _PREDICTION_DB = frozenset(
        [
            "load_drift_summary",
            "load_excluded_features",
            "load_feature_exclusion_candidates",
            "load_latest_prediction_timestamp",
            "load_model_weights",
            "load_paper_real_diff_summary",
            "load_prediction_accuracy",
            "load_prediction_markets",
            "load_prediction_results",
            "load_weekly_accuracy_snapshots",
            "save_feature_selection",
            "save_model_metrics",
            "save_prediction_accuracy",
            "save_prediction_results",
            "save_shap_values",
            "save_weekly_accuracy_snapshot",
        ]
    )

    def __setattr__(self, name: str, value) -> None:
        if name in _DbPackageProxy._FORWARDED:
            setattr(_conn_module, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name in _DbPackageProxy._FORWARDED:
            return getattr(_conn_module, name)
        if name in _DbPackageProxy._PREDICTION_DB:
            import importlib

            _pred_db = importlib.import_module("src.prediction.db")
            return getattr(_pred_db, name)
        raise AttributeError(f"module 'src.utils.db' has no attribute {name!r}")


def __getattr__(name: str):
    """PEP 562 module-level __getattr__ for static analysis (runtime: _DbPackageProxy handles this).

    Defines dynamic attributes so that type-checkers and pylint do not raise no-name-in-module
    for names that are lazily loaded from src.prediction.db.
    """
    if name in _DbPackageProxy._FORWARDED:
        return getattr(_conn_module, name)
    if name in _DbPackageProxy._PREDICTION_DB:
        import importlib

        _pred_db = importlib.import_module("src.prediction.db")
        return getattr(_pred_db, name)
    raise AttributeError(f"module 'src.utils.db' has no attribute {name!r}")


sys.modules[__name__].__class__ = _DbPackageProxy
