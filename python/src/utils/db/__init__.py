"""
DuckDB データベースアクセスパッケージ

アプリ全体の DB 接続・テーブル操作を一元管理する。
他のモジュールはこのパッケージの関数を使用して DB 操作を行うこと。

モジュール構成:
    _connection.py     - 接続管理（短命接続 + リトライ）・スキーマ DDL
    stock_features.py  - stock_features テーブル操作
    prediction.py      - prediction_results / model_metrics / prediction_accuracy テーブル操作
    market_data.py     - market_data_raw テーブル操作
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
from src.utils.db._connection import (  # noqa: F401
    _DB_CONFIG,
    _RETRY_COUNT,
    _RETRY_DELAY,
    _db_connection,
    _init_tables,
    close_connection,
    get_connection,
    get_readonly_connection,
    init_tables,
)

# --- market_data_raw ---
from src.utils.db.market_data import (  # noqa: F401
    load_all_raw_ohlcv_symbols,
    load_raw_ohlcv,
    upsert_raw_ohlcv,
)

# --- prediction_results / model_metrics / prediction_accuracy ---
from src.utils.db.prediction import (  # noqa: F401
    load_drift_summary,
    load_latest_prediction_timestamp,
    load_prediction_accuracy,
    load_prediction_markets,
    load_prediction_results,
    save_model_metrics,
    save_prediction_accuracy,
    save_prediction_results,
)

# --- stock_features ---
from src.utils.db.stock_features import (  # noqa: F401
    _ensure_columns,
    delete_stock_features,
    get_all_symbols,
    load_all_stock_features,
    load_stock_features,
    upsert_stock_features,
)


class _DbPackageProxy(types.ModuleType):
    """
    特定の属性への代入操作を _connection モジュールへ転送するプロキシ。
    _db_connection() は _connection.__dict__ から get_db_path 等を動的参照するため、
    このプロキシ経由で setattr するとテスト時のモンキーパッチが正しく機能する。
    """

    # _connection モジュールへ転送する属性名
    _FORWARDED = frozenset(["_tables_initialized", "get_db_path", "get_data_dir", "ensure_dir"])

    def __setattr__(self, name: str, value) -> None:
        if name in _DbPackageProxy._FORWARDED:
            setattr(_conn_module, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name in _DbPackageProxy._FORWARDED:
            return getattr(_conn_module, name)
        raise AttributeError(f"module 'src.utils.db' has no attribute {name!r}")


sys.modules[__name__].__class__ = _DbPackageProxy
