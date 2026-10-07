# flake8: noqa: F401
"""
PostgreSQL データベースアクセスパッケージ

アプリ全体の DB 接続・テーブル操作を一元管理する。
他のモジュールはこのパッケージの関数を使用して DB 操作を行うこと。

モジュール構成:
    _connection.py     - 接続管理（短命接続 + リトライ）・スキーマ DDL
    stock_features.py  - stock_features テーブル操作
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
#   db_module.get_database_url = lambda: "postgresql://test"
# ---------------------------------------------------------------------------
from src.utils.db import _connection as _conn_module  # noqa: E402

# --- 接続管理 ---
from src.utils.db._connection import (
    _db_connection,
    close_connection,
    db_connection,
    get_readonly_connection,
    init_tables,
    set_test_connection,
)

# --- experiment_runs ---
from src.utils.db.experiment import generate_run_id  # noqa: F401
from src.utils.db.experiment import load_best_run  # noqa: F401
from src.utils.db.experiment import load_experiment_runs, save_experiment_run

# --- factory_runs（戦略ファクトリー #369） ---
from src.utils.db.factory_runs import (  # noqa: F401
    count_factory_runs,
    ensure_factory_tables,
    load_factory_hashes,
    load_factory_specs,
    save_factory_run,
)

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
from src.utils.db.stock_features import get_active_symbols, get_all_symbols, load_all_stock_features
from src.utils.db.stock_features import load_stock_features as load_stock_features  # noqa: F401

# --- strategy_promotions（戦略ファクトリー自動昇格ループ） ---
from src.utils.db.strategy_promotions import (  # noqa: F401
    ensure_strategy_promotions_table,
    load_active_promotions,
    mark_promotion_rolled_back,
    promotion_exists,
    save_strategy_promotion,
)

# --- system_config ---
from src.utils.db.system_config import get_config_value  # noqa: F401
from src.utils.db.system_config import set_config_value  # noqa: F401


class _DbPackageProxy(types.ModuleType):
    """
    特定の属性への代入操作を _connection モジュールへ転送するプロキシ。
    _db_connection() は _connection.__dict__ から get_database_url 等を動的参照するため、
    このプロキシ経由で setattr するとテスト時のモンキーパッチが正しく機能する。

    かつてはここで src.prediction.db の関数を importlib で遅延再輸出していたが、
    import-linter から見えない層逆転（utils -> prediction）だったため撤去した
    （2026-09-22 DB 所有権の疎結合化 設計書）。予測系テーブルは domain のポートと
    src/infrastructure/persistence/ のアダプタ越しに読むこと。
    """

    # _connection モジュールへ転送する属性名
    _FORWARDED = frozenset(["_tables_initialized", "get_database_url"])

    def __setattr__(self, name: str, value) -> None:
        if name in _DbPackageProxy._FORWARDED:
            setattr(_conn_module, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name in _DbPackageProxy._FORWARDED:
            return getattr(_conn_module, name)
        raise AttributeError(f"module 'src.utils.db' has no attribute {name!r}")


def __getattr__(name: str):
    """PEP 562 module-level __getattr__ for static analysis (runtime: _DbPackageProxy handles this)."""
    if name in _DbPackageProxy._FORWARDED:
        return getattr(_conn_module, name)
    raise AttributeError(f"module 'src.utils.db' has no attribute {name!r}")


sys.modules[__name__].__class__ = _DbPackageProxy
