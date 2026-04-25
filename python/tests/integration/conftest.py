# Integration test専用 fixture。実DB・外部ライブラリ依存を含む。
import pytest

# ============================================
# 環境チェック Marker
# ============================================


@pytest.fixture(scope="session")
def has_xgboost():
    """Check if XGBoost is available."""
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_duckdb():
    """Check if DuckDB is available."""
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False
