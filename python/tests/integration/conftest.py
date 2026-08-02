# Integration test専用 fixture。実DB・外部ライブラリ依存を含む。
import pytest

# ============================================
# Discord 実送信ガード（integration テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_discord_http(monkeypatch):
    """Block real Discord HTTP calls from all integration tests.

    DISCORD_WEBHOOK_URL を除去することで _post_webhook が早期 return し、
    どのテストからも実際の webhook エンドポイントに到達しなくなる
    （tests/unit/conftest.py の同名フィクスチャと同じ理由）。

    Task 6 レビュー I-4: run_daily_pipeline() を呼ぶ integration テスト
    （tests/integration/test_scheduler_pipeline.py 等）は [6/6] 運用アラート
    評価の notifier=send_webhook_notification を patch していないため、
    ローカルに .env があると本物の Discord へ「🚨 運用アラート発報」等が
    送信されてしまう。CI は DISCORD_WEBHOOK_URL 未設定のため無害だが、
    ローカル実行の事故を構造的に防ぐため unit 側と同じガードを適用する。
    Discord 送信の有無を検証したいテストは @patch で個別に制御する。
    """
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)


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


@pytest.fixture(scope="session")
def _test_database_ready():
    from src.utils.db._connection import _get_pool
    from src.utils.db.migration_runner import run_migrations

    with _get_pool().connection() as con:
        run_migrations(con)
    yield


@pytest.fixture(autouse=True)
def _isolate_db(_test_database_ready):
    import psycopg

    from src.utils.data_path_utils import get_database_url
    from src.utils.db._connection import close_connection, set_test_connection

    con = psycopg.connect(get_database_url(), autocommit=False)
    # unit/conftest.py の _isolate_db と同じ理由でトランザクションを明示的に
    # 開始してから注入する（詳細はそちらのコメント参照）。
    con.execute("SELECT 1")
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        con.close()
        close_connection()
