# Unit test専用 fixture。外部DB・API依存は含まない。
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ============================================
# Discord 実送信ガード（unit テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_discord_http(monkeypatch):
    """Block real Discord HTTP calls from all unit tests.

    DISCORD_WEBHOOK_URL を除去することで _post_webhook が早期 return し、
    どのテストからも実際の webhook エンドポイントに到達しなくなる。
    Discord 送信の有無を検証したいテストは @patch で個別に制御する。
    """
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)


# ============================================
# healthchecks.io 実送信ガード（unit テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_heartbeat_ping(monkeypatch):
    """Block real healthchecks.io pings from all unit tests.

    HEALTHCHECKS_PING_KEY を空にすることで ping_heartbeat が早期 return し、
    本番 .env にキーが設定されていてもテスト実行が本番の監視 check を
    汚染しない（#548 と同型の「テストが本番リソースに到達する」事故の予防）。
    ping の有無を検証したいテストは settings を個別に monkeypatch する。
    """
    from config.settings import settings

    monkeypatch.setattr(settings, "HEALTHCHECKS_PING_KEY", "")


# ============================================
# Claude/外部API実呼び出しガード（unit テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_real_claude_calls(monkeypatch):
    """本番 .env で有効化されている *_ENABLED フラグが、ローカルの pytest 実行に
    漏れ込んで実際に Claude CLI/API を呼び出すことを防ぐ（#548と同型の
    「テストが本番リソースに到達する」事故の予防）。

    これらのフラグは各消費モジュールが `from config.settings import XXX` で
    モジュールレベル定数として束縛しているため、config.settings 側や
    config.feature_flags 側を書き換えても反映されない。消費先モジュール
    自身の名前空間を個別にFalseへ上書きする。
    個々のテストがこれらのフラグの挙動を検証したい場合は、当該テスト内で
    改めて `monkeypatch.setattr(...)` すればこの既定値を上書きできる。
    """
    monkeypatch.setattr("src.reporting.llm_review.LLM_REVIEW_ENABLED", False)
    monkeypatch.setattr("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", False)
    monkeypatch.setattr("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", False)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    monkeypatch.setattr("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    monkeypatch.setattr("src.quality.test_gap_review.TEST_GAP_ENABLED", False)
    # CLAUDE_TRADER_ENABLED・FACTORY_ENABLED は消費先（run_claude_trader.py /
    # run_nightly_strategy_factory()）が呼び出しの都度
    # `from config.settings import XXX` と遅延importするため、消費先モジュール
    # に固定の属性が存在しない。定義元（config.settings）側を上書きする。
    monkeypatch.setattr("config.settings.CLAUDE_TRADER_ENABLED", False)
    monkeypatch.setattr("config.settings.FACTORY_ENABLED", False)


# ============================================
# yfinance 実ネットワーク呼び出しガード（unit テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_real_yfinance_calls(monkeypatch):
    """Block real yfinance API calls from all unit tests.

    cProfile調査で判明: src.utils.sector_constraints.get_symbol_sector 等、
    yfinance の Ticker.info を呼ぶコードパスが一部の unit テストで未モックのまま
    実行され、実ネットワーク往復（Yahoo Finance のクッキー/crumb 認証を含む）で
    テストが数秒単位で遅くなっていた（#548 と同型の「テストが本番相当のリソースに
    到達する」事故の予防。Discord/healthchecks.io と異なりオフ用の環境変数が
    存在しないため、yfinance の Ticker.info プロパティ自体をブロックする）。

    このフィクスチャの主目的は実ネットワーク往復（＝テストの遅延）を防ぐことである。
    `yf.Ticker.info` プロパティをブロックし例外を送出するが、呼び出し元の実装次第では
    テスト失敗として顕在化しない点に注意。例えば
    `src.utils.sector_constraints.get_symbol_sector` は広い `except Exception` で
    この RuntimeError を握りつぶし、警告ログを出してフォールバック値を返す設計のため、
    このフィクスチャに引っかかってもテストは失敗せず黙って通ってしまう
    （＝「どのテストが何をモックし忘れているか」を明確なエラーで検知できる保証はない）。
    また `Ticker.history` や `get_earnings_dates` など `Ticker.info` を経由しない
    呼び出し経路は、そもそもこのフィクスチャの対象外である。同種の速度問題や
    モック漏れが他の経路で見つかった場合は、個別に追加のガードやモックが必要になる。
    実データが必要なテストは個別に `@patch("yfinance.Ticker.info", ...)` 等で
    上書きすること。
    """

    def _raise(self):
        raise RuntimeError(
            "unit テストから実 yfinance API (Ticker.info) が呼ばれました。"
            "呼び出し元のモックが不足しています。"
            "テスト側で該当関数（例: src.utils.sector_constraints.get_symbol_sector）を"
            "individually patch するか、yf.Ticker.info を個別にモックしてください。"
        )

    monkeypatch.setattr("yfinance.Ticker.info", property(_raise))


# ============================================
# トランザクションロールバックによる DB 隔離（unit テスト全体に適用）
# ============================================


@pytest.fixture(scope="session")
def _test_database_ready():
    """テストセッション開始時に1回だけ、テスト用Postgresへマイグレーションを適用する。

    DATABASE_URL 未設定時は get_database_url() の既定値（docker-compose.yml の
    `postgres-test` サービス、5433番ポート）を指す。本番の `postgres` サービス
    （5432番ポート）とは別ポート・別ボリュームで完全に分離されているため、
    同じホストで本番Postgresが稼働中でもテストが誤って到達することはない。
    CIでは services:postgres（ジョブごとに使い捨ての専用インスタンス）を指す。
    """
    from src.utils.db._connection import _get_pool

    with _get_pool().connection() as con:
        from src.utils.db.migration_runner import run_migrations

        run_migrations(con)
    yield


@pytest.fixture(scope="session")
def _shared_test_connection(_test_database_ready):
    """テストセッション全体で使い回す共有DB接続。

    従来は各テストで psycopg.connect() → rollback() → close() を繰り返しており、
    接続確立コスト（実測: 1回あたり中央値約23ms、2686件で合計約54秒）が
    unitテスト全体の実行時間の半分近くを占めていた。このフィクスチャで接続
    そのものはセッション中1本だけ張り、テストごとの分離は `_isolate_db` 側の
    「テストごとにトランザクションを開始してロールバックする」という既存の
    仕組みでそのまま担保する（分離の強度は変えず、接続確立の回数だけを減らす）。

    接続を使い回す副作用として、セッションレベルの状態（advisory lock、
    `LISTEN`、`SET`(非LOCAL)、`ON COMMIT DROP` の無い一時テーブル等）は
    テストをまたいで残り得る（従来の毎回close()する設計では自動的に消えていた）。
    現状このリポジトリでそのような状態を作るコードは `src/utils/db/_bulk.py`
    の一時テーブルのみで、そちらは使用前後で明示的に `DROP TABLE IF EXISTS`
    しており影響を受けない。今後セッションレベルの状態に依存するテストや
    フィクスチャを追加する場合は、この接続の使い回しを踏まえて後始末すること。
    """
    import psycopg

    from src.utils.data_path_utils import get_database_url

    con = psycopg.connect(get_database_url(), autocommit=False)
    yield con
    con.close()


@pytest.fixture(autouse=True)
def _isolate_db(_shared_test_connection):
    """全 unit テストを1トランザクションに包み、テスト終了時にロールバックする。

    _connection.py は呼び出し側が commit() しない設計のため、共有接続を
    そのままロールバックするだけで全ての書き込みを巻き戻せる
    （#548: 本番DB破損事故 / PR#556: filelock起因のCI一斉失敗、両方の
    事故クラスがこの方式では構造的に発生しなくなる）。

    接続自体は `_shared_test_connection`（セッションスコープ）で使い回し、
    ここではテストごとに「トランザクションの開始（SELECT 1）」と
    「ロールバック」だけを行う。
    """
    from src.utils.db._connection import close_connection, set_test_connection

    con = _shared_test_connection
    # 接続を明示的にトランザクション開始状態にしてから注入する。こうしないと
    # _connection.py 側で `with con.transaction():` のようなネスト保護を
    # 使った場合、テスト最初の呼び出しが「ネストではなく最外殻」と誤認されて
    # 誤ってCOMMITしてしまう（テスト分離が壊れ、本物のPostgresへ書き込みが
    # 漏れる）。SELECT 1 で毎テストの最初のトランザクションを確実に開始させておく。
    con.execute("SELECT 1")
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        close_connection()


# ============================================
# Mock オブジェクト Factory
# ============================================


@pytest.fixture
def mock_model_manager():
    """Fixture providing a mocked ModelManager."""
    mock = MagicMock()
    # predict は Series を返す
    mock.predict_with_model.return_value = [1.0, 0.5, -0.5, -1.0, 0.2]
    return mock


@pytest.fixture
def mock_signal_generator():
    """Fixture providing a mocked SignalGenerator."""
    mock = MagicMock()
    mock.generate.return_value = pd.Series([1, 0, -1, 0, 0])
    return mock


@pytest.fixture
def mock_data_loader():
    """Fixture providing a mocked DataLoader."""
    mock = MagicMock()
    # get_stock_data_auto は DataFrame を返す
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    mock.get_stock_data_auto.return_value = pd.DataFrame(
        {"Close": [100 + i for i in range(10)], "Volume": [1000 + i * 100 for i in range(10)]},
        index=dates,
    )
    return mock


# ============================================
# Backtester Factory
# ============================================


@pytest.fixture
def backtester_with_mocks(mock_model_manager, mock_signal_generator, mock_data_loader):
    """Backtester オブジェクトをモック依存で生成する fixture"""
    from src.backtest.backtester import Backtester

    return Backtester(
        model_manager=mock_model_manager,
        signal_generator=mock_signal_generator,
        data_loader=mock_data_loader,
        start_date="2024-01-01",
        end_date="2024-01-31",
        market="jp",
        symbol="7203",
        initial_cash=1_000_000,
        fee_rate=0.001,
        slippage=0.0,
    )
