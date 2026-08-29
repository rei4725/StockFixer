# テストスイート改善 Plan A（低リスク・即着手可能分）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** テストスイートの品質・速度を、ドメイン判断を要さない範囲で改善する。「開発ループを速くする」PR#666の続編にあたる。

**Architecture:** 3つの独立したタスク。(A) 実sleepに依存する脆いテストの排除＋開発者向けタスクランナースクリプトの追加。(B) 現在Postgres移行後の現実と乖離して機能していない`tests/integration/test_backtest_pipeline.py`をPostgres実データパスで書き直す。(C) unitテストの一部で発生している「実yfinanceネットワーク呼び出しによる異常な遅さ」を、系統的な安全網（autouseフィクスチャ）と具体的なモック追加の両輪で解消する。

**Tech Stack:** pytest, unittest.mock, psycopg（テスト分離）, PowerShell/bash

**Spec:** 本セッションでの直接調査結果（コード読解・実行・cProfileプロファイリング）。別ドキュメントなし、本プランが仕様を兼ねる。

## Global Constraints

- 既存テストの合格状態を絶対に壊さない（各ステップでローカル実行して確認する）
- `tests/e2e/conftest.py` には一切手を加えない（fixtureのライフサイクルが繊細なため、前回PR#666のレビューで touchしないと判断済み）
- 新しい抽象化（共有ファクトリ等）は導入しない。各タスクの変更はその場に閉じたローカルな修正に留める
- Postgresはローカルでは `docker compose up -d postgres-test`（ポート5433）で起動する。本番用 `postgres`（5432）を絶対に使わない

---

### Task A: 実sleep依存テストの解消 + テスト用タスクランナースクリプト追加

**Files:**
- Modify: `python/tests/unit/test_discord_rate_limiter.py`
- Create: `python/test.ps1`
- Create: `python/test.sh`
- Modify: `python/tests/README.md`

**Interfaces:** なし（テストコード・スクリプトのみ、srcへの変更なし）

**背景1（sleep除去）:** `DiscordRateLimiter`（`src/reporting/discord/rate_limiter.py`）は `time.monotonic()` を直接呼んでいる。既存テストの6メソッドが実際に `time.sleep(0.02)` 等で待ってTTL経過をシミュレートしており、CI負荷次第でマージンが不足しflaky化するリスクがある。`time.monotonic` をモックして即座にTTL経過を模擬する形に書き換える。

**背景2（タスクランナー）:** `make` はこのプロジェクトのWindows開発機に存在しない（確認済み）。既存の `check-ci.ps1`/`check-ci.sh` の二本立て流儀に合わせ、テスト層ごとの実行を簡略化する `test.ps1`/`test.sh` を追加する。

- [ ] **Step 1: `test_discord_rate_limiter.py` に FakeClock ヘルパーを追加し、実sleep依存の6メソッドを書き換える**

ファイル冒頭のimportブロックを以下に変更（`from unittest.mock import MagicMock, patch` は既存のまま維持し、下に追記）:

```python
"""ユニットテスト: Discord レート制限・デデュープ"""

from unittest.mock import MagicMock, patch

from src.reporting.discord.rate_limiter import DiscordRateLimiter


class _FakeClock:
    """time.monotonic() の代替。advance() で明示的に時刻を進める。"""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
```

`TestDedupeCache` クラス内の以下4メソッドを置き換える（`test_first_send_is_allowed`, `test_second_send_within_ttl_is_suppressed`, `test_suppression_counter_increments`, `test_different_messages_are_not_deduped` はsleepを使っていないため変更不要）:

```python
    def test_send_allowed_after_ttl_expires(self):
        clock = _FakeClock()
        with patch(
            "src.reporting.discord.rate_limiter.time.monotonic", side_effect=clock.monotonic
        ):
            limiter = DiscordRateLimiter(ttl=0.01)
            limiter.check_and_record("エラー")
            clock.advance(0.02)
            should_send, _ = limiter.check_and_record("エラー")
        assert should_send is True

    def test_suppression_summary_returned_after_ttl(self):
        clock = _FakeClock()
        with patch(
            "src.reporting.discord.rate_limiter.time.monotonic", side_effect=clock.monotonic
        ):
            limiter = DiscordRateLimiter(ttl=0.01)
            limiter.check_and_record("エラー")  # suppressed_count=0
            limiter.check_and_record("エラー")  # suppressed_count=1
            limiter.check_and_record("エラー")  # suppressed_count=2
            clock.advance(0.02)
            should_send, summary = limiter.check_and_record("エラー")
        assert should_send is True
        assert summary is not None
        assert "2" in summary

    def test_no_summary_when_no_suppressions_before_ttl(self):
        clock = _FakeClock()
        with patch(
            "src.reporting.discord.rate_limiter.time.monotonic", side_effect=clock.monotonic
        ):
            limiter = DiscordRateLimiter(ttl=0.01)
            limiter.check_and_record("エラー")
            clock.advance(0.02)
            should_send, summary = limiter.check_and_record("エラー")
        assert should_send is True
        assert summary is None

    def test_cache_reset_after_ttl(self):
        clock = _FakeClock()
        with patch(
            "src.reporting.discord.rate_limiter.time.monotonic", side_effect=clock.monotonic
        ):
            limiter = DiscordRateLimiter(ttl=0.01)
            limiter.check_and_record("エラー")
            clock.advance(0.02)
            limiter.check_and_record("エラー")
            key = limiter._hash_message("エラー")
        assert limiter._cache[key].suppressed_count == 0
```

`TestRateLimit` クラス全体を以下に置き換える:

```python
class TestRateLimit:
    def test_apply_rate_limit_sleeps_when_too_fast(self):
        clock = _FakeClock(start=1000.0)
        with (
            patch(
                "src.reporting.discord.rate_limiter.time.monotonic", side_effect=clock.monotonic
            ),
            patch("src.reporting.discord.rate_limiter.time.sleep") as mock_sleep,
        ):
            limiter = DiscordRateLimiter(rate_interval=0.1)
            limiter.apply_rate_limit()  # 初回: _last_send_time=0.0, now=1000.0 なので sleep しない
            clock.advance(0.02)  # 経過0.02秒 < rate_interval(0.1)
            limiter.apply_rate_limit()  # wait = 0.1 - 0.02 = 0.08 > 0 のため sleep が呼ばれるはず
        mock_sleep.assert_called_once()
        (wait_arg,), _ = mock_sleep.call_args
        assert wait_arg >= 0.05  # 少なくとも半分は待つ計算になっていること

    def test_no_sleep_when_interval_elapsed(self):
        clock = _FakeClock(start=1000.0)
        with (
            patch(
                "src.reporting.discord.rate_limiter.time.monotonic", side_effect=clock.monotonic
            ),
            patch("src.reporting.discord.rate_limiter.time.sleep") as mock_sleep,
        ):
            limiter = DiscordRateLimiter(rate_interval=0.01)
            limiter.apply_rate_limit()  # 初回: sleep しない
            clock.advance(0.02)  # rate_interval(0.01) より長く経過
            limiter.apply_rate_limit()  # wait = 0.01 - 0.02 < 0 のため sleep されないはず
        mock_sleep.assert_not_called()
```

`TestIntegrationWithSendWebhookNotification` クラスは変更不要（既にDiscordRateLimiterインスタンスごとモック済みでsleep非依存）。

- [ ] **Step 2: Step1の変更を実行して確認する**

Run: `cd python && py -m pytest tests/unit/test_discord_rate_limiter.py -v --durations=0`
Expected: 全件PASS。かつ `test_send_allowed_after_ttl_expires` 等の実行時間が(以前の0.02s実sleepではなく)ミリ秒未満になっていること。

- [ ] **Step 3: `python/test.ps1` を新規作成**

```powershell
param(
    [ValidateSet("unit", "integration", "e2e", "e2e-slow", "all")]
    [string]$Layer = "unit"
)

switch ($Layer) {
    "unit"        { py -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80 }
    "integration" { py -m pytest tests/integration/ -v --timeout=120 }
    "e2e"         { py -m pytest tests/e2e/ -v --timeout=60 -m "not slow" }
    "e2e-slow"    { py -m pytest tests/e2e/ -v --timeout=300 -m "slow" }
    "all"         { py -m pytest tests/unit/ tests/integration/ tests/e2e/ -v -m "not slow" }
}
```

- [ ] **Step 4: `python/test.sh` を新規作成**

```bash
#!/usr/bin/env bash
set -euo pipefail
LAYER="${1:-unit}"

case "$LAYER" in
  unit)        python -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80 ;;
  integration) python -m pytest tests/integration/ -v --timeout=120 ;;
  e2e)         python -m pytest tests/e2e/ -v --timeout=60 -m "not slow" ;;
  e2e-slow)    python -m pytest tests/e2e/ -v --timeout=300 -m "slow" ;;
  all)         python -m pytest tests/unit/ tests/integration/ tests/e2e/ -v -m "not slow" ;;
  *) echo "Usage: $0 {unit|integration|e2e|e2e-slow|all}" >&2; exit 1 ;;
esac
```

- [ ] **Step 5: 動作確認**

Run: `cd python && powershell -File test.ps1 unit`（Windowsの場合）または `cd python && bash test.sh unit`
Expected: unitテストが実行され2686件前後PASSする（Step1の変更でテスト数自体は変わらない）

- [ ] **Step 6: `python/tests/README.md` の「開発フロー推奨例」セクションに新スクリプトの案内を追記**

現在の開発フロー推奨例セクション（PR#666で書き換え済み）の末尾に、以下の注記ブロックを追加する:

```markdown

### ショートカットスクリプト

上記の各コマンドは `test.ps1` / `test.sh` でも実行できる:

```powershell
# Windows
.\test.ps1 unit          # unitテスト（カバレッジゲート付き）
.\test.ps1 integration   # integrationテスト
.\test.ps1 e2e           # e2e軽量部分（PRと同じ条件）
.\test.ps1 e2e-slow      # e2e重量級（develop push/手動発火と同じ条件、Postgres起動が必要）
```

```bash
# Linux/Mac
./test.sh unit
./test.sh integration
./test.sh e2e
./test.sh e2e-slow
```
```

- [ ] **Step 7: コミット**

```bash
cd python
git add tests/unit/test_discord_rate_limiter.py test.ps1 test.sh tests/README.md
git commit -m "test: rate_limiterテストの実sleep依存を解消しテスト用タスクランナースクリプトを追加"
```

---

### Task B: `tests/integration/test_backtest_pipeline.py` をPostgres実データパスで書き直す

**Files:**
- Modify: `python/tests/integration/test_backtest_pipeline.py`（全面書き換え）

**Interfaces:**
- Consumes: `tests/integration/conftest.py` の autouse fixture `_isolate_db`（各テスト関数ごとに独立したPostgresトランザクション+ロールバックを提供。追加の設定不要、既に全integrationテストに自動適用される）

**背景:** 現在このファイルは (1) ローカルにしか存在しない旧DuckDBファイル(`data/stockfixer.duckdb`、gitignore対象)を直接開いてテーブル存在確認をするテスト2件、(2) `run_backtest_single`/`run_backtest_walk_forward`を呼ぶが`set_model_manager_factory`の配線もテストデータの事前投入もしていないため実際には失敗する(実行して確認済み: 2 passed(旧DuckDBローカル依存)/1 failed(`model_manager_factory`未設定)/1 skipped)テスト2件、の計4件で構成されている。CI環境（旧DuckDBファイルが存在しない）では実質常にskip/failするテストであり、何の regression 検知にもなっていない。

`run_backtest_single`(`src/backtest/pipeline/runner.py`)は`load_features(market, symbol, source="file")`経由で**Postgresの`stock_features`テーブル**からデータを読み、内部でモデルを都度学習する（事前学習済みモデルは不要）。本番の起動経路（`run_backtest.py`, `run_backtest_portfolio.py`, `src/orchestration/scheduler.py`）はいずれも起動時に`set_model_manager_factory(ModelManager)`を呼んでおり、今回の`model_manager_factory`未設定エラーは**本番の欠陥ではなくこのテスト自身の準備不足**と確認済み。

このタスクでは、`tests/e2e/conftest.py`の`_generate_and_save_features`と同じ手順（合成OHLCV→技術指標→ラグ特徴量→`stock_features`へupsert）をこのファイル内にローカルヘルパーとして実装し、`set_model_manager_factory`を配線した上で、実際にバックテストパイプラインが実行できることを検証する形に書き直す。

- [ ] **Step 1: ファイル全体を以下の内容に置き換える**

`python/tests/integration/test_backtest_pipeline.py`:

```python
"""
Integration Test: バックテストパイプライン End-to-End

合成 OHLCV を Postgres の stock_features へ投入し、
run_backtest_single / run_backtest_walk_forward が実データパスで
完走することを検証する。
"""

import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_MARKET = "jp"
_SYMBOL = "TESTBT"
_N_DAYS = 200
_XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None


def _make_synthetic_ohlcv(n_days: int = _N_DAYS) -> pd.DataFrame:
    """n_days営業日分の固定OHLCV DataFrame（yfinance戻り値と同形式）を生成する。"""
    rng = np.random.default_rng(42)
    last_bday = pd.Timestamp.today().normalize()
    while last_bday.weekday() >= 5:
        last_bday -= pd.Timedelta(days=1)
    dates = pd.bdate_range(end=last_bday, periods=n_days)

    close = np.cumsum(rng.normal(0, 0.5, n_days)) + 100.0
    close = np.clip(close, 50.0, 300.0)

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.uniform(-0.005, 0.005, n_days)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "Close": close,
            "Volume": rng.integers(500_000, 2_000_000, n_days).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _seed_stock_features(market: str, symbol: str) -> None:
    """合成OHLCVを特徴量生成してstock_featuresへ保存する
    （tests/e2e/conftest.py の _generate_and_save_features と同等の手順）。
    """
    from src.market_data.saver import save_raw_ohlcv
    from src.market_data.technical import add_technical_indicators, create_basic_lag_features
    from src.utils.data_path_utils import normalize_col
    from src.utils.db import upsert_stock_features

    df = _make_synthetic_ohlcv()
    save_raw_ohlcv(market, symbol, df)

    work = df.copy()
    work = add_technical_indicators(work)
    if int(work.isnull().sum().sum()) > 0:
        work = work.ffill().bfill()

    X, y = create_basic_lag_features(work, n_lags=10)
    if X is None or X.empty:
        raise RuntimeError("テスト用特徴量の生成に失敗しました")

    X.columns = [normalize_col(c) for c in X.columns]
    data = X.copy()
    data["market"] = market
    data["symbol"] = symbol
    data["market_encoded"] = 1 if market == "jp" else 0
    data["y"] = y
    upsert_stock_features(market, symbol, data)


@unittest.skipUnless(_XGBOOST_AVAILABLE, "XGBoost not available")
class TestBacktestPipelineIntegration(unittest.TestCase):
    """バックテストパイプラインの Postgres 実データパス End-to-End テスト"""

    def setUp(self):
        from src.backtest.ports import set_model_manager_factory
        from src.prediction.manager import ModelManager

        set_model_manager_factory(ModelManager)
        _seed_stock_features(_MARKET, _SYMBOL)

    def test_backtest_single_runs_without_error(self):
        """単一期間バックテストが実行可能なことを確認"""
        from src.backtest.pipeline import run_backtest_single

        result_df, metrics, price_series = run_backtest_single(
            market=_MARKET,
            symbol=_SYMBOL,
            model_type="XGBoostModel",
            model_name="TestBacktestModel",
            task_name="return_regression",
            threshold=0.0,
            source="file",
            initial_cash=1_000_000,
            fee_rate=0.001,
            slippage=0.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            position_sizing="full",
            position_fraction=0.5,
            ensemble=False,
            start_date=None,
            end_date=None,
            train_ratio=0.8,
        )

        self.assertIsNotNone(metrics, "メトリクスが None でないこと")
        self.assertIn("final_cash", metrics, "final_cash メトリクスが存在すること")
        self.assertIn("total_return", metrics, "total_return メトリクスが存在すること")
        self.assertIn("sharpe_ratio", metrics, "sharpe_ratio メトリクスが存在すること")

    def test_backtest_walk_forward_runs_without_error(self):
        """Walk-Forward バックテストが実行可能なことを確認"""
        from src.backtest.pipeline import run_backtest_walk_forward

        _, _, wf_df = run_backtest_walk_forward(
            market=_MARKET,
            symbol=_SYMBOL,
            model_type="XGBoostModel",
            model_name="TestWalkForwardModel",
            task_name="return_regression",
            threshold=0.0,
            source="file",
            n_splits=3,
            initial_cash=1_000_000,
            fee_rate=0.001,
            slippage=0.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            position_sizing="full",
            position_fraction=0.5,
            ensemble=False,
        )

        self.assertIsNotNone(wf_df, "Walk-Forward 結果が None でないこと")
        self.assertGreater(len(wf_df), 0, "Walk-Forward 結果に最低1行以上のデータがあること")

        expected_cols = ["fold", "val_start", "val_end", "total_return", "sharpe_ratio"]
        for col in expected_cols:
            self.assertIn(col, wf_df.columns, f"{col} 列が存在すること")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Postgresを起動してテストを実行する**

Run:
```bash
cd python
docker compose up -d postgres-test
py -m pytest tests/integration/test_backtest_pipeline.py -v
```
Expected: `test_backtest_single_runs_without_error`, `test_backtest_walk_forward_runs_without_error` の2件がPASSする。xgboostが無い環境ではクラス全体がskipされる。

- [ ] **Step 3: 予期しない失敗が出た場合の切り分け**

`_seed_stock_features`実行後もデータ不足エラーが出る場合、`_N_DAYS`を200から400程度に増やして再実行する（train_ratio=0.8分割後もtrain/test双方に十分な行数を残すため）。それでも解決しない場合はエラーメッセージ全文をBLOCKEDとして報告する。

- [ ] **Step 4: integration層全体を実行し、他のテストへ悪影響が無いことを確認**

Run: `py -m pytest tests/integration/ -v --timeout=120`
Expected: 121件（+今回書き直した2件はそのまま数に含まれる）全てPASS

- [ ] **Step 5: コミット**

```bash
git add tests/integration/test_backtest_pipeline.py
git commit -m "test: test_backtest_pipeline.pyを旧DuckDB依存からPostgres実データパスへ書き直し"
```

---

### Task C: unitテストにおける実yfinanceネットワーク呼び出しの遮断 + 該当テストの修正

**Files:**
- Modify: `python/tests/unit/conftest.py`
- Modify: `python/tests/unit/test_order_execution_pipeline.py`

**Interfaces:** なし

**背景（cProfileで実測・特定済み）:** `test_order_execution_pipeline.py::TestRunDailyOrders::test_max_orders_per_run_respected`（4.98s）を`cProfile`で解析したところ、`curl_cffi._wrapper.curl_easy_perform`が33回・累計8.17秒消費していた。呼び出し元を`pstats.print_callers`で追跡した結果:

```
curl_easy_perform ← session.py:_request_once ← session.py:request ← session.py:get
  ← yfinance/data.py:_make_request ← ... ← yfinance/scrapers/quote.py:info
  ← yfinance/base.py:get_info
```

`src/utils/sector_constraints.py:28`に`info = yf.Ticker(ticker).info or {}`という実yfinance呼び出しがあり、これは`src/trading/execution/selection.py`（買い候補選定ロジック、`run_daily_orders`が内部で呼ぶ）から到達する。このテストファイルの`_patch_pipeline`ヘルパー（8個のpatchを持つ）にはこの呼び出し経路のモックが含まれておらず、買い候補数(`n_buy`)に応じて実際にYahoo Financeへ複数回アクセスしてしまっていた。

同種の問題は`test_model_training_pipeline_unit.py::TestLoadFeaturesForTraining::test_exclude_cols_not_in_X`（2.83s）でも確認済み（同じく`curl_cffi`が2回・累計2.2秒）。「unitテストなのに実ネットワークに到達する」という同型の問題が複数ファイルで独立に発生していることから、個別ファイルへのモック追加に加えて、**unitテスト全体に適用される安全網**を追加する（`tests/unit/conftest.py`には既に`_block_discord_http`/`_block_heartbeat_ping`/`_block_real_claude_calls`という同種の「本番リソースへの意図しない到達を防ぐ」autouseフィクスチャが存在し、その並びに追加する形になる）。

なお、`test_strategy_factory.py`の3件の遅いテスト（1.8〜1.9秒）は別の原因（`_PASSING_DATA_PERIODS = 700`という意図的にチューニングされた定数を使う現実的な規模のpandas演算）であり、値を減らすとテストの意図（「合格する」ケースの再現）が壊れる可能性があるため本タスクの対象外とする。

- [ ] **Step 1: `tests/unit/conftest.py` に実yfinance呼び出しを遮断するautouseフィクスチャを追加**

`tests/unit/conftest.py`の`_block_real_claude_calls`フィクスチャ（72行目付近）の直後に以下を追加する:

```python
# ============================================
# yfinance 実ネットワーク呼び出しガード（unit テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_real_yfinance_calls(monkeypatch):
    """unit テストから実 yfinance API への到達を防ぐ。

    cProfile調査で判明: src.utils.sector_constraints.get_symbol_sector 等、
    yfinance の Ticker.info を呼ぶコードパスが一部の unit テストで未モックのまま
    実行され、実ネットワーク往復（Yahoo Finance のクッキー/crumb 認証を含む）で
    テストが数秒単位で遅くなっていた（#548 と同型の「テストが本番相当のリソースに
    到達する」事故の予防。Discord/healthchecks.io と異なりオフ用の環境変数が
    存在しないため、yfinance の Ticker.info プロパティ自体をブロックする）。

    このフィクスチャに引っかかった場合は「どのテストが何をモックし忘れているか」を
    示す明確なエラーとして失敗させる（無言でNoneやfalsy値を返すと問題を隠してしまう
    ため、意図的に例外を送出する設計にしている）。実データが必要なテストは
    個別に `@patch("yfinance.Ticker.info", ...)` 等で上書きすること。
    """

    def _raise(self):
        raise RuntimeError(
            "unit テストから実 yfinance API (Ticker.info) が呼ばれました。"
            "呼び出し元のモックが不足しています。"
            "テスト側で該当関数（例: src.utils.sector_constraints.get_symbol_sector）を"
            "individually patch するか、yf.Ticker.info を個別にモックしてください。"
        )

    monkeypatch.setattr("yfinance.Ticker.info", property(_raise))
```

- [ ] **Step 2: フィクスチャ追加後にunitテスト全体を実行し、新たに落ちるテストを洗い出す**

Run: `cd python && py -m pytest tests/unit/ -q 2>&1 | tail -60`
Expected: `test_order_execution_pipeline.py`と`test_model_training_pipeline_unit.py`の該当テストが、今度は「8秒待って偶然pass」ではなく「即座にRuntimeErrorでfail」に変わるはず。他に同じ問題を抱えていたテストがあれば、ここで新たに失敗として顕在化する。**失敗したテストの一覧をすべてメモする**（Step 3で使う）。

- [ ] **Step 3: Step 2で失敗した各テストに、該当箇所のモックを追加する**

`test_order_execution_pipeline.py`の`_patch_pipeline`メソッド（930行目付近）のpatchリストに以下を追加する:

```python
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                return_value="Unknown",
            ),
```

（既存の8個のpatchのリストの末尾、`patch("src.trading.execution.runner.save_order_run_summary")`の後にカンマ区切りで追加する）

Step 2で`test_model_training_pipeline_unit.py`や他のファイルにも同様の失敗が出た場合、各ファイルの該当テスト（または共通のsetUp/patchヘルパー）に、Step2の失敗メッセージが示す実際の呼び出し元関数を個別にpatchで追加する。呼び出し元の特定に迷う場合は、このタスクのブリーフに記載した`cProfile`の手順（`python -m cProfile -o prof.out -m pytest <対象テスト> -q` → `pstats.Stats('prof.out').print_callers('...')`を末端の`yfinance`関数から遡る）と同じ方法で特定する。

- [ ] **Step 4: 全件PASSを確認し、実行時間が改善したことを確認**

Run: `cd python && py -m pytest tests/unit/ --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80 --durations=10 -q`
Expected: 2686件（Task B/他タスクでテスト数が変動していなければ）全てPASS、カバレッジ80%以上維持。`--durations=10`の一覧から`curl_easy_perform`起因の秒単位の遅延が消えていること（`test_max_orders_per_run_respected`が数十ms〜数百ms程度になっていることが目安）。

- [ ] **Step 5: コミット**

```bash
git add tests/unit/conftest.py tests/unit/test_order_execution_pipeline.py
git commit -m "test: unitテストから実yfinance呼び出しを遮断するガードを追加し該当テストのモック漏れを修正"
```

（Step3で他ファイルも修正した場合は、それらのファイルも`git add`に含める）

---

## 最終確認

- [ ] Task A〜C のコミットを積んだブランチで以下を実行し、全体に問題が無いことを確認する:
  - `cd python && py -m black . --check && py -m isort . --check && py -m flake8 .`
  - `cd python && py -m pytest tests/unit/ --cov=src --cov-branch --cov-fail-under=80`
  - `cd python && docker compose up -d postgres-test && py -m pytest tests/integration/ --timeout=120`
- [ ] PR本文の `version_impact` は `none`（テスト・スクリプトのみでプロダクトコード非変更）とする
