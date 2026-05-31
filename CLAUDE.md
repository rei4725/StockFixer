# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**StockFixer** is a Python-based algorithmic stock trading system that fetches market data (yfinance), computes technical indicators, trains ML models (XGBoost/LightGBM), generates buy/sell/hold signals, executes paper or real trades (Kabu Station API), and reports results to Discord. Scheduled jobs run daily/weekly via APScheduler.

---

## Commands

All commands run from the `python/` directory unless noted. Use `py` (not `python`) on Windows.

### Setup
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
pre-commit install --hook-type commit-msg
```

### PR 前の一括チェック（CI と同等）
```powershell
# Windows
cd python; .\check-ci.ps1
```
```bash
# Linux/Mac
cd python && bash check-ci.sh
```
lint / mypy / pylint / import-linter / unit tests (cov≥80%) / bandit / pip-audit を順に実行する。
`bandit` と `pip-audit` は未インストール時はスキップ（`pip install bandit pip-audit` で有効化）。

### Lint & Format（個別実行）
引数の設定は `pyproject.toml` / `.flake8` が正本。コマンドに直接書かないこと。
```bash
cd python/
black .
isort .
flake8 .
mypy src/
pre-commit run --all-files
```

### Tests
```bash
# Unit tests with coverage gate (≥80% required — same as CI)
python -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80

# Integration tests (real DB/API, takes minutes)
python -m pytest tests/integration/ -v

# E2E tests (excluding slow)
python -m pytest tests/e2e/ -v -m "not slow"

# Single test
python -m pytest tests/unit/path/to/test_file.py::ClassName::test_method -v
```

### Run Entry Points
```bash
python run_data_creation.py --market us --symbol AAPL     # fetch + feature engineering
python run_model_creation.py --market us --symbol AAPL    # train per-symbol model
python run_unified_model_training.py                       # train all-symbols ensemble
python run_predict.py --mode single|watchlist|top10        # generate predictions
python run_backtest.py --market us --symbol AAPL --start 2024-01-01 --end 2025-01-01
python run_backtest_optimize.py --market us --symbol AAPL  # grid search optimization
python run_discord_bot.py
python run_scheduler.py --with-bot
python run_scheduler.py --run-now daily|weekly|optimization
```

### Docker
```bash
docker build -t stockfixer:dev ./python
docker-compose up -d
docker-compose logs -f stockfixer
```

---

## Architecture

### Layered Architecture (strict top→down dependencies only)

```
run_*.py          CLI entry points — argument parsing only, NO business logic
    ↓
api/              HTTP health endpoints
orchestration/    APScheduler wiring — calls into bounded contexts
    ↓
backtest/  prediction/  trading/  reporting/  watchlist/  market_data/
          Bounded contexts — each owns its own types.py and pipelines
    ↓
utils/            DB, logging, retry, path helpers
```

> **Legacy modules** (未整理 — 将来 BC に吸収予定):
> `src/data/`, `src/models/`, `src/analysis/`, `src/strategy/`, `src/features/`
> `src/brokers/` は `src/trading/brokers/` への後方互換 re-export shim — 削除予定

### Bounded Contexts (`src/<context>/`)

各 BC は `src/` 直下に配置され、それぞれ `types.py` を持つ:

| Context | 実パス | Responsibility |
|---|---|---|
| `backtest/` | `src/backtest/` | Backtesting pipelines |
| `prediction/` | `src/prediction/` | Forecast pipelines + ranking + model training |
| `trading/` | `src/trading/` | Order execution |
| `reporting/` | `src/reporting/` | Discord notifications |
| `watchlist/` | `src/watchlist/` | Symbol CRUD |
| `market_data/` | `src/market_data/` | yfinance data loading + feature engineering |
| `orchestration/` | `src/orchestration/` | Scheduler wiring (APScheduler) |

### Key Data Types (defined in each BC's `types.py`)

- `SymbolTask` — batch execution unit (market, symbol, horizon)
- `FeatureLoadResult` — feature loading outcome
- `TrainingMetrics` — model stats (RMSE, accuracy, sample count)
- `PredictionResult` — single stock prediction (price, change rate, timestamp)

### Broker Abstraction

`src/trading/brokers/base.py` defines `BrokerBase`. Two implementations:
- `trading/brokers/paper/` — DuckDB-backed paper trading (default)
- `trading/brokers/kabu/` — Kabu Station API for real trades

Brokers are injected into services; never referenced concretely from above layers.

### Pipelines

各 BC 配下に配置:
- `market_data/pipeline.py` — fetch → technical analysis → DuckDB
- `prediction/training_pipeline.py` — train per-symbol XGBoost/LightGBM
- `prediction/unified_model_pipeline.py` — train ensemble across all symbols
- `prediction/prediction_pipeline.py` — predict + rank top10/worst10
- `backtest/pipeline.py` — backtesting execution
- `orchestration/scheduler.py` — wires APScheduler daily/weekly jobs

### Storage

- **DuckDB**: `python/data/stockfixer.duckdb`
  - `stock_features` — (market, symbol, row_num) → OHLCV + indicators + lags
  - `prediction_results` — (run_timestamp, market, symbol) → predictions
- **Models**: `python/models/` — joblib files per symbol
- **Logs**: `Logs/` — rotating INFO + ERROR logs
- **Results**: `python/results/` — backtest CSVs, `optimal_params.json`

---

## Git Workflow

**メインブランチ**: `develop`（PR のベースブランチはすべて `develop`）

### 作業開始前（スキップ禁止）
```bash
git fetch
git status          # または git branch -vv
git pull            # ベースブランチが古い場合
```

### コミットメッセージ規約（Conventional Commits）
```
<type>: <subject>   # type: feat/fix/docs/refactor/test/chore
```
例: `feat: 統合モデル予測機能を追加`

### ブランチ命名
- `feature/機能名`, `fix/内容`, `refactor/対象`, `docs/対象`

### PR ボディ必須セクション（CI の `validate-pr-body` がチェック）

```markdown
## version_impact
minor          # major / minor / patch / none のいずれか1語

## version_rationale
（変更根拠を1文以上）

## VERSION 更新
- version_update_required: yes   # major/minor/patch → yes, none → no
- version_before: X.Y.Z
- version_after: X.Y.Z

## VERSION 未更新理由
（version_update_required: yes の場合は「該当なし」でも見出しは必須）
```

バージョン判定の正本: `docs/VERSIONING_POLICY.md`

### requirements*.txt 変更時（PR 前に必須）
```bash
pip install pip-audit
pip-audit -r requirements.txt
pip-audit -r requirements-dev.txt
```

---

## Key Rules

- **`run_*.py` files** are CLI wrappers only: parse args, call service layer, print results. No business logic, data transforms, or direct DB access.
- **Strict layering**: upper layers import lower ones, never the reverse. BC modules (`backtest/`, `prediction/` 等) do not import from `orchestration/` or `api/`.
- **Types over dicts**: use dataclass types from each BC's `types.py`, not raw dicts.
- **Logging**: all modules use `get_logger(__name__)` from the unified factory in `utils/`. Never use bare `except: pass` — always `logger.error("...", exc_info=True)`.
- **Time**: UTC internally, display as `Asia/Tokyo`.
- **DuckDB parallel writes are forbidden**: DB writes must be sequential. Use `get_readonly_connection()` for read-only access from separate processes.
- **Prohibited files** — never read or modify without explicit user confirmation: `.env`, `src/env/**`, `*/config/secrets.*`, `*.pem`, any file containing API keys or tokens.

---

## Configuration

All constants live in `python/config/settings.py` with environment variable overrides via `.env`:

| Variable | Default |
|---|---|
| `DISCORD_BOT_TOKEN` | — |
| `DISCORD_WEBHOOK_URL` | — |
| `LOG_LEVEL` | INFO |
| `MAX_DAILY_LOSS_RATE` | 2% |
| `MAX_POSITION_RATE` | 10% |
| `MAX_POSITIONS` | 10 |
| `BUY_THRESHOLD` | +0.5% |
| `SELL_THRESHOLD` | -0.5% |
| `PAPER_INITIAL_BALANCE` | ¥1,000,000 |
