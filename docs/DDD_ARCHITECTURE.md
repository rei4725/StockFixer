# DDD アーキテクチャ移行計画

> 作成日: 2026-04-27  
> ステータス: **Accepted**（移行進行中）  
> 関連ドキュメント: [ARCHITECTURE.md](ARCHITECTURE.md) / [ROADMAP_IDEAS.md](ROADMAP_IDEAS.md)

---

## 目次

1. [なぜ DDD を採用するか](#1-なぜ-ddd-を採用するか)
2. [現状アーキテクチャの課題](#2-現状アーキテクチャの課題)
3. [Bounded Context の定義](#3-bounded-context-の定義)
4. [ターゲット構成（As-Should）](#4-ターゲット構成as-should)
5. [BC 間の依存ルール](#5-bc-間の依存ルール)
6. [移行スケジュール](#6-移行スケジュール)
7. [移行時の注意事項](#7-移行時の注意事項)
8. [ADR（アーキテクチャ決定記録）](#8-adrアーキテクチャ決定記録)

---

## 1. なぜ DDD を採用するか

現在の StockFixer は **技術レイヤー分割**（data / features / models / strategy / services / utils）を基本としているが、
機能追加・修正の際に複数のレイヤーにまたがる変更が常態化しており、以下の問題が顕在化している。

| 問題 | 具体例 |
|------|--------|
| 認知負荷が高い | 「予測機能」を修正するために `models/`・`features/`・`data/`・`services/prediction/`・`domain/types.py` の 5 箇所を確認する必要がある |
| 型定義の神ファイル化 | `domain/types.py` 1 ファイルにシステム全体の型が集中し、変更影響範囲が広い |
| レイヤーとドメインの混在 | `services/` だけが先行してドメイン別サブパッケージになっており、他の層と不整合がある |
| ユーティリティ層の肥大化 | `utils/yf_client.py`（市場データ取得）や `utils/optimal_params_loader.py`（戦略設定）が技術的無関係なものと同居 |
| BC 間境界の不明確さ | `brokers/` と `services/trading/` が分断されており、取引ドメインの責務が 2 層に散在する |

DDD の **Bounded Context（境界付きコンテキスト）** によりドメイン単位でコードを凝集させ、
「ある機能を追加・修正するとき、変更が 1 つの BC 内で完結する」状態を目指す。

---

## 2. 現状アーキテクチャの課題

### 2.1 構造図（現状 As-Is）

```
src/
├── api/              ← Discord 通知（表示層）
├── backtest/         ← backtester.py のみ（実行エンジン）
├── brokers/          ← kabu/, paper/（証券会社実装）
├── data/             ← loader, saver
├── domain/           ← types.py（全型を 1 ファイルに）
├── features/         ← technical_analysis.py, market_regime.py
├── models/           ← ML モデル実装・管理
├── strategy/         ← signal_generator.py のみ
├── utils/            ← logger, db, japan_time, yf_client, optimal_params_loader, ...（雑多）
└── services/         ← 半ドメイン化（フラットファイル + サブパッケージ混在）
    ├── backtest/
    ├── prediction/
    ├── reporting/
    ├── trading/
    ├── training/
    └── watchlist/
```

### 2.2 ファイルの現在位置と「あるべき BC」の対応

| 現在の場所 | ファイル | あるべき BC |
|-----------|---------|------------|
| `domain/types.py` | `SymbolTask` | `watchlist` |
| `domain/types.py` | `FeatureLoadResult` | `analysis` |
| `domain/types.py` | `TrainingMetrics` | `prediction` |
| `domain/types.py` | `PredictionResult` | `prediction` |
| `utils/yf_client.py` | yfinance ラッパー | `market_data` |
| `utils/optimal_params_loader.py` | 最適パラメータ読込 | `strategy` |
| `brokers/` | 証券会社実装 | `trading` |
| `backtest/backtester.py` | バックテスト実行エンジン | `backtest` |
| `models/` | ML モデル実装 | `prediction` |
| `api/` | Discord 表示 | `reporting` |

---

## 3. Bounded Context の定義

StockFixer の業務ドメインを以下の **8 つの Bounded Context** に分割する。

| BC | 責務 | 主なエンティティ |
|----|------|----------------|
| `shared` | 横断インフラ（DB・ロガー・時刻変換）。他の BC から参照されるが、BC 業務ロジックを持たない | — |
| `market_data` | 市場データの取得・保存・提供 | `OHLCVRecord`, `MarketDataQuery` |
| `analysis` | テクニカル指標計算・特徴量生成・マーケットレジーム判定。複数 BC が依存する**共有カーネル** | `FeatureLoadResult` |
| `prediction` | AI モデルの学習・予測。結果を `PredictionResult` として提供 | `PredictionResult`, `TrainingMetrics` |
| `strategy` | 予測結果とテクニカル分析を組み合わせた売買シグナル生成 | `SignalResult` |
| `backtest` | 戦略パラメータの検証・最適化・Walk-Forward・ストレステスト | `BacktestResult`, `OptimizationResult` |
| `trading` | 注文実行・リスク管理・ポジション管理（PaperBroker / KabuBroker） | `Order`, `Position`, `TradeResult` |
| `watchlist` | 監視対象銘柄の管理・バッチ実行 | `WatchlistItem`, `SymbolTask` |
| `reporting` | Discord Bot・月次レポート・通知（読取専用。他 BC の結果を表示するのみ） | `ReportResult` |

### 共有カーネルの扱い

`analysis`（特徴量）は `prediction`・`backtest`・`strategy` の複数 BC が依存するため、
完全な独立 BC にはせず **共有カーネル（Shared Kernel）** として扱う。
変更は関係 BC の合意の上で行い、`CHANGELOG` に記録する。

---

## 4. ターゲット構成（As-Should）

```
python/src/
│
├── shared/                          # 横断インフラ（依存方向: なし）
│   ├── db/                          # ← utils/db/
│   ├── logger.py                    # ← utils/logger.py
│   ├── japan_time.py                # ← utils/japan_time.py
│   ├── retry_helper.py              # ← utils/retry_helper.py
│   └── data_path_utils.py           # ← utils/data_path_utils.py
│
├── market_data/                     # [BC] 市場データ
│   ├── types.py                     # OHLCVRecord, MarketDataQuery
│   ├── loader.py                    # ← data/data_loader.py
│   ├── saver.py                     # ← data/data_saver.py
│   ├── yf_client.py                 # ← utils/yf_client.py
│   └── pipeline.py                  # ← services/data_pipeline.py
│
├── analysis/                        # [BC / 共有カーネル] テクニカル分析・特徴量
│   ├── types.py                     # FeatureLoadResult（← domain/types.py より移動）
│   ├── technical.py                 # ← features/technical_analysis.py
│   └── market_regime.py             # ← features/market_regime.py
│
├── prediction/                      # [BC] AI 予測
│   ├── types.py                     # PredictionResult, TrainingMetrics（← domain/types.py より移動）
│   ├── models/                      # ← models/
│   │   ├── base.py                  # ← models/base_model.py
│   │   ├── xgboost.py               # ← models/xgboost_model.py
│   │   └── lightgbm.py              # ← models/lightgbm_model.py
│   ├── manager.py                   # ← models/model_manager.py
│   ├── predict_single.py            # ← models/predict_single_stock.py
│   ├── predict_unified.py           # ← models/predict_unified.py
│   ├── training_pipeline.py         # ← services/training/model_training_pipeline.py
│   ├── prediction_pipeline.py       # ← services/prediction/prediction_pipeline.py
│   ├── unified_model_pipeline.py    # ← services/prediction/unified_model_pipeline.py
│   └── shadow_evaluation.py         # ← services/prediction/shadow_evaluation_pipeline.py
│
├── strategy/                        # [BC] シグナル生成
│   ├── types.py                     # SignalResult
│   ├── signal_generator.py          # ← strategy/signal_generator.py
│   └── optimal_params_loader.py     # ← utils/optimal_params_loader.py
│
├── backtest/                        # [BC] バックテスト
│   ├── types.py                     # BacktestResult, OptimizationResult
│   ├── backtester.py                # ← backtest/backtester.py
│   ├── optimizer.py                 # ← services/backtest/backtest_optimize_pipeline.py
│   ├── pipeline.py                  # ← services/backtest/backtest_pipeline.py
│   ├── portfolio.py                 # ← services/backtest/portfolio_backtest.py
│   ├── stress_test.py               # ← services/backtest/stress_test_pipeline.py
│   └── walk_forward.py              # ← services/backtest/walk_forward_report_pipeline.py
│
├── trading/                         # [BC] 取引実行
│   ├── types.py                     # Order, Position, TradeResult
│   ├── brokers/                     # ← brokers/（まるごと移動）
│   │   ├── base.py
│   │   ├── kabu/
│   │   └── paper/
│   ├── risk_manager.py              # ← services/trading/risk_manager.py
│   └── execution.py                 # ← services/trading/order_execution_pipeline.py
│
├── watchlist/                       # [BC] ウォッチリスト
│   ├── types.py                     # WatchlistItem, SymbolTask（← domain/types.py より移動）
│   ├── manager.py                   # ← services/watchlist/watchlist_manager.py
│   └── batch_runner.py              # ← services/batch_runner.py
│
├── reporting/                       # [BC] レポーティング・通知（読取専用）
│   ├── types.py                     # ReportResult
│   ├── discord/                     # ← api/（まるごと移動）
│   │   ├── bot.py
│   │   ├── formatters.py
│   │   ├── text.py
│   │   └── utils.py
│   ├── monthly.py                   # ← services/reporting/monthly_report_pipeline.py
│   └── query_service.py             # ← services/reporting/discord_query_service.py
│
└── orchestration/                   # スケジューラ・バッチ統括（最上位）
    ├── scheduler.py                 # ← services/scheduler_pipeline.py
    └── scheduler_queue.py           # ← services/scheduler_queue.py
```

### 移行後のレイヤー図

```
┌──────────────────────────────────────────────────────────────────┐
│  run_*.py  （CLI ラッパー・エントリーポイント）                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  orchestration/  （スケジューラ・バッチ統括）                    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────┬────────────┬────────────┬───────────┬──────────────┐
│ prediction │  backtest  │  trading   │ watchlist │  reporting   │
│     BC     │     BC     │     BC     │    BC     │     BC       │
└────────────┴────────────┴────────────┴───────────┴──────────────┘
                              ↓
┌───────────────────┬────────────────────────────────────────────┐
│    strategy BC    │            analysis BC（共有カーネル）      │
└───────────────────┴────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  market_data BC  （データ取得・保存）                            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  shared  （DB・ロガー・時刻変換・リトライ）                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. BC 間の依存ルール

### 許可される依存方向

```
reporting   → prediction, backtest, trading, watchlist, strategy
prediction  → analysis, market_data, shared
strategy    → analysis, shared
backtest    → strategy, analysis, market_data, shared
trading     → strategy, shared
watchlist   → shared
market_data → shared
analysis    → shared（共有カーネル）
orchestration → すべての BC（統括）
```

### 禁止事項

- **下位 BC から上位 BC への逆参照禁止**（例: `market_data` から `prediction` への参照）
- **BC をまたいだ内部モジュール直接参照禁止**（各 BC の `__init__.py` 経由のみ）
- **`shared` への業務ロジック混入禁止**（`shared` は純粋なインフラ）

### ドメイン型のルール

- 各 BC の `types.py` は当該 BC 内でのみ定義・管理する
- 他 BC の型を参照する場合は `from src.prediction.types import PredictionResult` のように BC 名を含む完全パスで import する
- 後方互換のために `domain/types.py` は移行期間中は re-export モジュールとして維持する

---

## 6. 移行スケジュール

移行は **4 フェーズ** に分け、各フェーズで全テスト（unit + integration）がグリーンであることを確認してから次フェーズに進む。

### フェーズ 0: 現状整理（即時着手可能 / リスク: 低）

**目的**: 移行の土台を整える。コードは動かさない。

| タスク | 内容 | 担当 BC |
|--------|------|---------|
| 0-1 | ARCHITECTURE.md を現状に追従させる | — |
| 0-2 | DDD_ARCHITECTURE.md を作成し移行方針を合意する（本ドキュメント） | — |
| 0-3 | ROADMAP_IDEAS.md に NF-601〜NF-604（DDD移行）を追加する | — |
| 0-4 | `domain/types.py` の各型を「あるべき BC」とコメントでマッピングする | — |

完了条件: 本ドキュメントがレビューされ、チームで方針合意済みであること

---

### フェーズ 1: 型の分散・utils 整理（低リスク）

**目的**: 神ファイル `domain/types.py` の解体と、utils 層の肥大化解消

| タスク | 移動元 | 移動先 | 影響範囲 |
|--------|--------|--------|---------|
| 1-1 | `utils/yf_client.py` | `market_data/yf_client.py` | `data/data_loader.py` の import 1 箇所 |
| 1-2 | `utils/optimal_params_loader.py` | `strategy/optimal_params_loader.py` | `services/trading/` の import 数箇所 |
| 1-3 | `domain/types.py` の `SymbolTask` | `watchlist/types.py` に移動 + `domain/types.py` で re-export | 全 batch_runner 呼び出し箇所 |
| 1-4 | `domain/types.py` の `FeatureLoadResult` | `analysis/types.py` に移動 + `domain/types.py` で re-export | `services/training/` と `models/` の import |
| 1-5 | `domain/types.py` の `TrainingMetrics` / `PredictionResult` | `prediction/types.py` に移動 + `domain/types.py` で re-export | 全 prediction 系の import |

**実施方針**:
- 新パスに `types.py` を作成し定義を移動する
- `domain/types.py` は移行完了まで `from src.xxx.types import YYY` の re-export のみとする
- `domain/types.py` の re-export で既存コードへの影響ゼロを保証する

完了条件: `python -m pytest tests/unit/ tests/integration/ -v` が全グリーン

---

### フェーズ 2: BC 境界の確立（中リスク）

**目的**: `brokers/` と `backtest/` の BC 統合。`services/` の二重管理解消。

| タスク | 移動元 | 移動先 | 影響範囲 |
|--------|--------|--------|---------|
| 2-1 | `brokers/` | `trading/brokers/` | `services/trading/` の import |
| 2-2 | `backtest/backtester.py` | `backtest/backtester.py`（フォルダを BC ルートに格上げ） | `services/backtest/` の import |
| 2-3 | `features/technical_analysis.py` | `analysis/technical.py` | 多数（テスト含め 20+ 箇所） |
| 2-4 | `features/market_regime.py` | `analysis/market_regime.py` | `services/prediction/` の import |
| 2-5 | `api/` | `reporting/discord/` | `run_discord_bot.py` の import |

**実施方針**:
- 1 タスクずつ実施し、都度 `pytest` を実行して確認する
- `features/` と `api/` は import 箇所が多いため、移動後に `features/__init__.py` / `api/__init__.py` で旧パスの re-export を残す（フェーズ 3 で削除）

完了条件: `python -m pytest tests/unit/ tests/integration/ -v` が全グリーン

---

### フェーズ 3: 大規模再構成（高リスク・計画的に実施）

**目的**: `models/` を `prediction/` に吸収し、全 BC が独立して完結する状態にする。

| タスク | 移動元 | 移動先 | 影響範囲 |
|--------|--------|--------|---------|
| 3-1 | `models/` | `prediction/models/` | テスト含め最多（全 model 系テスト） |
| 3-2 | `data/` | `market_data/`（フォルダ rename） | `services/data_pipeline.py` 等 |
| 3-3 | `services/data_pipeline.py` | `market_data/pipeline.py` | `run_data_creation.py` |
| 3-4 | `services/training/` | `prediction/training_pipeline.py` | `run_model_creation.py` |
| 3-5 | `services/prediction/` | `prediction/` 配下に統合 | `run_predict.py` |
| 3-6 | `services/scheduler_pipeline.py` | `orchestration/scheduler.py` | `run_scheduler.py` |
| 3-7 | `services/batch_runner.py` | `watchlist/batch_runner.py` | 全 batch 系 run_*.py |

**実施方針**:
- このフェーズは 1 タスクを 1 PR として分割し、リビューを挟む
- `models/` → `prediction/models/` は最も影響範囲が大きいため、移動直後の全テストを必須とする
- `data/` → `market_data/` の rename は sed/grep による一括置換と `pytest --tb=short` での確認を組み合わせる

完了条件: `python -m pytest tests/ -v` が全グリーン かつ `python run_scheduler.py --run-now daily` が正常完了

---

### フェーズ 4: 仕上げ（import パス統一・ドキュメント整合）

**目的**: re-export 互換パスを削除し、最終形を確定する。

| タスク | 内容 |
|--------|------|
| 4-1 | `domain/types.py` の re-export を削除し、直接 import に一括変換 |
| 4-2 | フェーズ 2 で残した `features/__init__.py` / `api/__init__.py` の re-export を削除 |
| 4-3 | `domain/` フォルダを削除（または空の `__init__.py` のみ残留） |
| 4-4 | copilot-instructions.md のディレクトリ構成・レイヤー構造を最終形に更新 |
| 4-5 | ARCHITECTURE.md を最終形に更新（DDD_ARCHITECTURE.md の内容を統合） |
| 4-6 | 全 linter（flake8・mypy）がクリーンであることを確認 |

完了条件: `python -m pytest tests/ -v` 全グリーン + flake8 / mypy エラーゼロ

---

## 7. 移行時の注意事項

### テスト戦略

- **フェーズ 0〜1**: `tests/unit/` のみでの確認で十分
- **フェーズ 2 以降**: `tests/integration/` も必須（DB・ファイル I/O が絡む移動が発生するため）
- import パスの変更は `grep -r "from src.XXX" python/` で影響箇所を事前把握してから実施する

### import パス変換コマンド（参考）

```powershell
# フェーズ 2: features → analysis への一括変換例
Get-ChildItem -Recurse -Filter "*.py" python/ | ForEach-Object {
    (Get-Content $_.FullName) -replace 'from src\.features\.', 'from src.analysis.' |
    Set-Content $_.FullName
}
```

### ロールバック戦略

- 各フェーズは専用ブランチ（`feature/ddd-phase-N-xxx`）で実施し、PR マージ前にテスト確認を必須とする
- フェーズをまたいで作業しない（フェーズ N 完了 → develop マージ → フェーズ N+1 開始）

### `sector_constraints.py` の扱い

`utils/sector_constraints.py` は trading BC と strategy BC の両方に関係するため、
フェーズ 2 での移動先は `trading/sector_constraints.py` とし、
`strategy` からは `trading` への依存ではなく `shared` への移動を検討する（フェーズ 2 着手時に再判断）。

---

## 8. ADR（アーキテクチャ決定記録）

### ADR-001: DDD Bounded Context 採用

- **決定日**: 2026-04-27
- **ステータス**: Accepted
- **コンテキスト**: 技術レイヤー分割により機能追加時の変更範囲が広く、認知負荷が高い
- **決定**: DDD の Bounded Context を採用し、ドメイン単位で凝集したフォルダ構成に移行する
- **結果**: 機能単位で変更が 1 BC に閉じるようになり、テスト・レビューの単位が明確になる
- **トレードオフ**: import パス変更コストが高い。段階移行（フェーズ 0〜4）で対処する

### ADR-002: `analysis` を共有カーネルとして維持

- **決定日**: 2026-04-27
- **ステータス**: Accepted
- **コンテキスト**: 特徴量生成は `prediction`・`backtest`・`strategy` の 3 BC が依存する
- **決定**: `analysis` BC を独立 BC にせず共有カーネルとして扱い、全 BC から参照可能にする
- **結果**: BC 間の間接層が不要になりシンプルになる
- **トレードオフ**: `analysis` の変更が複数 BC に影響する。変更時は全依存 BC のテストを実行する

### ADR-003: `domain/types.py` は移行期間中 re-export モジュールとして維持

- **決定日**: 2026-04-27
- **ステータス**: Accepted
- **コンテキスト**: 型定義を各 BC に移動するとすべての import が壊れる
- **決定**: 型を各 BC の `types.py` に移動したうえで、`domain/types.py` は re-export のみとする。フェーズ 4 で削除する
- **結果**: 既存コードへの影響ゼロで段階移行できる
- **トレードオフ**: フェーズ 4 まで二重管理になる

### ADR-004: `shared` は純粋なインフラとし業務ロジックを持たない

- **決定日**: 2026-04-27
- **ステータス**: Accepted
- **コンテキスト**: `utils/` が DB・ロガー・時刻変換・yf_client・optimal_params_loader を混在させている
- **決定**: `utils/` → `shared/` に rename し、業務知識を持つ `yf_client.py` / `optimal_params_loader.py` は各 BC に移動する
- **結果**: `shared` が純粋な技術インフラになり、どの BC からも安全に参照できる
- **トレードオフ**: フェーズ 1 で実施するため、影響は限定的
