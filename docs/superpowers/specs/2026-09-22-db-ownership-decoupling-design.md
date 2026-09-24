# DB 所有権の疎結合化 設計書（Phase 4）

**作成日:** 2026-09-22
**ステータス:** Accepted
**対象バージョン:** 2.14.2 → 2.15.0（minor）
**関連:** [実行モデル Phase 3 設計](2026-09-21-execution-model-design.md) / `.importlinter` / `docs/DDD_ARCHITECTURE.md`

---

## 1. 背景

Phase 3（screening ポート結線）の完了後、次の疎結合化候補を調査したところ、
`src/utils/db/__init__.py` が **文字列ベースの動的 import で BC 間の依存を隠している** ことが判明した。

```python
# src/utils/db/__init__.py:138, 154
_pred_db = importlib.import_module("src.prediction.db")
return getattr(_pred_db, name)
```

`_DbPackageProxy` が 18 個の関数名を `src.prediction.db` から遅延取得して再輸出している。
同ファイル 95 行のコメントは「モジュールレベルの `from src.prediction.db import ...` を避けることで
importlinter の BC 間接依存検出を回避する」と、**契約を通すために意図的に逃がしたことを自認している**。

### 何が問題か

1. **import-linter は文字列 import を検出できない。** 契約は緑だが依存は存在する。
2. **層が逆転している。** `src.utils` は最下層であり、BC を参照してはならない。
3. **ロンダリングとして機能している。** `src/reporting/*` は `utils.db` 経由でのみ prediction に触れており、
   BC 独立性契約（Contract 2）を迂回している。
4. **契約オプションが無効化されている。** `allow_indirect_imports = True` はこの再輸出が理由。
   間接依存が一切検査されていない。

---

## 2. 調査で判明した事実

### 2.1 抜け道は 1 ファイル・1 機構のみ

`src/` 配下の `importlib.import_module` 使用箇所は `src/utils/db/__init__.py` の 2 行だけである。
他に同種の迂回は存在しない。

### 2.2 `src/prediction/db/` の中身は BC 依存が薄い

| モジュール | 行数 | BC への依存 |
|---|---|---|
| `accuracy.py` | 262 | なし（`src.utils` のみ） |
| `features.py` | 229 | なし |
| `order_summary.py` | 80 | なし |
| `paper_real_diff.py` | 207 | なし |
| `model_metrics.py` | 124 | `src.prediction.types.TrainingMetrics` |
| `prediction_results.py` | 304 | `src.prediction.types.PredictionResult` |

さらに `PredictionResult` は実体が `src/domain/types.py` にあり、`src/prediction/types.py` は
`from src.domain.types import PredictionResult as PredictionResult` と **再輸出しているだけ** である。
真に prediction 固有の型に依存しているのは `model_metrics.py` の `TrainingMetrics` 1 本のみ。

### 2.3 プロキシが本当に必要なのは reporting と trading だけ

`orchestration` は上位層であり `src.prediction.db` を直接 import しても契約違反にならない。
越境しているのは以下に限られる。

| 束 | データ | 生産者 | 越境して読む者 |
|---|---|---|---|
| ① 取引の事実 | `paper_real_diff` / `order_run_summary` | trading | reporting |
| ② 予測の事実 | accuracy / drift / weekly snapshot | prediction | reporting |
| ③ 予測結果 | `prediction_results` | prediction | reporting, trading |

プロキシが抱える 18 名前の内訳は以下のとおり。

| 区分 | 件数 | 名前 |
|---|---|---|
| 越境消費者なし（prediction 内でのみ使用） | 7 | `load_excluded_features` / `load_model_weights` / `save_feature_selection` / `save_model_metrics` / `save_prediction_accuracy` / `save_prediction_results` / `save_shap_values` |
| `orchestration` のみが使用（上位層ゆえ直接 import で合法） | 2 | `load_feature_exclusion_candidates` / `save_weekly_accuracy_snapshot` |
| 束① | 3 | `upsert_paper_real_diff` / `load_paper_real_diff_summary` / `save_order_run_summary` |
| 束② | 3 | `load_drift_summary` / `load_prediction_accuracy` / `load_weekly_accuracy_snapshots` |
| 束③ | 3 | `load_latest_prediction_timestamp` / `load_prediction_markets` / `load_prediction_results` |

前半 9 件はプロキシから外すだけで済む（消費者は `src.prediction.db` を直接 import するか、
そもそも prediction 内にいる）。Phase 4 は続く 6 件を扱い、残る 3 件が Phase 4b の対象となる。

reporting 側で DB を読んでいるファイルは 4 本のみ:
`dashboard.py` / `kpi.py` / `discord/notifications_report.py` / `query_service.py`。
`llm_review.py` と `discord/notifications_drift.py` は **既に読み込み済みデータを引数で受け取っている**。

### 2.4 ポートは半分建っていて誰も住んでいない

`src/domain/ports.py` には `PredictionResultRepository` と `StockFeatureRepository` が定義済みだが、
`src/infrastructure/` に **DB アダプタが 1 つも存在しない**（実装は `in_memory.py` のテスト用のみ）。
唯一の利用者 `src/trading/execution/runner.py:70` は
`prediction_repo: PredictionResultRepository | None = None` と受け取り、None なら従来の直呼びに落ちる。

一方で合成ルートは既に関数注入を行っている:

```python
# run_auto_trade.py:52, run_claude_trader.py, orchestration/jobs/daily.py:331
PaperBroker(market_data_port=YFinanceMarketDataAdapter(), record_diff=upsert_paper_real_diff)
```

本設計は新しい流儀の導入ではなく、**建てかけの流儀を 1 本だけ本当に通す**作業である。

### 2.5 テストの patch が実装の偶然に依存している

`tests/` には `src.prediction.db` への参照が 70 箇所ある。
`@patch("src.prediction.db.load_drift_summary")` は、消費側が関数内 import しているという
実装の偶然によって効いている。一方 `tests/unit/test_dashboard.py:68` は
`@patch("src.reporting.dashboard.load_drift_summary")` と、正しい流儀で書かれている。
同一リポジトリ内に 2 つの流儀が混在している。

---

## 3. 資料の不整合（先に解決すべき教義問題）

調査中に、アーキテクチャ資料が **互いに矛盾し、かつ実装と食い違っている** ことが判明した。

| 資料 | 主張 |
|---|---|
| `docs/DDD_ARCHITECTURE.md`（2026-04-27, Accepted） | ADR-003:「`domain/types.py` は移行期間中の re-export。フェーズ 4 で削除する」／タスク 4-3「`domain/` フォルダを削除」／フェーズ 1 の目的「utils 層の肥大化解消」 |
| 実装の現状 | `domain/types.py` が本物の定義を持ち、`prediction/types.py` が再輸出側 |
| `CLAUDE.md` | `src/domain/` は「Shared kernel」、`src/infrastructure/` は「ポートの実装アダプタ」＝ヘキサゴナルの恒久的中核 |
| `docs/ARCHITECTURE.md`（637 行） | `domain` / `infrastructure` / `utils.db` に一切言及なし（DDD 文書のタスク 4-5 が未実施） |

唯一整合しているのは `docs/superpowers/specs/2026-07-31-prediction-microservice-design.md` で、
推論をステートレスな HTTP サービスに切り出し **DB 保存はモノリス側に残す**設計であるため、
永続化を BC の外へ出す本設計と方向が一致する。

### 決定（ADR）

**`src/domain` を恒久的な共有カーネルとする。** `docs/DDD_ARCHITECTURE.md` の ADR-003 および
フェーズ 4-1 / 4-3 を Superseded とする。

**根拠:** DDD 文書は 2026-04-27 起草。その後に `domain/ports.py` と `src/infrastructure/` が導入され、
Phase 1〜3 の全作業が実質的にこの立場を既成事実にしている。いま `domain/` を削除することは、
動作している設計を古い文書に合わせて壊す行為になる。判断すべきは
「どちらが正しいか」ではなく「どちらが既に真実か」である。

DDD 文書自体は削除せず、判断の履歴として残したうえで該当節に Superseded を明記する。

---

## 4. 目標アーキテクチャ

疎結合性は **依存グラフの形** で測る。理想は星形であり、すべての BC の出ていく辺が
抽象のみを持つ最下層に向かい、BC 同士は互いを知らない。

```
orchestration / api          合成ルート: アダプタを組み立て BC に注入する
        ↓
infrastructure/              domain のポートを実装するアダプタ（DB・yfinance・Discord・LLM）
        ↓
backtest / prediction / trading / reporting / ...
        ↓                    BC は domain と純粋 utils（logger 等）しか知らない
domain/                      共有カーネル: 型・ポート・ルール定数・例外。
                             src.domain 以外の src.* を一切 import しない
```

### 却下した代替案

**(a) 永続化を `utils/db/` に降ろす（move-down）。**
`src/prediction/db/` の 5 本を `src/utils/db/` へ移すだけで契約は緑になり、コストも小さい。
しかしこれは **import 結合をスキーマ結合に付け替えるだけ**である。共有層が全 BC のスキーマを知る状態は
変わらず、列を 1 本足せば複数の消費者が壊れうるが import-linter は緑のままになる。
疎結合化ではなく合法化であり、後から剥がすコストはむしろ増える。

**(b) analytics BC を新設して事実テーブルを集約する。**
誰でも書ける置き場は、共有データベースに BC の衣を着せたものに過ぎず、現在の `utils/db` と同じ病に至る。

---

## 5. Phase 4 のスコープ

Phase 4 は **束① + 束②** を扱う。作業の自然な単位は「束」ではなく「ファイル」であり、
`kpi.py` は束①（slippage / diff）と束②（drift / accuracy）を同じ集約関数の下で両方読んでいるため、
束ごとに分けると同一ファイルを二度いじることになる。

| | 内容 | 成果 |
|---|---|---|
| **Phase 4（本設計）** | 束①（書き: ポート＋アダプタ）＋ 束②（読み: 押し上げ＋入口ポート） | reporting 4 ファイルと trading が越境を断つ。プロキシ 18 → 3 |
| **Phase 4b（後続）** | 束③（`prediction_results`。`PredictionResultRepository` を完成） | プロキシ全撤去・`allow_indirect_imports` 削除 |

`allow_indirect_imports = True` の削除は **Phase 4b の完了条件**である。Phase 4 では外せない。

---

## 6. 設計

### 6.1 書き側 — ポートを注入する

trading は自分の仕事の最中に事実を書く。書く先を知らずに書くには注入が要る。

**ポート（`src/domain/ports.py`）**

```
TradeDiffSink        .record(record: TradeDiffRecord) -> None
OrderRunSink         .save(summary: OrderRunSummary) -> None
```

現行署名は引数が 12 個（`upsert_paper_real_diff`）と 10 個（`save_order_run_summary`）ある。
そのままポートに写さず、`src/domain/types.py` に `TradeDiffRecord` / `OrderRunSummary` の
dataclass を立てて 1 引数にする。

**アダプタ（新設 `src/infrastructure/persistence/`）**

`src/prediction/db/paper_real_diff.py` と `order_summary.py` の SQL をそのまま移設し、
Postgres 実装として両ポートを実装する。**SQL は一文字も変更しない。**

**結線**

- `run_auto_trade.py` / `run_claude_trader.py` / `src/orchestration/jobs/daily.py` の
  `record_diff=upsert_paper_real_diff` をアダプタ実体の注入に置き換える
- `src/trading/execution/runner.py:573` の `save_order_run_summary(...)` 直呼びを `OrderRunSink` に置き換える

### 6.2 読み側 — IO を入口へ押し上げる

reporting は整形屋である。データを自分で取りに行くのをやめ、渡してもらう。
この形の前例は `llm_review.py` と `notifications_drift.py` に既にある。

ただし reporting は平屋ではない（`dashboard → monthly → kpi` の三層があり、
`dashboard` と `query_service` はそれ自体が入口である）。押し上げは reporting の外に出ずに止まるため、
線引きの規則を定める。

> **入口から直に呼ばれるモジュールは問い合わせポートを注入される。その下の整形関数はデータを受け取る。**

| モジュール | 立場 | 受け取るもの |
|---|---|---|
| `dashboard.py` | 入口（`run_dashboard.py`） | ポート |
| `query_service.py` | 入口（Discord Bot / `api/external_v1.py` / `scheduler`） | ポート |
| `monthly.py` | 入口（`run_monthly_report.py` / `orchestration/jobs/periodic.py`） | ポート |
| `kpi.py` | 内側の集約 | データ |
| `discord/notifications_report.py` | 内側の整形 | データ |
| `llm_review.py` / `discord/notifications_drift.py` | 内側の整形 | 変更不要（既にデータ受け取り） |

**読み取りポート（`src/domain/ports.py`）**

```
AnalyticsQuery       .paper_real_diff_summary(recent_days: int = 7) -> dict
                     .drift_summary(horizon: int = 1, recent_n: int = 30) -> pd.DataFrame
                     .prediction_accuracy(market: str | None = None, symbol: str | None = None,
                                          horizon: int = 1, limit: int = 500) -> pd.DataFrame
                     .weekly_accuracy_snapshots(n_weeks: int = 4) -> pd.DataFrame
```

既存関数の署名をそのまま写す（既定値も含む）。移設時に署名を変えないことで、
アダプタ実装は本体を移すだけで済み、差分が「場所の変更」に限定される。

`monthly.py` は `run_monthly_report.py` / `orchestration/jobs/periodic.py` から直に呼ばれる入口であると同時に、
`dashboard.py` と `query_service.py` からも呼ばれる。後者 2 つは自身が入口としてポートを保持するため、
`monthly.py` へはそれを引き渡す。ポートを生成するのは最外周の入口（`run_*.py` / `orchestration` / Discord Bot / `api`）のみとする。

ファサードの再発ではないかという疑いは正当であるため、境界を明示する。
現行プロキシとの差は (1) 型付き抽象で実装を差し替えられる、(2) 注入されるためテストが偽物を渡せる
（`@patch` が不要）、(3) 4 メソッドに固定され増加は設計レビューの対象、の 3 点である。
**メソッドが 10 を超えた場合はファサード化の兆候とみなし、設計を見直す。**

### 6.3 除去する保険

`src/reporting/discord/notifications_report.py:57` は引数で受け取りつつ None なら自分で読みに行く。

```python
if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
    accuracy_df = load_drift_summary(horizon=horizon)
```

これは読み側版の `| None = None` である。押し上げの本体はこの保険を外すことにある。

**設計規則:** Phase 4 で導入するポートは必須引数とし、合成ルートが必ず渡す。
`prediction_repo: X | None = None` のような任意注入は行わない。None フォールバックが残る限り
旧経路が生き続け、「ポートを入れたのに何も変わらない」状態になる。

### 6.4 呼び出し元ゼロの関数

`load_turnover_comparison` と `load_open_close_advantage_summary` は本番コードからの呼び出し元が存在しない。
テスト側の参照を確認したうえで、呼び出し元ゼロなら移設せず削除する。

---

## 7. 契約とガード

### 7.1 `.importlinter`

```
layers =
    src.api | src.orchestration
    src.backtest | src.prediction | src.trading | src.reporting | src.watchlist | src.market_data | src.rule_engine | src.quality | src.screening
    src.utils
    src.domain
```

`src.infrastructure` は **載せない**。`src/infrastructure/*` が `src.market_data` を import する一方で
`src/backtest/*` `src/reporting/*` `src/quality/*` が `src.infrastructure` を import しており、
パッケージ単位で循環している。層に追加した時点で契約が落ちる。理由をコメントに明記し、別 Issue を起票する。

`allow_indirect_imports = True` は Phase 4b まで残す。

### 7.2 動的 import 禁止ガード

今回の抜け道は「文字列による動的 import は import-linter に見えない」という一点に尽きる。
契約で表現できないため、テストで縛る。

`src/` 配下で `importlib.import_module("src....")` を検出したら失敗する unit テストを追加する。
これが無ければ後続 Phase で同じ穴が掘られる。

---

## 8. 資料の整合（Phase 4 に含める）

- `docs/DDD_ARCHITECTURE.md` — ADR-003 とフェーズ 4-1 / 4-3 に Superseded を明記し、本設計へのリンクを追加
- `docs/ARCHITECTURE.md` — 「ディレクトリ構成」「レイヤーアーキテクチャ」節を実態へ更新
- `src/utils/db/__init__.py` の docstring — `prediction.py - prediction_results / model_metrics / prediction_accuracy テーブル操作` の記述を除去
- `.importlinter` — #338 に関するコメント群を現状へ更新
- `.github/copilot-instructions.md` — レイヤー記述の有無を確認して追随
- `CLAUDE.md` — `src/prediction/db/` に関する記述の有無を確認

---

## 9. PR 分割

分割の単位は **方向（書き / 読み）ではなくテーブル** とする。`paper_real_diff.py` は
`upsert_paper_real_diff`（書き）と `load_paper_real_diff_summary`（読み）が同一モジュールに同居しており、
方向で切ると 1 つのテーブルの SQL が 2 箇所に分かれる期間が生じるためである。

| PR | 内容 | 挙動変化 |
|---|---|---|
| **PR-1** | ADR + 資料整合 + `layers` に `src.domain` 追加 + 動的 import 禁止ガード（ratchet 方式） | なし |
| **PR-2** | `order_run_summary`: `OrderRunSummary` + `OrderRunSink` + `infrastructure/persistence/` アダプタ + in-memory 偽物 + trading と合成ルートの結線 + `load_turnover_comparison` 削除 | なし（SQL 無変更） |
| **PR-3** | `paper_real_diff`: `TradeDiffRecord` + `TradeDiffSink` + `AnalyticsQuery.paper_real_diff_summary` + 読み側の押し上げ | なし |
| **PR-4** | 束②（accuracy / drift / weekly）: `AnalyticsQuery` の残り 3 メソッド + reporting の押し上げ + テスト書き換え | なし |
| **PR-5** | 後片付け: プロキシ縮小（18→3）・越境消費者なし 7 件と orchestration のみ 2 件の再輸出を削除・`src/prediction/db/` 整理 | なし |

PR-1 を先頭に置く。教義を文書で確定してからコードを動かさなければ、PR-2 のレビューで
「domain に型を置いてよいのか」が蒸し返される。

PR-2 に `order_run_summary` を選ぶのは、80 行・書き手 1 箇所・越境読み手ゼロで、
**ポートの型を確立するのに必要な要素だけを含み余計な波及がない**ためである。
PR-3 以降は PR-2 で確立した「値オブジェクト → ポート → アダプタ → 合成ルート結線 → 旧経路撤去」の
5 ステップをテーブルごとになぞる。

### 動的 import 禁止ガードの補足

`src/utils/db/__init__.py` の 2 箇所は PR-5 まで残るため、ガードは ratchet 方式とする
（既知の違反のみを許容リストに載せ、新規追加を禁じる）。許容リストが空になることが Phase 4 の完了条件であり、
リポジトリ既存のファイル行数ゲート（GRANDFATHERED 方式）と同じ流儀である。

---

## 10. テスト戦略

- **書き側:** `src/infrastructure/in_memory.py` に `InMemoryTradeDiffSink` / `InMemoryOrderRunSink` を追加し、
  trading のテストはそれを注入する。`@patch` を使わない
- **読み側:** reporting のテストから patch を除去し、辞書 / DataFrame のリテラルを渡す形に書き換える
- **アダプタ:** SQL の実挙動は integration テストで担保する（unit では DB に触れない）
- **わざと壊す実験（必須）:** PR-3 で書き換えたテストを数本わざと壊し、赤くなることを確認する。
  patch を剥がした後に「緑のまま何も検証していない」状態になることが最も危険な失敗様式であるため省略しない

### 検証コマンド

```bash
# ローカル unit は空 DB で実行する（開発 DB は 86 万行あり timeout する）
cd python
python -m pytest tests/unit/ -n 2 -q

# import-linter（この呼び方でしか実挙動が見えない）
py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"

# 残留確認: Phase 4b 対象と model_metrics のみが残ること
grep -rn "src\.prediction\.db" src/ tests/
```

`python/check-ci.ps1` は壊れている記録があるため、結果を鵜呑みにせず個別コマンドでも確認する。

---

## 11. リスク

本番は `auto_deploy.ps1` が `develop` を監視して自動デプロイする。
**挙動不変**を守ることが最大の安全装置であり、SQL を一文字も変えないことを全 PR で貫く。

---

## 12. Phase 4 の対象外

- `model_metrics.py`（`TrainingMetrics` 依存・越境消費者なし）— prediction 内に留め、`db/` パッケージを畳むに留める
- 束③ `prediction_results` — Phase 4b
- `src/infrastructure` のパッケージ循環 — 別 Issue
- `src/strategy/` `src/features/` の空ディレクトリ残骸、`python/src/utils/db/_connection.py.orig` / `.rej`（git 未追跡）— 別 PR

---

## 13. 完了条件

- [ ] PR-1〜5 がすべて `develop` にマージされている
- [ ] `src/reporting/*` と `src/trading/*` から `src.utils.db` 経由の束①② 参照が消えている
- [ ] `_DbPackageProxy._PREDICTION_DB` が 3 要素（束③ のみ）になっている
- [ ] `src/` 配下の `importlib.import_module("src....")` が 0 件であり、ガードテストがそれを保証している
- [ ] `layers` 契約に `src.domain` が最下層として含まれ、lint-imports が緑である
- [ ] `docs/DDD_ARCHITECTURE.md` と `docs/ARCHITECTURE.md` が実装と整合している
- [ ] VERSION が 2.15.0 である
