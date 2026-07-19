# 戦略ファクトリー: 自動実装・自動昇格ループ

## 背景

戦略ファクトリー（#369）は夜間バッチでルール組合せ仮説をサンプリングし、過学習ゲート
（DSR / PBO / 取引数 / DD / 対チャンピオン改善）を通過した仮説のみ `results/factory/reports/`
へ不変 JSON レポートを出力する。IssueAgent の `--factory-intake` がこれを GitHub Issue
として起票するが、`strategy-factory` ラベルのみで `auto-ok` は付与されず、採否は人間の
判断に委ねられている（戦略ファクトリーの Phase 2「反映先定義・auto-ok 化」は従来
未着手のままだった）。

一方 IssueAgent には既に「`auto-ok` ラベル付き Issue を拾って実装 → PR 作成 → 全チェック
green を自己確認 → 自己承認 → 自動マージ」という無人フローが存在する（`orchestrator.py`
の `run_for_project` / `_wait_fix_merge_loop`）。ただし develop ブランチに GitHub 側の
branch protection（required status checks）は設定されておらず、実質的なマージゲートは
IssueAgent 自身のポーリングロジックのみに依存している。

ユーザーからの要望は「利益最大化のための改善を Claude Code に継続的に行わせたい。最終的には
コードを書き換えてコミットするところまで自動化したい」というもので、対象範囲・自動マージの
可否・昇格タイミング・ロールバックについて対話の中で以下の合意を得た。

## 目的

戦略ファクトリーの仮説生成を「既存ルールのパラメータ組合せ」だけでなく「LLM が発想する
新規特徴量・新規ルールの実装」まで拡張し、アイデア → 実装 → 検証 → 自動マージ → 即時昇格 →
実績監視 → 基準割れ時の自動ロールバック、という一連のループを無人で回す。

## 非目標

- 実売買の発注ロジック自体（`trading/brokers/`, `risk_management` 等）の自動変更・自動
  マージは対象外。引き続き人間レビュー必須とする。
- 昇格前のシャドウ運用期間は設けない（マージ＝即昇格）。この判断はユーザーが明示的に
  リスクを認識した上で選択したものであり、本設計ではその前提を変更しない。
- 既存の予測モデル向けシャドウ A/B・昇格ゲート（v1.30.0 で実装済みの
  `src.prediction.shadow_evaluation`）の仕組みを本機能が置き換えることはない。両者は
  独立した昇格経路として併存する。

## 合意事項サマリー（対話で確定した設計判断）

| 論点 | 決定 | 理由 |
|---|---|---|
| 自動化の到達点 | コード変更のコミットまで自動化する | ユーザー要望 |
| マージ形態 | IssueAgent 同様、自動 PR 作成＋条件を満たせば自動マージ | 既存 auto-ok 方針と整合 |
| 自動マージ対象範囲 | 戦略・特徴量・バックテスト周辺のみ | 実発注ロジックは金銭的リスクが桁違いに大きいため人間レビューを維持 |
| 昇格タイミング | マージ＝即昇格（シャドウ期間なし） | ユーザーが速度を優先しリスクを許容 |
| ロールバック | 実績が基準を下回ったら自動で前状態に戻す | 即時昇格のリスクを事後的に相殺する安全弁 |
| ロールバック実行経路 | 直接 `git revert && git push`（IssueAgent 非経由） | 通常フロー（最大2時間の遅延）だと損失拡大を防げないため |

## アーキテクチャ全体

```
[毎日05:00 夜間バッチ: factory.py]
  既存: sample_hypotheses()（ルール組合せ）
  新規: generate_llm_hypotheses()（新規特徴量/新規ルールのアイデア発想）
     ↓
  評価・ゲート（既存 evaluate_hypothesis / apply_gate）
  ※ アイデア型は実装前のためこの時点でバックテストできない（後述）
     ↓
  factory_report.py（拡張）
    パラメータ探索型 → labels: ["strategy-factory", "auto-ok"]
    アイデア型       → labels: ["strategy-factory-idea", "auto-ok"]
    Issue本文に機械可読な識別子ブロックを埋め込む

[07:00 IssueAgent --factory-intake]（既存・無変更）
     ↓
[2時間ごと IssueAgent 通常フロー]（既存・無変更）
  auto-ok Issue を実装 → PR作成

[PR作成時 GitHub Actions（新規2本）]
  1. strategy-scope-guard.yml   変更ファイルパスの許可範囲チェック
  2. backtest-gate-check.yml    独立した過学習ゲート再計算

[2時間ごと IssueAgent 通常フロー]（既存・無変更）
  全チェック green → 自己承認 → 自動マージ（squash）＝即昇格
  strategy_promotions テーブルに記録

[日次 rollback_monitor.py]（新規）
  昇格済み対象の実績を監視 → 基準割れで直接 revert & push
```

## コンポーネント詳細

### 1. `generate_llm_hypotheses()`（`src/backtest/factory.py` に追加）

- 入力: DuckDB に蓄積された過去の合格/不合格傾向、`compute_metrics_by_regime` によるレジーム別
  弱点、直近のチャンピオンメトリクス。
- 出力: アイデア仕様のリスト。各要素は `idea_id`（ハッシュ）, `category`
  （`new_feature` | `new_rule`）, `target_bc`（`market_data` | `rule_engine`）,
  `description`（自然言語）, `rationale` を持つ。
- バックエンド: 既存 `get_text_review_port()`（`src/infrastructure/llm/factory.py`）を再利用。
  `LLM_BACKEND` 切替もそのまま効く。
- システムプロンプトに「`trading/brokers/`, `risk_management`, `domain/ports` には触れる
  提案をしない」ことを明記する（CI ガードとの二重防御であり、これ自体は強制力を持たない）。
- 設定（`config/settings.py` に追加）:
  ```python
  FACTORY_LLM_IDEATION_ENABLED: bool = Field(default=False)
  FACTORY_LLM_IDEATION_MODEL: str = Field(default="claude-opus-4-8")
  FACTORY_LLM_IDEATION_MAX_TOKENS: int = Field(default=2048)
  FACTORY_LLM_IDEATION_MAX_IDEAS_PER_NIGHT: int = Field(default=3)
  ```
- 失敗時（例外・スキーマ不正）は空リストを返し、既存のパラメータ探索型のみで夜間バッチは
  通常通り継続する（`hypothesis_review.py` と同じ graceful degradation 方針）。

### 2. Issue 自動起票の拡張（`src/backtest/factory_report.py`）

- 既存のパラメータ探索型仮説（バックテスト済み）は `labels: ["strategy-factory", "auto-ok"]`。
- 新規のアイデア型（未実装のため未検証）は `labels: ["strategy-factory-idea", "auto-ok"]`
  として区別する。ラベルを分けておくことで、将来どちらかだけ `auto-ok` を止めたくなった場合
  に個別に制御できる。
- Issue 本文に、対象仮説/アイデアの識別子を機械可読ブロック（YAML frontmatter 形式）で
  埋め込む。この識別子は実装後のコードにも登録規約として反映し、CI がバックテスト対象を
  特定するのに使う。
- アイデア型の Issue 本文には「実装後は `run_factory_gate_check.py` で定義される評価基準
  （後述）を満たすこと。満たさない場合は PR を作成しない、または close すること」という
  明示的な指示を含める。

### 3. `strategy-scope-guard.yml`（新規 GitHub Actions ワークフロー）

- `pull_request` イベントで起動。`git diff --name-only origin/develop...HEAD` で変更ファイル
  一覧を取得し、許可ディレクトリ allowlist（`python/src/backtest/`, `python/src/market_data/`,
  `python/src/rule_engine/`, `python/src/prediction/`（features 関連のみ）, `python/tests/`,
  `python/config/optimal_params.json`）と照合する。
- allowlist 外のファイルが1件でも含まれていれば fail。
- 判定ロジック本体は `python/scripts/check_strategy_scope.py`（新規、純粋関数として実装し
  ユニットテスト可能にする）に切り出し、ワークフローはそれを呼ぶだけにする。

### 4. `backtest-gate-check.yml`（新規 GitHub Actions ワークフロー）

- `strategy-scope-guard` 通過後に実行（`needs:` で依存）。
- 新設 `run_factory_gate_check.py`（CLI ラッパー、既存 `run_*.py` パターンに倣う）が PR
  本文の識別子ブロックを読み取り、対象の新ルール/新特徴量を用いて `evaluate_hypothesis()` →
  `apply_gate()` を独立に再計算する。
- 合否は終了コードで返す。失敗すれば通常の CI red として扱われ、既存の
  `wait_for_pr_checks`（IssueAgent 側、無変更）がマージをブロックする。
- これが実質的な「利益に貢献するか」の最終審査であり、IssueAgent（実装エージェント）の
  自己申告に頼らない独立検証となる。

### 5. 昇格記録: `strategy_promotions` テーブル（DuckDB、新規）

| カラム | 説明 |
|---|---|
| `promotion_id` | 主キー |
| `merge_commit_hash` | マージコミットの SHA |
| `pr_number` | GitHub PR 番号 |
| `rule_or_feature_id` | gate-check で使う識別子と同一 |
| `promoted_at` | 昇格日時（UTC） |
| `pre_promotion_baseline` | 昇格直前のチャンピオン Sharpe 等 |
| `status` | `active` \| `rolled_back` |

マージ検知ジョブ（既存の日次/週次バッチの一部として、マージ済み PR のうち
`strategy-factory*` ラベル由来かつ未記録のものを検出）がこのテーブルへの書き込みを担う。

### 6. `rollback_monitor.py`（`src/orchestration/jobs/` に新規、日次実行）

- `strategy_promotions` から `status = 'active'` かつ昇格後 N 営業日（既定5日、設定化）
  経過したレコードを対象に、当該ルール/特徴量が寄与した実現損益を集計する。
- ロールバック判定基準（設定化、初期値は運用しながら調整）:
  - 直近 N 日の実現 Sharpe が `pre_promotion_baseline` を著しく下回る、または
  - 絶対損失が閾値を超える。
- 検知した場合の手順:
  1. `git revert <merge_commit_hash>`（**`strategy_promotions` に記録されたコミットのみ
     revert 対象にできる** — 任意コミットの誤 revert を防ぐガード）。
  2. push 前に高速セーフティチェック（import-linter ＋対象範囲の unit テストのみ、
     `check-ci.ps1` のフル実行ではなく数分で終わる部分集合）を実行。
  3. 通過すれば `git push origin develop`、`strategy_promotions.status` を `rolled_back`
     に更新、Discord へ通知（取り消し理由の数値付き）。
  4. セーフティチェックが失敗した場合は push せず、Discord へ緊急アラート（自動revertが
     実行できなかった旨）を出し人間判断に委ねる。直接 push という唯一のバイパス経路である
     ため、「develop を壊さない」の最終防波堤として残す。
- 設定:
  ```python
  ROLLBACK_MONITOR_ENABLED: bool = Field(default=False)
  ROLLBACK_MONITOR_MIN_DAYS: int = Field(default=5)
  ROLLBACK_MONITOR_SHARPE_DROP_THRESHOLD: float = Field(default=0.5)  # baseline比-0.5、運用しながら調整
  ```

## エラーハンドリング方針

- LLM 呼び出し（アイデア発想）は graceful degradation。失敗してもパラメータ探索型の
  夜間バッチは継続する。
- 新設 CI 2本は「落ちたらマージしない」のみで、IssueAgent 側のロジックは無変更で機能する。
- 直接 revert&push だけが唯一の「CI を経ずに develop に反映される」経路のため、push 前の
  セーフティチェックを必須とする。

## テスト戦略

- `generate_llm_hypotheses()`: `hypothesis_review.py` 系のテストパターン（LLM ポートをモック、
  スキーマ検証、`FACTORY_LLM_IDEATION_ENABLED=False` 時の no-op 確認）を踏襲。
- `check_strategy_scope.py`: allowlist 判定ロジックを純粋関数として単体テスト（境界値・
  ネストしたパス・許可外拡張子等）。
- `run_factory_gate_check.py`: 既存の `evaluate_hypothesis` / `apply_gate` の単体テストを
  再利用しつつ、識別子からの対象特定ロジックを追加テスト。
- `rollback_monitor.py`: 昇格記録のフィクスチャに対し、閾値割れ検知 → revert 対象コミット
  特定のロジックを単体テスト。実際の `git revert` / `git push` はサブプロセス呼び出しを
  モックして検証する（本番 develop 相手の E2E は行わない）。

## スコープ外（将来検討）

- 予測モデル向けシャドウ A/B 昇格ゲート（`src.prediction.shadow_evaluation`）との統合・共通化。
- ロールバック判定基準の具体的な閾値チューニング（初期値は保守的に設定し、運用実績を
  見ながら調整する前提）。
- `strategy-factory-idea` 型に対する追加の人間レビューフラグ（現状は他の auto-ok Issue と
  同列。運用してみて事故が多ければラベルを分離してある強みを活かし、ここだけ auto-ok を
  外す変更を later で検討できる）。
