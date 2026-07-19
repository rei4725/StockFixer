# 戦略ファクトリー: 仮説単位の自動批判的レビュー

## 背景

戦略ファクトリー（#369）は夜間バッチでルール組合せ仮説をサンプリングし、過学習ゲート
（DSR / PBO / 取引数 / DD / 対チャンピオン改善）を通過した仮説のみ `results/factory/reports/`
へ不変 JSON レポートを出力する。IssueAgent の `--factory-intake` がこれを GitHub Issue
として起票するが、`strategy-factory` ラベルのみで `auto-ok` は付与されず、採否は人間の
批判的レビューに委ねられている（実績: 初のゲート合格仮説 `fb44f0011174` → Issue #564）。

既存の `src/backtest/critical_review.py`（B-6）は Claude によるレビューを既に実装しているが、
これはバックテストの**方法論全体**（設定定数のスナップショット・`optimal_params.json` の
メトリクス分布）を対象にした静的レビューであり、ファクトリーが生成する**個々の仮説**は
レビュー対象に含まれていない。

## 目的

ゲートを通過した仮説ひとつひとつに対し、Claude が窓別リターン・PBO・DSR 等を精査して
「これは偶然のパターンではないか」を指摘し、Issue 本文に自動で添付する。人間の最終採否判断を
補助するだけの役割とし、ゲート判定（合格/不合格）そのものには一切関与しない。

## 非目標

- ゲートの合否をレビュー結果で上書きしない（レビューは常にゲート判定の後、かつ判定に無関係）。
- 過去の仮説履歴との比較は行わない（当該仮説の情報のみを Claude に渡す）。
- `auto-ok` ラベルの付与や自動マージへの関与は行わない（既存方針を維持）。

## アーキテクチャ

### 配置

新規モジュール `src/backtest/hypothesis_review.py` を `critical_review.py` の兄弟として追加する。
`run_factory_batch`（`src/backtest/factory.py`）から、ゲート通過が確定した各 `FactoryEvaluation`
について `write_report` 直前に呼び出す。

```
run_factory_batch
  └─ evaluate_hypothesis (既存)
  └─ apply_gate (既存)
  └─ [gate_passed のみ] review_hypothesis(evaluation, champion_sharpe)  ← 新規
  └─ write_report(evaluation, champion_sharpe, period)  ← issue_body にレビュー結果を含める
```

### 失敗時の扱い

レビュー呼び出しが例外・スキーマ不正・タイムアウト等で失敗した場合は `logger.warning` を出し、
レビューセクションなしでレポートを書き込む（`critical_review.py` の graceful degradation と
同じ思想）。ゲート判定・レポート書き込み自体はレビューの成否に依存しない。

## 入力（Claude に渡す情報）

当該仮説の情報のみ。過去の仮説履歴やデータベース比較は含めない（プロンプトの単純さと
レイテンシを優先）。

- `hypothesis.rule_spec`（ルール構造 JSON）
- `hypothesis.market` / `hypothesis.lookback_years`
- `evaluation.sharpe_ratio` / `dsr` / `pbo` / `num_trades` / `max_drawdown` / `win_rate` / `total_return`
- `evaluation.window_returns`（窓別リターン列）
- `champion_sharpe`（対照群ベースラインとの差分文脈として）

## 出力スキーマ

`critical_review.py` の `_FINDINGS_SCHEMA` と同系統の構造化出力を JSON Schema で強制する。

```json
{
  "risk_level": "low" | "medium" | "high",
  "assessment": "2〜3文の総評",
  "concerns": ["懸念点1", "懸念点2", "..."]
}
```

- `risk_level`: 過学習・偶然性・方法論的懸念の総合評価。
- `assessment`: 人間が読んですぐ判断材料になる短い総評。
- `concerns`: 箇条書きの具体的懸念（0件も許容 = 特に懸念なしの場合）。

システムプロンプトは `critical_review.py` の `_SYSTEM_PROMPT` を踏襲しつつ、対象が
「バックテスト方法論全体」ではなく「単一仮説の窓別成績パターン」であることを明記する。
重点観点: 窓間のリターンのばらつき（一部の窓だけに依存していないか）、パラメータが
探索グリッドの端に位置していないか（過学習の兆候）、取引数に対して sharpe が
不自然に高くないか、PBO/DSR とリターン分布の整合性。

## Issue 本文への埋め込み

`_build_issue_body`（`factory.py`）のメトリクス表の直後に `### Claude批判的レビュー`
セクションを追加する。

```markdown
### Claude批判的レビュー

{risk_level == "high" の場合のみ ⚠️ バナー（PBO警告と同じ形式）}

{assessment}

**懸念点:**
- {concern1}
- {concern2}
```

レビューが実行されなかった/失敗した場合、このセクション自体を省略する（プレースホルダーは
表示しない）。

## 設定

`config/settings.py` に既存 `BACKTEST_REVIEW_*` と対になる新設定を追加する。

```python
FACTORY_HYPOTHESIS_REVIEW_ENABLED: bool = Field(default=False)
FACTORY_HYPOTHESIS_REVIEW_MODEL: str = Field(default="claude-opus-4-8")
FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS: int = Field(default=2048)
```

`BACKTEST_REVIEW_MAX_TOKENS`（4096）より小さい値とする。単一仮説の短い総評のみを
生成するため、B-6（方法論全体のレビュー、複数 findings を返しうる）より出力量が少ない。

モジュールレベルの再エクスポート（`settings.py` 下部の平坦化ブロック）にも同様に追加する。

## LLM バックエンド

既存の `get_text_review_port()`（`src/infrastructure/llm/factory.py`）をそのまま再利用する。
`LLM_BACKEND`（sdk = API課金 / cli = サブスク認証）の切替もそのまま効く。新規のポート実装は
不要。

## コスト

ゲート条件が厳格なため、通過仮説自体が稀（実績ベースで週次バッチ数件レベル）。呼び出し回数は
ゲート通過数に比例するため、B-6（バッチ全体で毎回1回）より総コストは低い想定。

## テスト

- `tests/unit/test_hypothesis_review.py`（新規）: `test_critical_review.py` のモック方式を踏襲。
  - 正常系: モックした `get_text_review_port` が妥当なスキーマを返す → レビューセクションが
    issue_body に含まれる。
  - スキーマ不正 / JSON パース失敗 → 例外を握りつぶし `None` 相当を返す。
  - `FACTORY_HYPOTHESIS_REVIEW_ENABLED=False` → 呼び出し自体が行われない。
- `tests/unit/test_strategy_factory.py`（既存に追加）: レビュー呼び出しが失敗してもゲート判定・
  レポート書き込みが通常通り完了することを確認する統合的なケースを1件追加。

## スコープ外（将来検討）

- Phase 2「反映先定義」（人間が Issue を承認した後、実際に `rule_engine`/`screening` へ
  反映する自動パイプライン）は本設計の対象外。別途ブレインストーミングが必要。
- 仮説履歴との比較によるデータスヌーピング検知の強化は、まずは単発レビューを運用してから
  必要性を判断する。
