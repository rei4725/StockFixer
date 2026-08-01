# 予測出力の健全性チェック（出力 invariant）設計

- 作成日: 2026-08-01
- 対象バージョン: 2.2.1 時点の develop
- ステータス: 設計承認済み（実装計画はこれから）

## 背景

直近に修正した3件のバグは、すべて同一の失敗モードを持つ。

| Issue / PR | 現象 | 発覚の経緯 |
|---|---|---|
| #612 | 全行 NULL の特徴量列が object dtype で返り、XGBoost 予測が全銘柄でスキップされた | 偶然 |
| #613 | 同じ問題が `stock_fundamentals` にも存在 | #612 の調査中 |
| #615 | LightGBM が特徴量アラインメントを外れて毎回失敗し、アンサンブルが実質 XGBoost 単独に縮退 | 無関係な PR #614 のレビュー中にサブエージェントがモデルファイルを検分して発覚 |

共通するのは **「パイプラインがもっともらしい出力を出しながら、中身が縮退している」** という点である。3件とも例外が外に出ず、予測値そのものは出力され続けたため、正常系と区別がつかなかった。#615 は 705 銘柄すべてが半分のアンサンブルで動いていたが、系のどこもそれを検知しなかった。

#615 の修正（PR #616）に含まれる `179ac53`「アンサンブル縮退時のログを WARNING から ERROR に引き上げ」はログレベルを上げただけである。読まれないログは警報ではない。

### 既存資産の発見

`python/src/utils/alert_service.py` に条件付きアラート機構（NF-303）が実装済みである。

- 4 ルール（パイプライン連続失敗 / 損失上限連続発動 / ドリフト警告連続検出 / health degraded 継続）
- `AlertResult` データクラス、ストリークの永続化（`system_config`）、Discord 送信
- `tests/unit/test_alert_service.py` に単体テスト

しかし **本番経路から一度も呼ばれていない**。`run_conditional_notification` / `evaluate_alert_conditions` の呼び出し元は単体テストのみで、`run_scheduler.py` にも `src/orchestration/jobs/daily.py` にも配線がない。「アラート機構を作ったが繋いでいない」状態であり、これ自体が本設計の主題（壊れているのに誰も気づかない）と同じ病理である。

## 目的

日次パイプラインの予測出力に対して不変条件を評価し、違反を Discord へ発報する。あわせて未配線の `alert_service` を本番経路に繋ぐ。

### 非目的

- 自動発注の停止、予測結果の保存中止（下記「決定事項」参照）
- 特徴量そのものの検証（NULL 列・dtype 異常の検知）。#612 / #613 の根であるが層が異なるため別途起票する
- healthchecks.io 連携（#496。ユーザーの API キー取得待ち）
- 閾値の環境変数化

## 決定事項

| 論点 | 決定 | 理由 |
|---|---|---|
| 検知時の挙動 | **通知のみ**。予測結果の保存も後続の自動発注も続行する | 誤検知でその日の取引機会を丸ごと失う設計を避ける。止める判断は人が行う |
| 不変条件の種類 | **前回ランからの急変 + 絶対値の一致チェック** の2本立て | 急変検知だけでは #615 型（最初から壊れている定常縮退）を検出できない。`model_count` はずっと 1 のままで急変が存在しないため |
| 期待値の与え方 | ハードコードせず **実際にロードできたモデル数から導く** | モデル構成を変えても設定変更が不要 |
| 条件非成立時の日次サマリー | **送らない**（違反時のみ発報） | 毎日同じ文面が流れると人が読まなくなる |
| 実装方式 | 既存 `alert_service` にルールを追加し、日次パイプラインから配線する | 器・テスト・通知経路が既にある。未配線の4ルールも同時に生き返る |

条件非成立時のサマリーを送らない決定により、「チェック自体が動いていない」状態と「正常」が区別できなくなる。緩和として評価結果は毎回 INFO で構造化ログに出力する。チェックの死活そのものを検知する正しい道具は #496 の死活監視であり、本決定によりその優先度が上がる。

## アーキテクチャ

```
daily_pipeline [2/5] 予測
    ├ preload_models(model_types)  → ロード成功したモデル名リストを返す（戻り値を新設）
    ├ predict_all_unified()        → output_rows: list[PredictionResult]
    └ output_top_worst_results()   → prediction_results へ保存
                    │
                    ▼
    src/prediction/output_invariants.py  ← 新設（純関数。DB / ネットワークに触れない）
        evaluate_output_invariants(
            loaded_model_names,   ← ランタイム情報（絶対値チェック用）
            output_rows,          ← 今回ランの実物
            previous_stats,       ← 前回ラン統計（DB から取得。無ければ None）
        ) -> InvariantReport
                    │
                    ▼
daily_pipeline [5/5] 通知
    alert_service.check_prediction_output_rule(report) -> AlertResult (NF-303-5)
        └ evaluate_alert_conditions() に合流（既存4ルールも同時に評価される）
            └ run_conditional_notification(notifier=send_webhook_notification)
```

### 設計上の制約

「ロードできたモデル数 vs 実際に推論に成功したモデル数」の一致は**ランタイムの情報**である。DB に残るのは成功数（`prediction_results.model_count`）のみで、期待値は残らない。したがって後追いで DB を読むだけの独立ジョブでは絶対値チェックが原理的に成立しない。評価は予測処理の文脈内で行う必要がある。

### 各要素

**`src/prediction/output_invariants.py`（新設）**

純関数として実装する。入力は「ロードできたモデル名」「今回の予測結果」「前回ラン統計」の3つのみ。DB もネットワークも触れないため、単体テストがモック地獄にならない。#615 の教訓（テストは通るが実物では動かない / モックが実物と食い違っていた）を踏まないための選択である。

型:

- `PredictionRunStats` — ラン単位の集計（銘柄数、`model_count` 中央値、`diff_ratio` の標準偏差）
- `InvariantViolation` — 違反1件（ID、実測値、閾値、説明）
- `InvariantReport` — 違反リストと評価サマリー

**`preload_models()` の戻り値追加**

現在は `None` を返し、ロードの成否をログに出すのみ。ロードに成功したモデル名のリストを返すように変更する。既存の呼び出し元は戻り値を無視できるため後方互換である。

**前回ラン統計の取得**

`prediction_results` から `model_version='production'` に限り、今回を除く直近の `predicted_at` の集計を1件引くクエリを `src/prediction/db/` に追加する。

**通知**

`send_webhook_notification(title, message, color) -> bool` が `alert_service.NotifierFn` 型（`Callable[[str, str, int], bool]`）とそのまま合致するため、アダプタは不要である。

## 判定ロジック

不変条件は絶対値3本、急変3本の計6本。

### 絶対値チェック（前回ラン不要・初回から効く）

| ID | 条件 | 違反判定 | 根拠 |
|---|---|---|---|
| A-1 モデルロード欠損 | 要求した `model_types` のうちロードできた数 | ロード数 < 要求数 なら違反 | モデルファイルの欠損・破損は議論の余地なく異常 |
| A-2 アンサンブル縮退 | `model_count < ロード成功数` の銘柄が占める割合 | **50% 以上**で違反 | #615 は 705/705 = 100% ゆえ確実に検出する。数銘柄が特徴量欠損で片肺になるのは正常運用の揺らぎとして鳴らさない |
| A-3 予測0件 | `len(output_rows) == 0` | 違反 | `predict_all_unified` の `wrapper` は例外を握って `None` を返すため、全銘柄が失敗しても `output_rows` は空リストになるだけで例外にならず、[2/5] の CRITICAL 判定を素通りする。#612 の実際の経路である |

### 急変チェック（前回ラン統計との比較。無ければスキップ）

| ID | 指標 | 違反判定 | 根拠 |
|---|---|---|---|
| B-1 予測銘柄数 | 前回比の減少率 | **20% 以上の減少**で違反。増加は鳴らさない | #612 は 705→0 で確実に検出する。銘柄追加は正常であるため片側判定 |
| B-2 `model_count` 中央値 | 前回中央値との比較 | **低下**したら違反。上昇は鳴らさない | 片肺化が広がった瞬間を検出する |
| B-3 `diff_ratio` の標準偏差 | 前回比 | **半分未満 または 2倍超**で違反。加えて **標準偏差 0 は前回統計を要さず違反** | 「予測は出ているが全銘柄が同じ値」＝中身が無い状態を検出する |

### 発報の形

- 6本の結果を **1つの `AlertResult`（`rule_id="NF-303-5"`）に集約**し、`details` に違反項目の内訳（実測値・閾値・鳴った ID）を格納する
- **ストリークは使わない。1回の違反で即 `triggered=True`**（`threshold=1`）。既存4ルールは「2回連続」で発報するが、定常縮退は連続するためストリークにすると発見が丸一日遅れるだけで得がない。通知のみで実害が出ない以上、即時でよい
- 前回ラン統計は `model_version='production'` に限り、今回を除く直近の `predicted_at` を1件参照する

### 閾値の置き場所

`output_invariants.py` のモジュール定数とする（`alert_service.py` の `PIPELINE_FAIL_THRESHOLD` 等と同じ流儀）。環境変数化は運用して煩わしくなってから行う。

### 誤検知の見積もり

B-3 が最も鳴きやすい。市況が凪いだ日は予測分散が素直に縮むため、「半分未満」は月に数回踏む可能性がある。通知のみで実害がないため、初期は鳴らして肌感を掴む。数週間運用して煩わしければ 1/3 に緩める。

## エラー処理

- **評価そのものは NON_CRITICAL** — `_handle_stage_error(PipelineStage.NON_CRITICAL, ...)` で包む。健全性チェックが本体パイプラインを停止させては本末転倒である
- **前回統計の取得に失敗したら `previous_stats=None` として絶対値のみ評価する** — 握りつぶさず `logger.error(..., exc_info=True)` を出力する（`except: pass` は使わない）
- **通知送信の失敗は `run_conditional_notification` の戻り値 `False` をログに記録するのみ**。ここも NON_CRITICAL

## テスト方針

純関数であるためモックは DB 層の1点で済む。要は実物を再現した回帰テストである。

- **#615 再現**: 705銘柄すべて `model_count=1`・ロード成功2モデル → A-2 が `triggered=True`
- **#612 再現**: `output_rows=[]` → A-3 が `triggered=True`
- **正常系**: 全銘柄 `model_count=2`・前回比の変動が閾値内 → 全ルール `triggered=False`
- **前回統計 None**: 急変ルール B-1〜B-3 がスキップされ、絶対値のみ評価される
- **境界値**: 縮退率ちょうど 50%、銘柄数ちょうど 20% 減、標準偏差ちょうど 1/2 倍

`PredictionResult` は実データと同じ形で組み立てる。`model_count` を捏造した dict などで代替しない。

## 変更範囲

### 新規

| ファイル | 内容 |
|---|---|
| `src/prediction/output_invariants.py` | 純関数と型定義 |
| `tests/unit/prediction/test_output_invariants.py` | #615・#612 再現を含む回帰テスト |

### 変更

| ファイル | 内容 |
|---|---|
| `src/prediction/predict_unified.py` | `preload_models()` がロード成功したモデル名リストを返す（後方互換） |
| `src/prediction/db/` | 前回ラン統計を引くクエリを1本追加 |
| `src/utils/alert_service.py` | `check_prediction_output_rule()` 追加、`evaluate_alert_conditions()` に合流、条件非成立時のサマリーを送らないよう `run_conditional_notification` を調整 |
| `src/orchestration/jobs/daily.py` | [2/5] 直後に評価、[5/5] 付近で `run_conditional_notification` を呼ぶ配線 |
| `tests/unit/test_alert_service.py` | `len(results)` の期待値を 4→5（`test_alert_service.py:293`）、サマリー非送信の挙動 |

## 副次的な影響

配線により、眠っていた既存4ルールが本番で初めて動作する。パイプライン連続失敗・損失上限3日連続・ドリフト警告連続・health degraded 継続の4本である。初日から既存ルールが発報する可能性があるが、それは誤発報ではなく、今まで見えていなかった実態が可視化されることを意味する。

## 検証

- 単体テストのカバレッジゲート 80% を維持する
- `check-ci.ps1` 相当（lint / mypy / pylint / import-linter / unit / bandit / pip-audit）を通す
- 本番反映後、最初の `daily_pipeline`（営業日 07:30 JST）で評価がログに出ることを確認する

## 関連

- Issue #615 / PR #616 — 本設計の直接の動機
- PR #612 / #613 — 同一失敗モードの先行事例
- Issue #496 — 死活監視（healthchecks.io）。本設計の決定により優先度が上がる
- PR #614 — 予測配信マイクロサービス。フェーズ2の疎通確認で「`/health` が片肺ロードでも `ok` を返す」「モデル未ロードによる縮退がリクエスト時に無言」という同型の穴が報告されており、サービス側にも同等の invariant が必要である
