# StockFixer ロードマップ

> 更新日: 2026-04-21  
> 正本: このドキュメントで**収益改善施策**と**非機能改善施策（NF）**の優先度・進捗を管理する

---

## 1. 目的

### 収益改善（R-xxx）
- 目的1: 手数料・スリッページ控除後の実現可能な収益を継続的に向上する
- 目的2: 最大ドローダウンを抑え、運用停止リスクを下げる
- 目的3: 予測精度偏重ではなく、執行品質とリスク制御を含めて最適化する

### 非機能改善（NF-xxx）
- 目的4: CI/CD パイプラインと品質ゲートを強化し、デグレ検出を自動化する
- 目的5: 監視・可観測性を向上し、障害の検知と原因特定を迅速化する
- 目的6: コード品質・例外設計・型安全性を高め、保守コストを低減する
- 目的7: 運用制度（Runbook・ADR・API仕様）を整備し、属人化を排除する

---

## 2. 管理方針

- 本書: 何を、いつまでに、どの指標で改善するかを管理する
- GitHub Issue: 実装・検証・レビューの実行単位として管理する
- 運用手順: 実行コマンドや運用オペレーションは `docs/OPERATIONS.md` に記載する
- 設計詳細: レイヤー構造やモジュール責務の変更は `docs/ARCHITECTURE.md` に記載する
- 進捗更新: 施策の着手・完了時に「進捗ボード」を更新する
- ロードマップ本文には GitHub URL を大量に埋め込まず、必要な場合は Issue 番号だけを記載する
- 完了済み施策は「完了済みアーカイブ」セクションに圧縮し、作業中・未着手施策にのみ詳細を記載する

---

## 3. KPI（評価指標）

### 3.1 最重要KPI

| KPI | 定義 | 目標 |
|---|---|---|
| Net Return | 手数料・スリッページ控除後の期間収益率 | 四半期でプラス維持 |
| Max Drawdown | 期間中最大ドローダウン | 15%以下 |
| Sharpe Ratio | リスク調整後リターン | 1.0以上 |
| Turnover | 売買回転率 | 過剰売買の抑制（前期比 -20%） |

### 3.2 補助KPI

| KPI | 定義 | 目標 |
|---|---|---|
| Hit Rate | 方向一致率 | 参考指標として監視 |
| Avg Slippage | 約定スリッページ平均 | 前期比で改善 |
| Stop Trigger Rate | 日次損失上限発動率 | 異常増加時に原因調査 |

### 3.3 非機能KPI

| KPI | 定義 | 目標 |
|---|---|---|
| CI Pass Rate | GHA パイプライン成功率 | 95% 以上 |
| Test Coverage | pytest --cov カバレッジ | 80% 維持（現状維持） |
| Broad Except Count | `except Exception` の件数 | services/brokers/models 層でゼロ |
| Health Check Uptime | /health エンドポイント疎通率 | 99% 以上 |
| Deploy Lead Time | コミット → 本番反映のリードタイム | weekly_redeploy で自動計測 |

---

## 4. 優先順位（2026-04-11 時点）

### 現在の優先順（収益改善）

1. R-203 月次レポート自動化（DONE）
2. R-211 実験トラッキング基盤（DONE）
3. R-205 ストレステスト（歴史的クラッシュ再現）★前倒し（高リスク根拠確立）
4. R-212 マルチホライズン統合シグナル（DONE）
5. R-213 出来高プロファイル特徴量（DONE）
6. R-215 ショートサイド活用
7. R-216 BT最適パラメータ自動ロード ★新規（高リスク方針の中核）
8. R-217 Kelly実績更新（BT実測値を calc_position_size へ反映） ★新規
9. R-307 ドローダウン適応型資本配分 ★前倒し（高確信度集中投資の前提）
10. R-206 Optuna による自動ハイパーパラメータ探索
11. R-210 動的スリッページモデル
12. R-214 リバランス頻度最適化 ★後ろ倒し（高リスク方針と逆行）
13. R-201 マクロ指標・イベント特徴量強化
14. R-202 アンサンブル重み最適化（DONE）
15. R-207 シャドーモード A/B テスト基盤
16. R-209 サバイバーシップバイアス補正
17. R-204 収益化機能 PoC

### 現在の優先順（非機能改善）

1. ~~NF-101 `requirements-dev.txt` 分離（低コスト・即効）~~ ✅ 完了
2. ~~NF-102 broad `except Exception` 撲滅（services/brokers/models 層）~~ ✅ 完了
3. ~~NF-201 GHA Integration/E2E テスト追加~~ ✅ 完了
4. ~~NF-202 依存脆弱性スキャン（pip-audit）~~ ✅ 完了
5. ~~NF-203 SAST スキャン（bandit）~~ ✅ 完了
6. NF-301 `/health` エンドポイント実装
7. NF-302 構造化ログ（JSON 形式）
8. NF-303 アラートルール定義
9. NF-401 カスタム例外階層整備
10. NF-402 Pylint 有効化
11. NF-403 DB マイグレーション戦略正式化
12. NF-404 services 層サブパッケージ化（DDD移行後の整理）
13. NF-405 yfinance データソース抽象化
14. NF-406 PredictionResult 型強化
15. NF-407 batch_runner エラー集約構造化
16. NF-408 import アーキテクチャ lint（CI） ★新規（#108/#112 根本解決）
17. NF-501 デプロイ Runbook 作成
18. NF-502 障害対応フロー（Incident Response）文書化
19. NF-503 ADR（Architecture Decision Records）導入
20. NF-504 API 仕様書（OpenAPI）整備
21. **NF-601〜605 DDD アーキテクチャ移行**（詳細: [DDD_ARCHITECTURE.md](DDD_ARCHITECTURE.md)）

### 優先順の考え方（収益改善）

- R-203・R-211・R-212・R-213・R-202 は完了済み。次フェーズへ移行する
- **高リスク方針への転換**: BT検証に裏打ちされた根拠のもと、積極的なポジション集中・Kelly比率引き上げを目指す
- R-205 を最初に実施し「取れるリスクの上限」を統計的に確立してから攻勢に出る
- R-216・R-217 で BT最適パラメータ・実測 Kelly を実運用に自動フィードバックし、高リスク方針の実装基盤を整える
- R-307 を Q1 2027 から Q4 Sprint 4 に前倒しし、高確信度銘柄への集中投資ロジックを年内に投入する
- R-214 は Turnover 削減（＝リスク低減方向）のため優先度を下げ、高リスク施策の後段に回す
- R-210 でコスト推定を精緻化した後、R-206 で改善サイクルを定型化する
- R-201・R-209 は基盤整備後に特徴量やモデルを拡張するフェーズで着手する
- R-204 は内部KPIが安定してから着手し、外部提供を後回しにする

### 優先順の考え方（非機能改善）

- NF-101・NF-102 は変更量が小さく即着手可能。収益施策と並行できる
- NF-201〜NF-203 はCIパイプライン強化セット。NF-101 完了後に着手
- NF-301〜NF-303 は監視基盤セット。R-303 運用ダッシュボードの前提として整備する
- NF-401〜NF-403 はコード品質セット。リファクタリングコストが高いため中期で段階投入
- NF-501〜NF-504 は制度整備セット。実装よりもドキュメント作業が中心。隙間時間に進める
- NF-601〜NF-605 は DDD 移行セット。NF-Phase 3（監視）・NF-Phase 4（コード品質）完了後に着手する。フェーズ 0（NF-601）は完了済み

### Issue 対応表

| Roadmap ID | Issue |
|---|---|
| R-211 | #14 |
| R-210 | #22 |
| R-205 | #13 |
| R-203 | #18 |
| R-301 | #31 |
| R-302 | #34 |
| R-303 | #32 |
| R-304 | #33 |
| NF-101 | （Issue未採番） |
| NF-102 | （Issue未採番） |
| NF-201 | （Issue未採番） |
| NF-202 | （Issue未採番） |
| NF-301 | （Issue未採番） |
| NF-404 | #69 |
| NF-405 | #70 |
| NF-406 | #72 |
| NF-407 | #71 |
| NF-408 | #108 / #112 |
| R-408 | #80 |
| R-409 | #82 |
| R-410 | #83 |

---

## 5. 四半期ロードマップ

## Q4 2026（収益効率改善フェーズ）

### 到達目標

- コスト控除後で評価できる運用サイクルを確立
- 既存データ資産を活用した低コスト収益改善施策を複数投入
- 改善サイクルの定型化（自動ハイパーパラメータ・A/Bテスト）

### 実施項目

| ID | 施策 | 優先度 | 完了条件 |
|---|---|---|---|
| R-203 | 監視・月次レポート自動化 | P2 | KPIと主要トレードを自動出力 |
| R-211 | 実験トラッキング基盤 | P2 | 学習ごとにパラメータ・メトリクス・特徴量来歴をDuckDBに自動記録 |
| R-212 | マルチホライズン統合シグナル | P2 | 1d/3d/5d/10dを時間軸で重み付け統合し、単独シグナルより Hit Rate が向上 |
| R-213 | 出来高プロファイル特徴量 | P2 | 相対出来高比・出来高移動平均乖離率を追加し学習・予測パイプラインへ反映 |
| R-205 | ストレステスト（歴史的クラッシュ） | P1 | コロナ/リーマン期間でMDD 15% 以下を検証し、許容リスク上限を統計的に確立する |
| R-215 | ショートサイド活用 | P2 | Worst10の空売りシグナルをPaperBrokerで検証し下落局面の収益機会を定量評価 |
| R-216 | BT最適パラメータ自動ロード | P1 | `optimal_params.json` を `SignalGenerator` / `RiskManager` が自動参照し、BT検証済みの閾値・SL・TPを実運用に反映する |
| R-217 | Kelly実績更新（BT実測値フィードバック） | P1 | バックテスト実測の `win_rate`/`avg_win`/`avg_loss` を `calc_position_size` に渡し、固定デフォルト値を廃止する |
| R-307 | ドローダウン適応型資本配分 | P2 | DD進行中に資本量を非線形縮小し、回復期に段階的増加する関数を RiskManager に追加（Q1 2027 から前倒し） |
| R-201 | マクロ/イベント特徴量強化 | P2 | 特徴量寄与分析を記録 |
| R-202 | アンサンブル重み最適化 | P2 | 単純平均よりKPI改善 |
| R-206 | Optuna 自動ハイパーパラメータ探索 | P2 | 週次 Walk-Forward 連動でパラメータ自動更新 |
| R-207 | シャドーモード A/B テスト基盤 | P2 | 新旧モデルを並行記録し定量評価後に切り替え |
| R-209 | サバイバーシップバイアス補正 | P2 | index_membership_history テーブルを DuckDB に追加 |
| R-210 | 動的スリッページモデル | P2 | 出来高・注文サイズ連動の市場インパクトモデルを導入 |
| R-214 | リバランス頻度最適化 | P3 | 予測変動量が閾値未満の場合は発注スキップし Turnover 前期比 -20% を検証（高リスク方針と逆行するため後段） |
| R-204 | 収益化機能PoC | P2 | 有料配信またはAPIのPoC完了 |

### Q4 実行順序

1. R-203・R-211・R-212・R-213・R-202 は完了済み
2. R-205 で許容リスク上限（MDD・連敗分布）を統計的に確立し、高リスク方針の根拠を得る
3. R-215 でショートサイド収益機会を確認しつつ、R-216・R-217 で BT→実運用フィードバックループを構築する
4. R-307 で高確信度銘柄への集中投資ロジックと DD 適応縮小を同時整備し、R-206 で改善サイクルを定型化する
5. R-210 で paper/real 乖離データをバックテストへ還元する
6. R-214・R-201・R-209 は基盤整備後の精度改善テーマとして後段に置く
7. R-204 は内部KPIの安定後に PoC 判断を行う

### Q4 スプリント案

| Sprint | 主施策 | 完了条件 |
|---|---|---|
| Sprint 1 | R-203 / R-211 | 月次サマリー自動生成 + run_id が学習成果物と紐づく（完了済み） |
| Sprint 2 | R-212 / R-213 / **R-205** | 完了済み施策に加え、コロナ/リーマン期間で許容MDD上限を統計的に確認 |
| Sprint 3 | R-215 / **R-216** / **R-217** | ショートサイド定量評価 + BT最適パラメータ自動適用 + Kelly実績フィードバック |
| Sprint 4 | **R-307** / R-206 / R-210 | 高確信度集中投資ロジック + Optuna自動更新 + スリッページ推定値参照 |

---

## Q1 2027（運用高度化フェーズ）

### 到達目標

- モデル改善の実験から本番昇格までを定量基準で接続する
- 予測精度だけでなく資本配分と運用監視まで最適化対象に含める
- 相場環境変化への追随力（セクターローテーション・クロスアセット）を強化する

### 実施項目

| ID | 施策 | 優先度 | 完了条件 |
|---|---|---|---|
| R-301 | 資本配分エンジン高度化 | P3 | 予測値、信頼度、相関、レジームを使って保有比率を一括最適化できる |
| R-302 | モデル昇格ゲート自動化 | P3 | shadow 成績と本番成績の比較条件を満たしたモデルのみ昇格する |
| R-303 | 運用監視ダッシュボード | P3 | 損益、停止理由、ドリフト、乖離、実験結果を1画面で確認できる |
| R-304 | 外部提供向け API/配信準備 | P3 | 内部運用と分離した read-only 出力経路を構築し、公開前のセキュリティ要件を定義する |
| R-305 | セクターローテーション戦略 | P3 | R-101 のレジーム判定をシグナル生成に接続し、bull/bear/range ごとにセクター配分を切り替え |
| R-306 | クロスアセット相関特徴量 | P3 | USD/JPY・VIX・米国債利回りを特徴量に追加し、Hit Rate または Sharpe が改善 |
| R-307 | ドローダウン適応型資本配分 | P3 | DD進行中に資本量を非線形縮小し、回復期に段階的増加する関数を RiskManager に追加 |
| R-308 | 分割エントリー/エグジット | P3 | 予測確信度に応じて2〜3分割で約定し、スリッページ実測値（R-107）と比較検証 |
| R-408 | 予測信頼区間の可視化と活用 | P3 | Quantile Regression で予測の上下幅を算出し、信頼区間をシグナル生成・ポジションサイジングに活用する（Issue #80） |
| R-409 | 引け前サマリー通知（15:00 アラート） | P3 | 15:00 時点の保有ポジションを再評価し、引け前に Discord へサマリーを自動送信する。R-303 の前段として実用価値が高い（Issue #82） |
| R-410 | ドリフト監視閾値の動的設定 UI | P3 | ドリフト監視の閾値を Discord コマンドで動的に変更・確認できるようにし、運用中の調整を即時反映する（Issue #83） |

### Q1 2027 の方針

- R-301・R-302 を先行し、改善したモデルを安全に本番へ昇格できる運用を作る
- R-305・R-306 は R-101・R-106（完了済み）の基盤をシグナル生成に延伸し、低コストで収益源を強化する
- R-307 は現在の一律停止（R-003）より資本効率が高く、Sharpe 改善直結のため早期着手を推奨
- R-303 は UI を急がず、DuckDB と月次レポートの集約可視化層として開始する
- R-304 は収益化そのものではなく、内部基盤と配信境界の設計を目的とする

---

---

## NF 非機能改善ロードマップ

---

## NF-Phase 1: 即効・低コスト（Q4 2026 Sprint 1〜2 並行）✅ COMPLETE

### 到達目標

- Dockerイメージから開発用ツールを排除し、本番攻撃面を縮小する ✅
- services/brokers/models 層のサイレント障害を排除し、エラー追跡を可能にする ✅

### 実施項目

| ID | 施策 | 優先度 | ステータス | 完了日 | 完了条件 |
|---|---|---|---|---|---|
| NF-101 | `requirements-dev.txt` 分離 | P1 | DONE | 2026-04-21 | black/isort/flake8/mypy/pytest 等の開発依存を分離し、Dockerfile は requirements.txt のみ参照する |
| NF-102 | broad `except Exception` 撲滅 | P1 | DONE | 2026-04-21 | services・brokers・models 層で `except Exception: pass` または `except Exception:` をゼロにし、具体的な例外型 + `logger.error(..., exc_info=True)` に置換する |

---

## NF-Phase 2: CI/CD パイプライン強化（Q4 2026 Sprint 2〜3）✅ COMPLETE

### 到達目標

- PR 時に Integration / E2E テストが自動実行され、デグレを即検出できる ✅
- 依存パッケージの既知 CVE と Python コードの危険パターンを CI で自動検出する ✅

### 実施項目

| ID | 施策 | 優先度 | ステータス | 完了日 | 完了条件 |
|---|---|---|---|---|---|
| NF-201 | GHA Integration/E2E テスト追加 | P2 | DONE | 2026-04-21 | `.github/workflows/` に integration-tests.yml を追加し、PR 時に `tests/integration/` と `tests/e2e/` を実行する |
| NF-202 | 依存脆弱性スキャン（pip-audit） | P2 | DONE | 2026-04-21 | GHA で `pip-audit` を実行し、HIGH 以上の CVE があればパイプラインを FAIL にする |
| NF-203 | SAST スキャン（bandit） | P2 | DONE | 2026-04-21 | GHA で `bandit -r src/ -ll` を実行し、HIGH severity の検出でパイプラインを FAIL にする |

---

## NF-Phase 3: 監視・可観測性（Q4 2026 Sprint 4 〜 Q1 2027 Sprint 1）

### 到達目標

- コンテナの死活と業務的な健全性を1エンドポイントで確認できる
- ログを構造化し、将来の集約基盤（ELK/Loki）への移行コストを最小化する
- Discord 通知を「条件付きアラート」化し、通知疲れを防ぐ

### 実施項目

| ID | 施策 | 優先度 | 完了条件 |
|---|---|---|---|
| NF-301 | `/health` エンドポイント実装 | P2 | Flask に `/health` を追加し、DB 接続・スケジューラ最終実行時刻・直近予測実行時刻を JSON で返す。Docker HEALTHCHECK がこのエンドポイントを叩く |
| NF-302 | 構造化ログ（JSON 形式） | P2 | `logger.py` に JSON フォーマッタを追加し、`LOG_FORMAT=json` 環境変数で切替可能にする |
| NF-303 | アラートルール定義 | P2 | 「日次パイプライン N 回連続失敗」「日次損失上限 3 日連続発動」等の条件をコード化し、条件非成立時はサマリーのみ Discord 送信する |

---

## NF-Phase 4: コード品質・保守性（Q1 2027）

### 到達目標

- カスタム例外階層でエラー制御を層別に整理し、横断的なエラーハンドリングを可能にする
- 型安全性を強化し、実行時バグの発生率を下げる
- DB スキーマ変更を追跡可能にし、ロールバックを安全に行えるようにする
- `services/` の肥大化を解消し、各機能の変更範囲を明確にする
- `run_*.py` のアーキテクチャ違反を CI で自動検出し、レイヤー規約を機械的に守れるようにする

### 実施項目

| ID | 施策 | 優先度 | 完了条件 |
|---|---|---|---|
| NF-401 | カスタム例外階層整備 | P3 | `src/domain/exceptions.py` に `StockFixerError` 基底クラスと `DataFetchError` / `ModelTrainingError` / `BrokerError` / `PipelineError` を定義し、各層で使用する |
| NF-402 | Pylint 有効化 | P3 | `.pre-commit-config.yaml` の pylint コメントアウトを解除し、Git バージョン問題を解消してフックを有効化する |
| NF-403 | DB マイグレーション戦略正式化 | P3 | `src/utils/db/migrations/` ディレクトリに連番 SQL ファイルを配置し、起動時にバージョンチェック + 未適用マイグレーションを自動実行する |
| NF-404 | services 層サブパッケージ化 | P3 | `src/services/` を機能別サブパッケージに分割し、16 ファイル超のフラット構造を解消する（Issue #69） |
| NF-405 | yfinance データソース抽象化 | P3 | `DataSourceBase` を定義し、yfinance 実装を差し替え可能にしてテスト容易性を向上する（Issue #70） |
| NF-406 | PredictionResult 型強化 | P3 | `PredictionResult` のマルチホライズンフィールドを `Optional` ではなく専用型に変更し、型安全性を向上する（Issue #72） |
| NF-407 | batch_runner エラー集約構造化 | P3 | `BatchResult` 型を導入し、バッチ実行の成功/失敗を構造化して集約・通知できるようにする（Issue #71） |
| NF-408 | import アーキテクチャ lint（CI） | P3 | `run_*.py → src.utils 直接 import` 等のレイヤー違反を CI で自動検出し、#108/#112 を根本解決する |

---

## NF-Phase 5: 運用制度・ドキュメント（Q1〜Q2 2027）

### 到達目標

- デプロイ・障害対応の手順を属人化から脱却させる
- 設計判断の根拠を記録し、将来の変更コストを下げる
- 外部向け API の仕様を明文化し、R-304 収益化 PoC の前提を整える

### 実施項目

| ID | 施策 | 優先度 | 完了条件 |
|---|---|---|---|
| NF-501 | デプロイ Runbook 作成 | P3 | `docs/RUNBOOK_DEPLOY.md` に「正常デプロイ」「ロールバック」「手動デプロイ」「バージョン切り戻し」の手順を記載する |
| NF-502 | 障害対応フロー文書化 | P3 | `docs/INCIDENT_RESPONSE.md` に障害レベル定義（P1〜P3）・エスカレーションフロー・ポストモーテムテンプレートを記載する |
| NF-503 | ADR 導入 | P3 | `docs/adr/` ディレクトリを作成し、「DuckDB 採用理由」「short-lived connection 採用理由」等の過去決定を遡及的に記録する |
| NF-504 | API 仕様書（OpenAPI）整備 | P3 | Flask エンドポイントと Discord コマンド仕様を `docs/API_SPEC.md` にまとめる。R-304 着手前に完了する |

---

## NF-Phase 6: DDD アーキテクチャ移行（Q2〜Q3 2027）

### 到達目標

- 技術レイヤー分割から Bounded Context（ドメイン駆動）構成への段階移行を完了する
- 機能追加・修正の変更範囲が 1 BC 内で完結する状態を達成する
- `domain/types.py` の神ファイルを解体し、型の責務を各 BC に帰属させる

詳細設計・移行ステップは **[docs/DDD_ARCHITECTURE.md](DDD_ARCHITECTURE.md)** を参照。

### 実施項目

| ID | 施策 | 優先度 | フェーズ | 完了条件 |
|---|---|---|---|---|
| NF-601 | DDD フェーズ0: 現状整理・方針合意 | P2 | Phase 0 | DDD_ARCHITECTURE.md 作成・合意完了（✅ 2026-04-27 完了） |
| NF-602 | DDD フェーズ1: 型の分散・utils 整理 | P2 | Phase 1 | `domain/types.py` の型を各 BC に移動し re-export で互換維持。全テストグリーン |
| NF-603 | DDD フェーズ2: BC 境界確立 | P2 | Phase 2 | `brokers/`→`trading/`・`features/`→`analysis/`・`api/`→`reporting/` 移動。全テストグリーン |
| NF-604 | DDD フェーズ3: 大規模再構成 | P3 | Phase 3 | `models/`→`prediction/`・`data/`→`market_data/`・`services/`解体。全テストグリーン |
| NF-605 | DDD フェーズ4: import パス統一・仕上げ | P3 | Phase 4 | re-export 互換削除・`domain/` 削除・linter クリーン |

### 優先順の考え方

- NF-601 は 2026-04-27 完了済み
- NF-602 は影響範囲が限定的（import 修正数少）で即着手可能。NF-Phase 3〜4 完了後に着手を推奨
- NF-603 は `features/` の影響箇所が多いため、全テストグリーンを確認しながら 1 PR ずつ進める
- NF-604・NF-605 は大規模変更のため、収益改善施策（R 系）が安定した Q3 2027 以降に実施する

---



### 到達目標

- 収益源を現行の方向性予測に加えて多様化する
- モデルアーキテクチャの本質的な刷新を行う
- 市場中立・ヘッジ戦略を導入し、下落局面の損失を構造的に抑える

### 実施項目

| ID | 施策 | 優先度 | 完了条件 |
|---|---|---|---|
| R-401 | Transformerベースモデル追加 | P4 | 時系列長期依存を捉えるモデルを追加し、アンサンブルの多様性向上を確認 |
| R-402 | MLベースエグジット最適化 | P4 | 保有後の最適売却タイミングをモデルで予測し、固定TP/SL超の成績改善を確認 |
| R-403 | ペアトレード（市場中立戦略） | P4 | R-102 の相関データを転用してスプレッド収束狙いの方向リスク中立戦略を実装 |
| R-404 | オルタナティブデータ（ニュースセンチメント） | P4 | 日経・FOMC等のセンチメントスコアを特徴量に追加し、イベント前後の方向予測を強化 |
| R-405 | 寄付/引け注文の戦略的使い分け | P4 | 予測ホライズン・流動性に応じて寄成/引成を選択し、価格優位性を定量評価 |
| R-406 | テールリスクヘッジ | P4 | VIX急騰時にポジション縮小またはインバースETF連動のヘッジを導入 |
| R-407 | 相関ベースのポートフォリオリスク管理 | P4 | 動的相関行列で実効分散度を監視し、相関上昇時に強制分散を実施 |

---

## 6. 進捗ボード

### アクティブ施策（収益改善）

| ID | ステータス | 期限 | 更新日 | メモ |
|---|---|---|---|---|
| R-203 | DONE | 2026-04-12 | 2026-04-12 | monthly_report_pipeline / run_monthly_report.py / /monthlyreport コマンド実装完了 |
| R-211 | DONE | 2026-10-12 | 2026-04-13 | experiment_runs DDL追加・db/experiment.py CRUD・model_training_pipeline run_id自動記録実装完了 |

### アクティブ施策（非機能改善）

| ID | ステータス | 期限 | 更新日 | メモ |
|---|---|---|---|---|
| NF-101 | DONE | 2026-04-21 | 2026-04-21 | requirements-dev.txt を作成し Dockerfile の COPY を requirements.txt のみに変更 |
| NF-102 | DONE | 2026-04-21 | 2026-04-21 | services/brokers/models 層の broad except を置換 |
| NF-201 | DONE | 2026-10-31 | 2026-04-21 | integration-tests.yml による統合テスト自動実行を確認済み |
| NF-202 | DONE | 2026-10-31 | 2026-04-21 | pip-audit JSON パース統一・GITHUB_STEP_SUMMARY 出力・全脆弱性 FAIL |
| NF-203 | DONE | 2026-11-07 | 2026-04-21 | bandit -ll フラグ・HIGH のみ FAIL・MEDIUM は warning・GITHUB_STEP_SUMMARY 出力 |
| NF-301 | DONE | 2026-11-30 | 2026-04-30 | src/api/health.py に Flask /health 追加。DB接続・スケジューラ最終実行時刻・直近予測時刻を JSON 返却。run_scheduler.py に start_health_server() 組込み。Dockerfile HEALTHCHECK を /health 叩く形式に更新 |
| NF-302 | TODO | 2026-12-07 | - | logger.py に JSON フォーマッタ追加、LOG_FORMAT 環境変数対応 |
| NF-303 | TODO | 2026-12-14 | - | 条件付きアラートルール定義 |
| NF-401 | TODO | 2027-01-25 | - | src/domain/exceptions.py 作成 |
| NF-402 | TODO | 2027-02-01 | - | pylint 有効化（Git バージョン問題解消） |
| NF-403 | TODO | 2027-02-15 | - | src/utils/db/migrations/ ディレクトリ + マイグレーションランナー実装 |
| NF-404 | TODO | 2027-02-22 | - | services/ サブパッケージ化（#69） |
| NF-405 | TODO | 2027-03-01 | - | DataSourceBase 導入・yfinance 抽象化（#70） |
| NF-406 | TODO | 2027-03-08 | - | PredictionResult マルチホライズン型強化（#72） |
| NF-407 | TODO | 2027-03-08 | - | BatchResult 導入・バッチエラー集約（#71） |
| NF-408 | TODO | 2027-03-15 | - | import レイヤー違反 CI lint チェック（#108/#112 根本解決） |
| NF-501 | TODO | 2027-01-18 | - | docs/RUNBOOK_DEPLOY.md 作成 |
| NF-502 | TODO | 2027-02-08 | - | docs/INCIDENT_RESPONSE.md 作成 |
| NF-503 | TODO | 2027-03-01 | - | docs/adr/ ディレクトリ作成・過去ADR遡及記録 |
| NF-504 | TODO | 2027-03-15 | - | docs/API_SPEC.md 作成（R-304 の前提） |
| R-212 | DONE | 2026-11-02 | 2026-04-12 | compute_multi_horizon_score / apply_multi_horizon_score_column 実装・order_execution_pipeline の buy/sell 判定を統合スコアへ移行 |
| R-213 | DONE | 2026-11-09 | 2026-04-14 | volume_ratio / volume_price_trend / volume_ma_deviation を add_technical_indicators() に追加。モデル再学習が必要。 |
| R-205 | TODO | 2026-11-02 | - | 歴史的クラッシュ期間リストを docs に整備、許容MDD上限を統計的に確立（高リスク方針の根拠和り） |
| R-215 | DONE | 2026-11-09 | 2026-04-30 | paper_broker.get_pnl_summary / RiskManager._get_daily_realized_loss / _get_consecutive_losses に SHORT_COVER を追加。ショートサイドPnLが正確に日次損失・連続損失に反映される |
| R-216 | DONE | 2026-11-16 | 2026-04-30 | execution.py に _resolve_kelly_params を追加し、buy/short 両ループで BT実績 win_rate/avg_win/avg_loss を calc_position_size に渡す。optimal_params.json 未登録銘柄はデフォルト値にフォールバック |
| R-217 | DONE | 2026-11-16 | 2026-04-30 | optimizer.py save_optimal_params_json の metrics に avg_win/avg_loss がすでに実装済みを確認。次回 run_backtest_optimize.py 実行時から自動反映 |
| R-214 | TODO | 2026-12-14 | - | 予測変動量閾値による発注スキップロジックを order_execution_pipeline に追加（高リスク方針と逆行するため後回し） |
| R-214 | TODO | 2026-11-16 | - | 予測変動量閾値による発注スキップロジックを order_execution_pipeline に追加 |
| R-215 | TODO | 2026-11-23 | - | Worst10 空売りシグナルの PaperBroker 検証 |
| R-205 | DONE | 2026-11-30 | 2026-04-23 | stress_test_pipeline.py 実装・統合テスト・CLIスモークテスト追加。コロナ/リーマンシナリオ対応。 |
| R-201 | TODO | 2026-10-26 | - | |
| R-202 | DONE | 2026-11-02 | 2026-04-14 | model_metrics テーブルの directional_accuracy を使い softmax 重み付きアンサンブルを実装（predict_single_stock / predict_unified）。 |
| R-206 | DONE | 2026-11-09 | 2026-04-30 | optimizer.py に run_optuna_optimization / run_optuna_batch 追加。scheduler.py に USE_OPTUNA / OPTUNA_N_TRIALS 環境変数対応。requirements.txt に optuna>=3.6.0 追加 |
| R-207 | TODO | 2026-11-16 | - | |
| R-209 | TODO | 2026-11-30 | - | index_membership_history テーブルを DuckDB に追加 |
| R-210 | DONE | 2026-12-07 | 2026-04-30 | backtest/slippage.py 新規作成（平方根市場インパクトモデル・calibrate_alpha・make_slippage_fn）。backtester.py に slippage_fn パラメータと _get_slippage() ヘルパー追加 |
| R-204 | TODO | 2026-12-14 | - | |
| R-301 | TODO | 2027-01-18 | - | 資本配分を単銘柄判定からポートフォリオ最適化へ拡張 |
| R-302 | TODO | 2027-02-15 | - | shadow 成績と本番成績の比較による昇格ゲートを実装 |
| R-303 | TODO | 2027-03-15 | - | DuckDB と月次レポートを集約する監視ダッシュボードを整備 |
| R-304 | TODO | 2027-04-12 | - | 外部配信向け read-only API と公開条件を分離設計 |
| R-305 | TODO | 2027-01-25 | - | R-101 market_regime.py のシグナル生成接続 |
| R-306 | DONE | 2027-02-01 | 2026-04-14 | fetch_cross_asset_features() を data_loader.py に追加し、data_pipeline / predict_single_stock で結合。モデル再学習が必要。 |
| R-307 | DONE | 2026-12-07 | 2026-04-30 | compute_dd_capital_scale 純粋関数・update_peak_balance・get_current_dd_ratio を risk_manager.py に追加。dd_state テーブルを _connection.py DDL に追加。execution.py の run_daily_orders 先頭で update_peak_balance() 呼び出し |
| R-308 | TODO | 2027-02-22 | - | R-107 スリッページ実測値と比較する分割発注ロジック |
| R-401 | TODO | 2027-Q2 | - | |
| R-402 | TODO | 2027-Q2 | - | |
| R-403 | TODO | 2027-Q2 | - | |
| R-404 | TODO | 2027-Q2 | - | |
| R-405 | TODO | 2027-Q2 | - | |
| R-406 | TODO | 2027-Q2 | - | |
| R-407 | TODO | 2027-Q2 | - | |
| R-408 | TODO | 2027-Q2 | - | Quantile Regression 信頼区間の可視化と活用（#80） |
| R-409 | TODO | 2027-01-25 | - | 引け前サマリー通知 15:00 ポジション再評価（#82） |
| R-410 | TODO | 2027-02-08 | - | ドリフト監視閾値 Discord コマンド UI（#83） |

ステータス定義：TODO / DOING / BLOCKED / DONE（KPI評価済み）

> **非機能施策の完了判定**: 収益KPIではなく、完了条件欄に記載した技術的な受け入れ基準を満たした時点で DONE とする

---

### 完了済みアーカイブ

| ID | 施策 | 完了日 |
|---|---|---|
| R-001 | コスト込みシグナル閾値導入（gross/net KPI全出力） | 2026-03-29 |
| R-002 | ボラ連動ポジションサイズ導入（ATR連動・上下限比率） | 2026-04-03 |
| R-003 | 日次損失上限ガード（2%上限・翌営業日自動解除） | 2026-04-03 |
| R-004 | Walk-Forward標準ジョブ化（差分CSV/MD・Discord通知） | 2026-03-29 |
| R-101 | 市場レジーム判定（bull/bear/range・portfolio backtest連携） | 2026-04-04 |
| R-102 | 相関制約付き銘柄選定（MAX_SECTOR_POSITIONS） | 2026-04-04 |
| R-103 | 執行品質改善（出来高・日次レンジ proxy で MARKET/LIMIT 自動切替） | 2026-04-05 |
| R-104 | ドリフト監視と再学習トリガー | 2026-04-04 |
| R-105 | SHAP 特徴量寄与の継続監視（Discord通知） | 2026-04-04 |
| R-106 | マルチタイムフレーム特徴量（週足・月足トレンド） | 2026-04-04 |
| R-107 | paper/real 乖離追跡（DuckDB自動集計・週次Discord） | 2026-04-05 |
| R-108 | 適応的シグナル閾値（予測分散連動の動的閾値） | 2026-04-04 |
| R-109 | モデル信頼度によるポジション加減（confidence_ratio） | 2026-04-04 |
| R-110 | 決算カレンダー回避フィルター（±3営業日マスク） | 2026-04-05 |
| R-208 | 特徴量選択の自動化（Permutation Importance・SHAP保護） | 2026-04-05 |
| NF-101 | `requirements-dev.txt` 分離（本番依存と開発ツールの分離） | 2026-04-21 |
| NF-102 | broad `except Exception` 撲滅・ロギング改善 | 2026-04-21 |
| NF-201 | GHA Integration/E2E テスト追加（integration-tests.yml 確認） | 2026-04-21 |
| NF-202 | 依存脆弱性スキャン（pip-audit JSON パース統一・GITHUB_STEP_SUMMARY 出力） | 2026-04-21 |
| NF-203 | SAST スキャン（bandit -ll 追加・HIGH のみ FAIL・MEDIUM は warning） | 2026-04-21 |
| R-214 | 予測変動量閾値スキップロジック（MIN_CHANGE_RATIO・save_order_run_summary 接続） | 2026-04-21 |
| R-215 | ショートサイド活用（SHORT_COVER PnL を日次損失・連続損失・get_pnl_summary に反映） | 2026-04-30 |
| R-216 | BT最適パラメータ自動ロード（_resolve_kelly_params 追加・buy/short 両ループで実績 Kelly 適用） | 2026-04-30 |
| R-217 | Kelly実績更新（optimizer.py metrics に avg_win/avg_loss は実装済みを確認） | 2026-04-30 |
| R-307 | ドローダウン適応型資本配分（compute_dd_capital_scale・update_peak_balance・dd_state テーブル追加） | 2026-04-30 |
| R-206 | Optuna 自動ハイパーパラメータ探索（run_optuna_batch・USE_OPTUNA 環境変数・optuna 依存追加） | 2026-04-30 |
| R-210 | 動的スリッページモデル（slippage.py 新規・平方根市場インパクト・backtester.py 統合） | 2026-04-30 |
| NF-301 | /health エンドポイント実装（DB接続・スケジューラ最終実行時刻・直近予測時刻 JSON 返却・Docker HEALTHCHECK 更新） | 2026-04-30 |

---

## 7. 実行ルール

### 共通
- 新規施策は必ず ID を採番してから着手する
- 実装に着手する施策は原則として対応する GitHub Issue を持つ
- ロードマップ側には Issue のフル URL を多用せず、Issue 番号または対応表で参照する
- 施策中止時は理由をメモし、類似施策の再検討条件を残す
- 毎週1回、進捗ボードと KPI トレンドを更新する
- 完了済み施策は「完了済みアーカイブ」へ移動し、本文から詳細を削除する

### 収益改善施策（R-xxx）
- 完了判定は「実装完了」ではなく「KPI評価完了」で行う

### 非機能改善施策（NF-xxx）
- ID採番ルール: `NF-1xx` CI/CD、`NF-2xx` 即効品質、`NF-3xx` 監視、`NF-4xx` コード品質、`NF-5xx` 制度文書
- 完了判定は各施策の「完了条件」欄に記載した技術的受け入れ基準で行う
- 収益施策のスプリントと並行着手を原則とし、収益施策の遅延要因にしない
- NF-Phase 1（NF-101/102）は他フェーズに先行して即着手する

---

## 8. 変更履歴

- 2026-03-29 〜 2026-04-05: P0〜P2 初期施策（R-001〜R-004, R-101〜R-110, R-208）を順次完了。詳細はアーカイブを参照
- 2026-04-11: R-203 の前提整備として Discord API 層と時刻処理ポリシーを整備
- 2026-04-11: 完了済み施策をアーカイブに圧縮し、未着手・実施中施策のみ詳細表示に再構成
- 2026-04-11: 収益改善追加施策として R-212〜R-215（Q4 2026 即効・低コスト）、R-305〜R-308（Q1 2027 中期）、R-401〜R-407（Q2 2027+ 長期）を追加
- 2026-04-19: 非機能改善施策（NF-101～NF-504）を新設。CI/CD・監視・コード品質・制度文書の5フェーズ15施策を追加
- 2026-04-20: 高リスク方針への転換に伴う優先度見直し。R-205・R-307を前倒し、R-216（BT最適パラメータ自動ロード）・R-217（Kelly実績更新）を新規追加。R-214（リバランス頻度最適化）を低優先度に後回し。Q4スプリント案を再構成
- 2026-04-30: Sprint 3完了（R-215/R-216/R-217）・Sprint 4完了（R-307/R-206/R-210）・NF-301完了（Flask /health エンドポイント・Docker HEALTHCHECK 更新）
