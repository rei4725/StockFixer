# StockFixer ドキュメント

本フォルダには、StockFixer（CuteStock）プロジェクトのドキュメントが格納されています。

## ドキュメント一覧

| ファイル | 内容 |
|---------|------|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | プロジェクト概要、システム構成、アーキテクチャ図 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 詳細アーキテクチャ、レイヤー構造、各モジュールの説明、データフロー |
| [DOCKER_DB_ARCHITECTURE.md](DOCKER_DB_ARCHITECTURE.md) | Docker・DB アーキテクチャ、ロック制約、コンテナライフサイクル |
| [OPERATIONS.md](OPERATIONS.md) | 運用手順書、Dockerビルド・デプロイ、命名規約、コマンドリファレンス |
| [ROADMAP_IDEAS.md](ROADMAP_IDEAS.md) | 収益改善ロードマップ（優先度、KPI、四半期計画、進捗管理） |
| [IMPLEMENTATION_BACKTEST_OPTIMIZE.md](IMPLEMENTATION_BACKTEST_OPTIMIZE.md) | ⭐ バックテスト最適化実装詳細、テスト結果、技術仕様 |
| [OPTIMAL_PARAMS_GUIDE.md](OPTIMAL_PARAMS_GUIDE.md) | ⭐ 最適化パラメータ運用ガイド、利用方法、トラブルシューティング |
| [DDD_ARCHITECTURE.md](DDD_ARCHITECTURE.md) | DDD アーキテクチャ移行計画、Bounded Context 定義、BC 間の依存ルール、移行スケジュール |
| [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) | DuckDB テーブル定義（stock_features・prediction_results・market_data_raw・experiment_runs） |
| [API_SPEC.md](API_SPEC.md) | Flask REST API エンドポイントと Discord スラッシュコマンドの仕様、公開範囲の区別 |
| [RUNBOOK_DEPLOY.md](RUNBOOK_DEPLOY.md) | デプロイ Runbook、Docker 単一プロセス起動の制約、正常デプロイ手順 |
| [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) | 障害対応フロー、障害レベル定義（P1〜P3）、初動対応、エスカレーション、ポストモーテム |
| [runbooks/postgres_cutover.md](runbooks/postgres_cutover.md) | DuckDB → PostgreSQL 切り替えランブック（本番実施用の前提と手順） |
| [LOCK_DETECTION_GUIDE.md](LOCK_DETECTION_GUIDE.md) | DuckDB・ファイル操作のロック問題をコミット前に自動検出する仕組みと検出パターン |
| [PRE_COMMIT_GUIDE.md](PRE_COMMIT_GUIDE.md) | Pre-Commit 自動コードレビューのセットアップと、コミット前のチェック内容 |
| [VERSIONING_POLICY.md](VERSIONING_POLICY.md) | バージョン管理の正本（SemVer 基準、version_impact の定義、PR 必須要件） |
| [STRESS_TEST_PERIODS.md](STRESS_TEST_PERIODS.md) | ストレステスト対象の暴落期間定義（正本）、シナリオ一覧、合格基準 |

## ADR（Architecture Decision Records）

過去の設計判断は `adr/` ディレクトリに記録されています。

| ファイル | 内容 |
|---------|------|
| [adr/0000-template.md](adr/0000-template.md) | ADR テンプレート |
| [adr/0001-duckdb-adoption.md](adr/0001-duckdb-adoption.md) | DuckDB 採用理由（PostgreSQL / SQLite との比較） |
| [adr/0002-short-lived-connection.md](adr/0002-short-lived-connection.md) | short-lived connection 採用理由（ロック対策） |
| [adr/0003-settings-trading-policy-separation.md](adr/0003-settings-trading-policy-separation.md) | `config/settings.py` vs `config/trading_policy.py` の責務分離方針 |
| [adr/0004-broker-di.md](adr/0004-broker-di.md) | `BrokerBase` 抽象化による DI 採用理由 |

新しい ADR を追加する場合は `adr/0000-template.md` をコピーして連番でファイルを作成してください。

## クイックリンク

- **初めての方**: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) から読み始めてください
- **開発者向け**: [ARCHITECTURE.md](ARCHITECTURE.md) で詳細な実装を確認できます
- **設計判断の背景**: [adr/](adr/) で過去のアーキテクチャ意思決定を確認できます
- **計画確認**: [ROADMAP_IDEAS.md](ROADMAP_IDEAS.md) で収益改善の優先順位と進捗を確認できます

## 関連ファイル

- [GitHub Copilot設定](../.github/copilot-instructions.md) - コーディングガイドライン・開発ルール
