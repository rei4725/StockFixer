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
