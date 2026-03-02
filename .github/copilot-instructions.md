# GitHub Copilot Instructions - StockFixer

## プロジェクト概要

StockFixer（CuteStock）は、Pythonによる戦略ロジックとAI予測、C#による注文実行とUI表示を組み合わせたリアルタイム自動売買システムです。

### 技術スタック
- **Python**: データ取得、テクニカル分析、AI予測、シグナル生成、REST API、Discord Bot
- **C#**: 証券会社連携、注文実行、WPF UI（将来実装予定）

---

## 禁止事項（重要）

### 機密ファイル - 絶対に読み取り・変更禁止
- `.env` ファイル
- `src/env` 配下のファイル
- `*/config/secrets.*`
- `*/.pem`
- API キー、トークン、認証情報を含むファイル全般

機密ファイルの編集が必要な場合は、ユーザーに確認してください。

### セキュリティガイドライン
- 機密情報（APIキー、パスワードなど）はハードコーディングしない
- ユーザー入力は必ず検証する
- 環境変数を適切に使用する
- ログや出力に認証情報を含めない

---

## ディレクトリ構成

```
python/
├── run_*.py                    # 実行スクリプト（1階層に集約）
├── requirements.txt            # 依存パッケージ
├── Dockerfile                  # Docker設定
├── データ取得対象.csv           # 対象銘柄リスト
├── src/
│   ├── api/                    # API・Discord Bot（最上位層）
│   ├── services/               # オーケストレーション層（データパイプライン等）
│   ├── backtest/               # バックテスト
│   ├── models/                 # AI予測モデル
│   ├── strategy/               # シグナル生成
│   ├── features/               # テクニカル分析・特徴量生成
│   ├── data/                   # データ取得・保存（生データのみ）
│   ├── sbi/                    # SBI証券連携（Flask非依存）
│   └── utils/                  # ユーティリティ（最下位層）
├── tests/                      # テスト（Unit/Integration分離）
│   ├── unit/                  # ユニットテスト（Mock完全・11ファイル）
│   ├── integration/           # 統合テスト（実DB/API依存・11ファイル）
│   ├── conftest.py            # pytest共有Fixture
│   └── README.md              # テスト戦略ドキュメント
├── data/                       # 株価データ保存先
├── models/                     # 学習済みモデル保存先
└── results/                    # 予測結果保存先
```

### レイヤー構造（上位→下位への参照のみ許可）
```
run_*.py → api層 → services層 → models/strategy/backtest層 → features層 → data層 → utils層
```

### runレイヤー原則（必須）
`run_*.py` は **CLIラッパーに徹する**こと。以下のみを許可する：
- `argparse` による引数パース
- services層（またはapi層）の関数呼び出し
- 結果の標準出力

**禁止事項:**
- ビジネスロジック・データ変換・条件分岐の実装
- モデル・DB・外部APIへの直接アクセス
- `import` で features層・data層・utils層を直接参照すること

ロジックが必要な場合は `src/services/` にパイプライン関数を作成し、`run_*.py` からはそれを呼び出すだけにする。

---

## コーディングガイドライン

### 一般原則
- シンプルで読みやすいコード
- 適切な命名（変数、関数、クラスなど）
- 一つの関数は一つの責務を持つ
- エラーハンドリングを適切に実装
- コメントは必要な箇所にのみ付ける

### Python開発ルール
- Pythonコマンドは `py` を推奨、仮想環境は `py -m venv .venv`
- ライブラリインストールは `pip install -r requirements.txt`
- Windowsではパス区切りに `\` を使用
- モジュール化で再利用性・可読性向上、`__init__.py` 配置でimportエラー防止
- importパスは `python/` からの絶対パス指定で統一
- テスト実行: `python -m pytest tests/unit/ -v` (高速) / `python -m pytest tests/integration/ -v` (完全)

### ファイル・ディレクトリ運用
- 命名は一貫性を重視し、スネークケースを推奨（例: `user_profile.py`）
- サブディレクトリを極力排除し、主要 `.py` ファイルは1階層に集約
- `[market]_[symbol]` サブディレクトリ＋用途明示ファイル名で混在・誤保存・誤読込を防止
- 生データは `python/data/`、モデルは `python/models/[market]_[symbol]/` に保存

---

## Skills

### プルリクエスト作成
プルリクエストを作成する際は、GitHub CLI の `gh pr create` コマンドを使用してください。

### Agent Skills（SKILL.md）
タスク固有の詳細な手順は `.github/skills/` 配下のSKILL.mdに定義されています。
オンデマンドで必要なスキルのみがコンテキストに読み込まれます。

| スキル | 説明 |
|--------|------|
| `data-pipeline` | 株価データ取得・特徴量生成・DB保存 |
| `model-training` | 銘柄別 / 統合モデル学習 |
| `prediction` | 予測実行・Top10/Worst10出力 |
| `backtest` | バックテスト（単一期間・Walk-Forward） |
| `scheduler-ops` | スケジューラー運用・日次/週次ジョブ管理 |
| `add-symbol` | 新規銘柄追加フロー |
| `duckdb-ops` | DuckDB操作・データ確認・移行 |
| `docker-ops` | Docker環境のビルド・起動・管理 |
| `git-ops` | Git操作（コミット、ブランチ、push、diff、status等） |
| `version-mgmt` | Gitタグによるバージョン管理・Dockerイメージ連動 |
| `troubleshooting` | よくあるエラーの原因と対処法 |

---

## 技術的なTips

### データ取得・処理
- `yfinance` で株価・為替レート等を安定取得
- yfinanceのMultiIndexカラムはフラット化して後続処理を簡素化
- バッチ処理は market/symbol のみ渡し、既存csv全削除で一貫性担保
- 特徴量生成は全数値列にラグ特徴量・テクニカル指標を自動付与
- 期間指定は end_date を現在日時、start_date を5年前に自動設定

### モデル運用
- 複数モデルの予測値は平均してバイアス低減
- 並列処理は競合バグの原因となるため同期集計が安定
- モデルファイルは `joblib` 形式で保存
- モデル保存パスは `python/models/[market]_[symbol]/モデル名.joblib` で統一
- 予測値は直近データの翌営業日終値

### テスト運用
- **Unit Test（tests/unit/）**: Mock完全・外部依存なし・<5秒実行・開発中に常時実行
- **Integration Test（tests/integration/）**: 実DB/API依存・分単位実行・PR/本番前検証
- テスト実行: `python -m pytest tests/unit/ -v` (高速) / `python -m pytest tests/integration/ -v` (完全)
- テスト用モッククラス（MockBaseModel）で外部依存排除
- `register_model_type` は BaseModel 継承クラスのみ登録可能
- `unittest.mock` や `MagicMock` で外部依存をモック
- conftest.py に共有Fixture（sample_price_df、mock_model_manager等）を配置

### Discord Bot
- `/forecast` コマンドで全マーケットのTop10・ワースト10を送信
- 計算処理は外部スクリプトで事前実行し、BotはCSVのみを参照
- Discordメッセージ長制限（2000文字）に注意し、分割送信処理を実装
- Webhook URLは環境変数で管理し、ハードコーディングしない

### Discord出力仕様
- DataFrameをDiscord向けに出力する場合は `convert_df_for_discord` 関数で統一
- 列名・順序は「シンボル」「現在値」「予想終値」「予想変化率」に統一
- 変化率は `(予想終値-現在値)/現在値` で計算し、有効数字2桁のパーセント形式で出力
- 値段は少数第3位で切り捨て

### パス・ティッカー補正
- `data_path_utils.py` に `get_data_subdir`, `get_models_subdir`, `get_ticker` 等を実装
- 日本株ティッカー補正（7203→7203.T、二重付与防止）

---

## 参照ドキュメント
- `docs/PROJECT_OVERVIEW.md` - システム構成・アーキテクチャ図
- `docs/ARCHITECTURE.md` - 詳細アーキテクチャ・レイヤー構造・データフロー
