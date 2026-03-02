# code-review スキル: コードレビュー実行ガイド

## 概要

**目的**: AI + 自動チェック機構を活用した包括的なコードレビュー

**対象**: Python コードの品質保証、ロック問題検出、セキュリティ確認

**検証対象:**
- ✅ コード品質（PEP8、命名規則、複雑度）
- ✅ ロック問題（DuckDB並行競合、ファイルロック漏れ）
- ✅ セキュリティ（機密情報漏洩、入力検証）
- ✅ パフォーマンス（不要な処理、N+1クエリ）
- ✅ テスト可能性（疎結合、モック化可能性）

---

## 前提条件

### 環境要件
```bash
# 1. 仮想環境が有効化されていることを確認
.\.venv\Scripts\Activate.ps1

# 2. 必要な tools がインストール済み
.\.venv\Scripts\python -m pip list | Select-String "black|flake8|mypy|pylint|pre-commit"

# 3. Pre-commit が初期化済み（初回のみ）
.\.venv\Scripts\python -m pre_commit install
```

### 対象ファイル
- `python/src/**/*.py` - メインコード
- `python/run_*.py` - 実行スクリプト
- 除外: `tests/`, `.venv/`, `__pycache__/`

---

## コードレビュー実行手順

### ステップ1: 自動チェック実行

#### 1-1. Pre-commit フックで自動検証
```bash
cd c:\src\StockFixer

# コミット前に全フックを実行（自動修正も含む）
.\.venv\Scripts\python -m pre_commit run --all-files

# または特定ファイルのみ
.\.venv\Scripts\python -m pre_commit run --files python/src/services/*.py
```

**出力例:**
```
black....................................................................Passed
isort....................................................................Passed
flake8..................................................................Failed
  python/src/services/data_pipeline.py:45:10: E501 line too long (123 > 100)
check-duckdb-concurrency...............................Passed
check-file-lock..........................................Passed
mypy....................................................................Failed
  python/src/models/predictor.py:12: error: Incompatible types in assignment
```

#### 1-2. 失敗したチェックを個別実行
```bash
# Flake8 のみ詳細表示
.\.venv\Scripts\python -m flake8 python/src --max-line-length=100 --show-source

# mypy の詳細チェック
.\.venv\Scripts\python -m mypy python/src --ignore-missing-imports

# Pylint 品質スコア確認
.\.venv\Scripts\python -m pylint python/src --exit-zero
```

---

### ステップ2: ロック問題検出

#### 2-1. DuckDB並行処理チェック
```bash
# データパイプラインの並行問題を検出
.\.venv\Scripts\python .\.github\hooks\check_duckdb_concurrency.py \
  python/src/services/data_pipeline.py \
  python/src/services/batch_runner.py
```

**検出対象パターン（重大度順）:**

| Severity | パターン | 対策 |
|----------|---------|------|
| 🔴 Critical | `async def` + DB操作 | asyncioを使用しない |
| 🟠 High | `ThreadPoolExecutor` + `execute()` | 2フェーズ化（並列I/O + 順序DB write） |
| 🟠 High | 並列 `upsert_raw_ohlcv()` | `defer_raw_save=True` フラグ使用 |
| 🟡 Medium | `DELETE + INSERT` 非原子 | `INSERT OR REPLACE` または トランザクション |

#### 2-2. ファイルロック検出
```bash
# ファイル操作の排他性問題を検出
.\.venv\Scripts\python .\.github\hooks\check_file_lock.py \
  python/src/data/*.py \
  python/src/services/*.py
```

**検出対象パターン:**

| Severity | パターン | 対策 |
|----------|---------|------|
| 🟠 High | `ThreadPoolExecutor` + `.to_csv()` | 順序実行に変更 |
| 🟠 High | 並列 `joblib.dump()` | 順序実行に変更 |
| 🟡 Medium | `open(..., 'w')` ロック未指定 | `fcntl.flock()` または `threading.Lock` |

---

### ステップ3: 手動コードレビュー

#### 3-1. 変更内容の確認
```bash
# 前回コミットからの変更を表示
git diff HEAD~1

# または特定ブランチとの差分
git diff origin/feature/training..HEAD
```

#### 3-2. レビューチェックリスト

**アーキテクチャ:**
- [ ] レイヤー構造ガイドを遵守（run層 → api層 → services層→ models層 → features層 → data層 → utils層）
- [ ] 上位層が下位層のみを参照（逆参照なし）
- [ ] 適切なレイヤーに機能を配置

**コード品質:**
- [ ] 関数が単一責任を持つ（30行以下が目安）
- [ ] 命名が一貫性・明確性を保つ（スネークケース）
- [ ] 複雑な処理にはコメント記載
- [ ] 定数は `UPPERCASE` で定義

**エラーハンドリング:**
- [ ] 外部API呼び出しに try-except
- [ ] DB操作に例外処理（トランザクション破棄対策）
- [ ] ファイル操作に例外処理
- [ ] エラーメッセージが詳細で対処可能

**パフォーマンス:**
- [ ] 不要なループ・n+1クエリなし
- [ ] 大規模データセットの機械学習時にメモリ効率を確認
- [ ] 外部API呼び出しの最小化

**セキュリティ:**
- [ ] 機密情報（APIキー・パスワード）をハードコードしない
- [ ] ユーザー入力を全て検証
- [ ] SQLインジェクション対策（パラメータ化クエリ）
- [ ] 許可リスト方式のファイル操作

**テスト可能性:**
- [ ] 外部依存（DB・API）がモック可能な設計
- [ ] ピュア関数（副作用なし）の活用
- [ ] テストケースが存在
- [ ] カバレッジ 80% 以上

---

### ステップ4: 修正と再検証

#### 4-1. 自動修正
```bash
# Black によるフォーマット修正
.\.venv\Scripts\python -m black --line-length=100 python/src

# isort によるインポート整理
.\.venv\Scripts\python -m isort --profile=black python/src

# これらの両方を実行
.\.venv\Scripts\python -m pre_commit run --all-files --show-diff
```

#### 4-2. 手動修正
- Flake8 の E501（行長超過）= 関数分割・文字列短縮
- mypy の型エラー = 型ヒント追加 （`Optional[str]` 等）
- セキュリティ警告 = 入力検証追加

#### 4-3. 修正後の検証
```bash
# 全フック再実行（パス確認）
.\.venv\Scripts\python -m pre_commit run --all-files --show-diff

# テスト実行
.\.venv\Scripts\python -m pytest tests/unit/ -v

# 統合テスト（オプション）
.\.venv\Scripts\python -m pytest tests/integration/ -v
```

---

### ステップ5: PR（プルリクエスト）作成

#### 5-1. コミット
```bash
# ステージング
git add -A

# コミット（Conventional Commits）
git commit -m "feat: <機能説明>

- <詳細な変更内容1>
- <詳細な変更内容2>

Closes #<issue_number>"
```

**コミットメッセージ規則:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

| Type | 説明 | 例 |
|------|------|-----|
| feat | 新機能 | feat(data-loader): yfinanceのバッチ処理を追加 |
| fix | バグ修正 | fix(pipeline): DuckDBロック競合を修正 |
| docs | ドキュメント更新 | docs: コードレビューガイドを追加 |
| style | フォーマット（機能変更なし） | style: Black による整形 |
| refactor | リファクタリング | refactor(strategy): 信号生成ロジック簡素化 |
| test | テスト追加・修正 | test(unit): predict関数のテスト追加 |
| chore | 依存パッケージ等 | chore: requirementsを更新 |

#### 5-2. プルリクエスト作成
```bash
# GitHub CLIを使用
gh pr create --title "feat: <概要>" \
  --body "## 変更内容
- <説明1>
- <説明2>

## チェックリスト
- [x] Black/isort/Flake8パス
- [x] DuckDB並行チェック完了
- [x] テスト実行完了（カバレッジ80%+）
- [x] セキュリティ確認完了"
```

#### 5-3. コードレビューを受ける
```bash
# ローカルで Copilot にレビューを依頼（オプション）
.\.venv\Scripts\python -m pre_commit run check-duckdb-concurrency --all-files \
  --show-diff
```

---

## よくあるエラーと対処法

### ❌ エラー1: mypy で "Incompatible types in assignment"
```python
# ❌ NG
def process_data(value: int) -> str:
    result: int = "text"  # エラー: str を int に代入
    return result

# ✅ OK
def process_data(value: int) -> str:
    result: str = str(value)
    return result

# または Optional を使用
from typing import Optional
def process_data(value: Optional[int]) -> Optional[str]:
    return str(value) if value else None
```

### ❌ エラー2: Flake8 で "E501 line too long"
```python
# ❌ NG (150文字)
result = process_very_long_function_name_with_many_parameters(param1, param2, param3, param4, param5)

# ✅ OK
result = process_very_long_function_name_with_many_parameters(
    param1, param2, param3, param4, param5
)

# または変数に分割
params = {
    'param1': param1,
    'param2': param2,
    'param3': param3,
}
result = process_very_long_function_name_with_many_parameters(**params)
```

### ❌ エラー3: DuckDB並行チェック "High: ThreadPoolExecutor内でDB write"
```python
# ❌ NG: 並列フェーズで直接DB write
with ThreadPoolExecutor(max_workers=4) as executor:
    for symbol in symbols:
        executor.submit(save_to_db, symbol)

# ✅ OK: 2フェーズ化（並列データ処理 + 順序 DB write）
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_symbol, symbols))

# フェーズ2: 順序実行
for symbol, data in results:
    save_to_db(symbol, data)
```

### ❌ エラー4: ファイルロック "High: ThreadPoolExecutor + to_csv"
```python
# ❌ NG
with ThreadPoolExecutor(max_workers=4) as executor:
    for symbol in symbols:
        executor.submit(df.to_csv, f"data/{symbol}.csv")

# ✅ OK
dfs = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    dfs = dict(executor.map(fetch_and_process, symbols))

# ファイル write は順序実行
for symbol, df in dfs.items():
    df.to_csv(f"data/{symbol}.csv")
```

### ❌ エラー5: secrets が コミットされた
```bash
# 検出時の処理
git reset HEAD python/.env  # ステージング解除
echo "python/.env" >> .gitignore
git commit -m "chore: .envを除外設定"

# または過去コミットから削除（git管理から）
git filter-branch --tree-filter 'rm -f python/.env'
```

---

## コードレビュー効率化

### コマンド集

```bash
# すべてのチェック一括実行
.\.venv\Scripts\python -m pre_commit run --all-files

# 特定ファイルのみレビュー
.\.venv\Scripts\python -m pre_commit run --files python/src/services/data_pipeline.py

# DuckDB並行問題のみ確認
.\.venv\Scripts\python .\.github\hooks\check_duckdb_concurrency.py python/src/

# ファイルロック問題のみ確認
.\.venv\Scripts\python .\.github\hooks\check_file_lock.py python/src/

# テスト + カバレッジ表示
.\.venv\Scripts\python -m pytest tests/unit/ -v --cov=python/src --cov-report=html

# Git差分を詳細表示
git diff HEAD~1 --unified=5

# コミット予定を表示
git status --short
```

### VS Code 拡張推奨

- **Pylance** - Python 型チェック・インテリセンス
- **Pylint extension** - コード品質メトリクス表示
- **GitLens** - コミット履歴・責任者表示
- **DuckDB** - SQL構文チェック

---

## チェック リスト（コミット前）

コミット時に以下を確認：

```bash
# 1. コード品質チェック
☐ .\.venv\Scripts\python -m black --check --line-length=100 python/src
☐ .\.venv\Scripts\python -m flake8 python/src --max-line-length=100
☐ .\.venv\Scripts\python -m mypy python/src --ignore-missing-imports

# 2. ロック問題検出
☐ .\.venv\Scripts\python .\.github\hooks\check_duckdb_concurrency.py python/src
☐ .\.venv\Scripts\python .\.github\hooks\check_file_lock.py python/src

# 3. テスト実行
☐ .\.venv\Scripts\python -m pytest tests/unit/ -v
☐ .\.venv\Scripts\python -m pytest tests/integration/ -v

# 4. コミットメッセージ格式確認
☐ feat/fix/docs/style/refactor/test/chore で開始
☐ 詳細な変更内容を含む
☐ 可能であれば Closes #issue_number を記載

# 5. Git 確認
☐ git status --short で意図しないファイル変更がないか確認
☐ .env / secrets.* はコミットされていないか確認
```

---

## 参照ドキュメント

| ドキュメント | 説明 |
|------------|------|
| [LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) | DuckDB/ファイルロック問題の詳細ガイド |
| [PRE_COMMIT_GUIDE.md](../../docs/PRE_COMMIT_GUIDE.md) | Pre-commit フック全体の使用方法 |
| [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) | システムアーキテクチャ・レイヤー構造 |
| [copilot-instructions.md](../copilot-instructions.md) | コード標準・禁止事項 |

---

## よくある質問 (FAQ)

**Q: コードレビューはいつ実行するべき？**

A: コミット前に必ず実行してください。Pre-commit フックが自動で検出するため、意識する必要はありません。

```bash
git add .
git commit -m "feat: ..."  # → 自動的にチェック実行
```

**Q: High警告が出た場合、コミットしても良い？**

A: 原則として**修正すべき**です。やむを得ずスキップする場合：

```bash
git commit --no-verify -m "feat: ..."  # Skip all hooks

# または特定フックをスキップ（推奨）
SKIP=check-duckdb-concurrency git commit -m "feat: ..."
```

**Q: テストはコードレビューに含まれる？**

A: はい。テスト実行はコードレビューの一部です。

```bash
# ユニットテスト（開発中に常に実行）
.\.venv\Scripts\python -m pytest tests/unit/ -v

# 統合テスト（PR作成前）
.\.venv\Scripts\python -m pytest tests/integration/ -v

# カバレッジ確認
.\.venv\Scripts\python -m pytest tests/ --cov=python/src
```

**Q: 型チェックで "Unresolved import" エラーが出た**

A: 以下を実施してください：

```bash
# 1. 型スタブをインストール
.\.venv\Scripts\python -m pip install types-requests types-PyYAML pandas-stubs

# 2. mypy 実行時に型チェック対象外に設定
.\.venv\Scripts\python -m mypy python/src --ignore-missing-imports
```

**Q: 既存コードに違反が多い場合は？**

A: 段階的に修正してください：

1. **新規追加コード**: 100% コンプライアンス（必須）
2. **既存コード修正時**: レビュー対象の行のみ修正
3. **技術負債**: チケット化して段階的に対処

---

## 次のステップ

✅ このスキルを完了した後：

1. 変更内容に対して `data-pipeline` / `model-training` スキルを実行
2. バックテスト結果を確認（`backtest` スキル）
3. PR を作成して レビューを受ける（`git-ops` スキル）

