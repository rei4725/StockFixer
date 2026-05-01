AI + 自動チェック機構を活用した包括的なコードレビューを実行する。

**検証対象**: コード品質・DuckDB並行ロック問題・セキュリティ・パフォーマンス・テスト可能性

## ステップ1: 自動チェック実行
```bash
cd C:\src\StockFixer

# 全フックを実行（自動修正も含む）
.\.venv\Scripts\python -m pre_commit run --all-files

# 個別実行
.\.venv\Scripts\python -m flake8 python/src --max-line-length=100 --show-source
.\.venv\Scripts\python -m mypy python/src --ignore-missing-imports
.\.venv\Scripts\python -m pylint python/src --exit-zero
```

## ステップ2: ロック問題検出
```bash
# DuckDB 並行処理チェック
.\.venv\Scripts\python .\.github\hooks\check_duckdb_concurrency.py python/src/services/data_pipeline.py python/src/services/batch_runner.py

# ファイルロック検出
.\.venv\Scripts\python .\.github\hooks\check_file_lock.py python/src/data/*.py python/src/services/*.py
```

**DuckDB チェック対象パターン:**
| Severity | パターン | 対策 |
|---|---|---|
| Critical | `async def` + DB操作 | asyncio を使用しない |
| High | `ThreadPoolExecutor` + `execute()` | 2フェーズ化（並列I/O + 順序DB write） |
| High | 並列 `upsert_raw_ohlcv()` | `defer_raw_save=True` フラグ使用 |

## ステップ3: レビューチェックリスト

**アーキテクチャ:**
- [ ] レイヤー構造遵守（run→api→services→models→features→data→utils）
- [ ] 上位層が下位層のみを参照（逆参照なし）
- [ ] 層をまたぐデータは各BCの `types.py` の dataclass で受け渡す（生 dict 禁止）

**コード品質:**
- [ ] 関数が単一責任を持つ（30行以下が目安）
- [ ] `except Exception: pass` は禁止（必ず `logger.error("...", exc_info=True)`）
- [ ] `run_*.py` にビジネスロジックがない

**セキュリティ:**
- [ ] `.env` / API キー・パスワードをハードコードしない
- [ ] SQL インジェクション対策（パラメータ化クエリ）

**テスト可能性:**
- [ ] 外部依存（DB・API）がモック可能な設計
- [ ] カバレッジ 80% 以上

## ステップ4: 自動修正 & 再検証
```bash
.\.venv\Scripts\python -m black --line-length=100 python/src
.\.venv\Scripts\python -m isort --profile=black python/src
.\.venv\Scripts\python -m pytest tests/unit/ -v
```

## よくあるエラーパターン

### DuckDB 並行問題（High）
```python
# NG: 並列フェーズで直接 DB write
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(save_to_db, symbols)

# OK: 2フェーズ化
results = list(executor.map(process_symbol, symbols))  # フェーズ1: 並列処理
for symbol, data in results:
    save_to_db(symbol, data)                           # フェーズ2: 逐次write
```

### ファイルロック問題（High）
```python
# NG: 並列 to_csv
with ThreadPoolExecutor() as ex:
    ex.map(lambda s: df.to_csv(f"{s}.csv"), symbols)

# OK: 逐次write
dfs = dict(executor.map(fetch_and_process, symbols))
for symbol, df in dfs.items():
    df.to_csv(f"{symbol}.csv")
```

## コミット前チェックリスト
```bash
pre-commit run --all-files
python -m pytest tests/unit/ -v
git status  # .env 等が含まれていないか確認
```
