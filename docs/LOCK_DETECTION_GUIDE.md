# コードレビュー - ロック問題自動検出ガイド

## 概要

**DuckDB** と **ファイル操作** における並行実行時のロック問題（デッドロック、排他ロック競合）を、コミット前に**自動で検出・警告**するシステムです。

### 対象範囲

| 問題カテゴリ | 検出スクリプト | 検出対象 |
|-----------|-------------|--------|
| **DB ロック競合** | `check_duckdb_concurrency.py` | スレッド/プロセス + DB write の同時実行 |
| **ファイルロック競合** | `check_file_lock.py` | 並列ファイル操作でのロック漏れ |

---

## DuckDB 並行処理チェック

### 検出対象パターン

#### 1️⃣ **Critical: async/await + DB操作**

```python
# ❌ NG: asyncio は DuckDB でサポートされていない
async def fetch_data():
    con.execute("SELECT * FROM table")  # エラー！
```

**修正方法**: asyncio を使用しない。同期的に実行してください。

#### 2️⃣ **High: ThreadPoolExecutor + DB write**

```python
# ❌ NG: 複数スレッドが同時に execute()
with ThreadPoolExecutor(max_workers=4) as executor:
    for symbol in symbols:
        executor.submit(con.execute, f"INSERT INTO table VALUES ('{symbol}')")
```

**修正方法**:
```python
# ✅ OK: 並列フェーズ（データ処理）と順序フェーズ（DB write）を分離
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_symbol, symbols)  # データ処理は並列

# フェーズ2: DB write は順序実行
for symbol, data in results:
    con.execute(f"INSERT INTO table VALUES ('{symbol}', '{data}')")
```

#### 3️⃣ **High: ProcessPoolExecutor + DB write**

```python
# ❌ NG: 別プロセスから DB 操作
def worker(symbol):
    con.execute(f"INSERT INTO table VALUES ('{symbol}')")

with ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(worker, symbols)
```

**修正方法**: プロセス内でのファイル保存に変更し、メインプロセスで DB write

#### 4️⃣ **High: 並列データフレームアップサート**

```python
# ❌ NG: run_parallel 内で upsert を実行
@run_parallel
def fetch_symbol(symbol):
    df = fetch_data(symbol)
    save_features_to_db(symbol, df)  # ← INSERT OR REPLACE の重複実行
```

**修正方法**: `defer_raw_save=True` で DB write を分離
```python
# ✅ OK: defer_raw_save=True で2フェーズ化
@run_parallel
def fetch_symbol(symbol):
    df = fetch_data(symbol)
    return symbol, df  # データのみ返却

# フェーズ2: 順序実行する
for symbol, df in results:
    save_features_to_db(symbol, df, defer_raw_save=True)
```

#### 5️⃣ **Medium: DELETE + INSERT の非原子実行**

```sql
-- ❌ NG: DELETE と INSERT に隙間がある
DELETE FROM stock_features WHERE symbol = 'AAPL';
INSERT INTO stock_features SELECT ... WHERE symbol = 'AAPL';
```

**修正方法**: `INSERT OR REPLACE` またはトランザクション使用
```sql
-- ✅ OK: 1ステップで完了
INSERT OR REPLACE INTO stock_features SELECT ... WHERE symbol = 'AAPL';

-- または トランザクション
BEGIN TRANSACTION;
DELETE FROM stock_features WHERE symbol = 'AAPL';
INSERT INTO stock_features SELECT ...;
COMMIT;
```

#### 6️⃣ **Low: try-except なしの execute()**

```python
# ⚠️   警告: エラー時にロック解放漏れの可能性
con.execute("INSERT INTO table VALUES (...)")
```

**修正方法**: 例外ハンドリング
```python
# ✅ OK
try:
    con.execute("INSERT INTO table VALUES (...)")
except Exception as e:
    print(f"Error: {e}")
    # ロック自動解放（DuckDB は自動で処理）
```

---

## ファイルロック検出

### 検出対象パターン

#### 1️⃣ **High: ThreadPoolExecutor + CSV 書き込み**

```python
# ❌ NG: 複数スレッドが同時に to_csv()
with ThreadPoolExecutor(max_workers=4) as executor:
    for symbol in symbols:
        executor.submit(df.to_csv, f"data/{symbol}.csv")
```

**修正方法**: ファイル操作を順序実行に変更
```python
# ✅ OK: 並列でデータ処理、順序で書き込み
dfs = []
with ThreadPoolExecutor(max_workers=4) as executor:
    dfs = executor.map(process_symbol, symbols)  # 並列処理

# ファイル書き込みは順序実行
for symbol, df in dfs:
    df.to_csv(f"data/{symbol}.csv")
```

#### 2️⃣ **High: ThreadPoolExecutor + joblib.dump()**

```python
# ❌ NG: 複数スレッドがモデルを同時保存
with ThreadPoolExecutor(max_workers=4) as executor:
    for symbol in symbols:
        executor.submit(joblib.dump, model, f"models/{symbol}.joblib")
```

**修正方法**: 順序実行
```python
# ✅ OK
for symbol, model in models.items():
    joblib.dump(model, f"models/{symbol}.joblib")
```

#### 3️⃣ **Medium: open(..., 'w') での無保護ファイル操作**

```python
# ⚠️  警告: 複数プロセスからのアクセスで破損リスク
with open("data.csv", "w") as f:
    f.write(data)
```

**修正方法**: fcntl.flock() または Lock を使用
```python
import fcntl

# ✅ OK: ファイルロック使用
with open("data.csv", "w") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
    try:
        f.write(data)
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# または threading.Lock
import threading
file_lock = threading.Lock()

with file_lock:
    with open("data.csv", "w") as f:
        f.write(data)
```

#### 4️⃣ **Medium: to_csv() → read_csv() の競合**

```python
# ⚠️  警告: read_csv() が不完全なファイルを読む可能性
df.to_csv("output.csv")
df2 = pd.read_csv("output.csv")  # to_csv() が完了していない？
```

**修正方法**: 同期化フラグまたはロック
```python
import time

# ✅ OK: to_csv() 後に同期確認
df.to_csv("output.csv")
while not os.path.exists("output.csv") or os.path.getsize("output.csv") == 0:
    time.sleep(0.1)

df2 = pd.read_csv("output.csv")
```

#### 5️⃣ **Low: shutil.move/copy での並列実行**

```python
# ⚠️  警告: 並列環境では排他性が保証されない
shutil.move("temp/file.csv", "data/file.csv")
```

**修正方法**: 順序実行またはロック機構
```python
# ✅ OK: Lock で保護
import shutil
import threading

file_op_lock = threading.Lock()

with file_op_lock:
    shutil.move("temp/file.csv", "data/file.csv")
```

---

## 使用方法

### 1. 自動実行（コミット時）

```bash
git add .
git commit -m "feat: データパイプライン改善"
```

コミット時に自動的にチェックが実行されます。

### 2. 手動実行

```bash
# すべてのファイルをチェック
pre-commit run --all-files

# 特定のフックのみ実行
pre-commit run check-duckdb-concurrency --all-files
pre-commit run check-file-lock --all-files
```

### 3. スキップ（非推奨）

```bash
# 一度のコミット時にスキップ
git commit --no-verify

# フック自体を無効化（非推奨）
pre-commit uninstall
```

---

## チェック結果の解釈

### 重大度レベル

| レベル | 説明 | アクション |
|------|------|---------|
| 🔴 **Critical** | 即座に実行エラーになる | **コミットブロック** - 必須修正 |
| 🟠 **High** | ロック競合が高確率で発生 | **コミットブロック** - 必須修正 |
| 🟡 **Medium** | リスク状況である | **警告表示** - 修正推奨 |
| 🔵 **Low** | ベストプラクティス逸脱 | **情報表示** - 参考値 |

### 出力例

```
python/src/services/data_pipeline.py:45: [HIGH] 警告: ThreadPoolExecutor内でfetch_stock_data_with_featuresを呼んでいるが、defer_raw_save=True が指定されていない可能性。並列フェーズからDB書き込みを分離してください。

python/src/data/data_loader.py:123: [MEDIUM] ⚠️  並列処理がありますがロック機構(Lock/fcntl.flock)が見当たりません。ファイルロック競合の可能性があります。

===========================================================================
DuckDB 並行処理チェック結果
===========================================================================
Critical: 0
High:     1
Medium:   1
Low:      0

⚠️  High リスクの警告があります。修正を推奨します。
```

---

## 修正テンプレート

### パターン1: 並列→順序フェーズの分離

```python
# Phase 1: 並列処理（ネットワーク I/O, データ処理）
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_and_process, symbols))

# Phase 2: 順序実行（DB write, ファイル save）
for symbol, data in results:
    save_to_database(symbol, data)
    save_to_csv(symbol, data)
```

### パターン2: defer_raw_save の活用

```python
# DuckDB への raw OHLCV save を遅延させる
market, symbol, features_df, raw_data_to_save = fetch_stock_data_with_features(
    market, symbol, defer_raw_save=True
)

# 後で順序実行
con.execute(...)  # 優先度高い操作
save_raw_ohlcv(market, symbol, raw_data_to_save)  # 遅延した save
```

### パターン3: ロック機構の導入

```python
import threading

# グローバルロック（モジュール初期化時）
db_write_lock = threading.Lock()
file_io_lock = threading.Lock()

# DB 操作時
with db_write_lock:
    con.execute("INSERT INTO table VALUES (...)")

# ファイル操作時
with file_io_lock:
    df.to_csv("output.csv")
    cache = pd.read_csv("output.csv")
```

---

## FAQ

**Q: 警告が出ているがコミットしたい場合は？**

A: 以下の対応をしてください：
```bash
# 1. 修正を加える（推奨）
git add .
git commit -m "fix: ロック競合を修正"

# 2. やむを得ずスキップ（非推奨・要理由記載）
git commit --no-verify -m "feat: 一時的にスキップ、チケット#123で対応予定"
```

**Q: False Positive（誤検出）が発生している場合は？**

A: `.pre-commit-config.yaml` の対象ファイルを調整してください：
```yaml
exclude: ^(tests/|docs/|config/)?
files: ^python/src/
```

**Q: 自分のチェック基準を追加したい場合は？**

A: `.github/hooks/check_duckdb_concurrency.py` の `danger_patterns` 辞書にパターンを追加：
```python
self.danger_patterns["my_pattern"] = {
    "pattern": r"my_regex_pattern",
    "message": "My custom message",
    "severity": "high",
}
```

---

## 参照ドキュメント

- [OPERATIONS.md](../OPERATIONS.md) - データベース競合対策セクション
- [DOCKER_DB_ARCHITECTURE.md](../DOCKER_DB_ARCHITECTURE.md) - バッチ差分更新の実装ポリシー
- [PRE_COMMIT_GUIDE.md](../PRE_COMMIT_GUIDE.md) - Pre-commit フック全体ガイド
- [copilot-instructions.md](../.github/copilot-instructions.md) - レイヤー構造・禁止事項
