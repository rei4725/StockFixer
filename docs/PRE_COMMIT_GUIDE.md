# Pre-Commit 自動コードレビュー ガイド

StockFixer では、Git コミット前に自動的にコード品質チェック・型検証・フォーマット検査を行う仕組みを導入しています。

---

## セットアップ

### 前提条件
- Python 3.8以上がインストール済み
- `pip` が使用可能

### インストール手順

#### 1. 依存パッケージのインストール
```powershell
cd C:\src\StockFixer\python

# 仮想環境が有効化済みの場合
pip install -r requirements.txt

# 仮想環境がない場合
py -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

#### 2. Pre-commit hooksの登録
```powershell
cd C:\src\StockFixer

# 標準的なpre-commitフックをインストール
pre-commit install

# コミットメッセージ検証もインストール
pre-commit install --hook-type commit-msg
```

> **確認方法**:
> ```powershell
> ls .git\hooks\
> ```
> `pre-commit` と `commit-msg` ファイルが作成されていれば成功

---

## チェック内容

### 1. コードフォーマット（Black）
PEP 8 スタイルに従うようコードを自動修正します。

```python
# 修正前
x=1+2
y=func(a,b,c)

# 修正後
x = 1 + 2
y = func(a, b, c)
```

更新: `--line-length=100`

### 2. インポート整理（isort）
Python importをアルファベット順に自動整理します。

```python
# 修正前
from src.utils import db
import os
import pandas as pd
from src.api import discord_utils

# 修正後
import os

import pandas as pd

from src.api import discord_utils
from src.utils import db
```

### 3. PEP 8 コンプライアンス（Flake8）
違反を検出し、報告します（自動修正ではなく確認）。

**検出例**:
- 行の長さが100文字超過
- トレーリング空白
- 複数行間隔
- 未使用 import・変数

### 4. 型安全性（mypy）
Python型ヒントをチェックします（テストファイル除外）。

```python
# エラー例
def add (a: int, b: int) -> int:
    return a + b

result: str = add(1, 2)  # ❌ int を str に割り当て
```

### 5. コード品質（pylint - 軽量版）
致命的なエラーのみ検出（完全なチェックはCI/CDで実施）。

### 6. 基本ファイル整備（pre-commit-hooks）
- 末尾の改行修正
- トレーリング空白削除
- 大ファイル（5MB超）のブロック
- YAML/JSON構文検証
- マージコンフリクト残存検出

### 7. コミットメッセージ形式（カスタムスクリプト）
Conventional Commits スタイルを検証します。

---

## 使用方法

### 通常のコミット（自動レビュー実行）

```powershell
# ステージング
git add python/run_data_creation.py

# コミット （自動レビューが走る）
git commit -m "fix(data-pipeline): エラーハンドリング改善"
```

**フロー**:
1. コミットコマンド実行
2. Pre-commit hooks が自動実行
3. エラー検出 → 該当ファイルを修正表示
4. ユーザーが対応 → `git add` & `git commit` を再実行

### エラーが出た場合

```powershell
# 例: Black がコードを修正していた場合
# → git add で再ステージング
git add python/run_data_creation.py
git commit -m "fix(data-pipeline): エラーハンドリング改善"
```

### 手動でチェック実行

```powershell
# 全ファイルをチェック
pre-commit run --all-files

# 特定のファイルのみ
pre-commit run --files python/run_data_creation.py

# 特定のhookのみ
pre-commit run black --all-files
pre-commit run mypy --all-files
pre-commit run flake8 --all-files
```

### レビューを一時スキップ（緊急時のみ）

```powershell
git commit --no-verify -m "fix: 緊急パッチ"
```

> ⚠️ 原則として使用を避けてください。PRレビュー時に指摘が増えます。

### フックを無効化

```powershell
# 一時停止
pre-commit uninstall

# 再度有効化
pre-commit install
pre-commit install --hook-type commit-msg
```

---

## コミットメッセージ形式

### 基本形式

```
<type>(<scope>): <subject>

<body（オプション）>
<footer（オプション）>
```

### Type（タイプ）

| タイプ | 説明 | 例 |
|--------|------|-----|
| `feat` | **新機能** | feat(data-pipeline): 株価取得エラーハンドリング追加 |
| `fix` | **バグ修正** | fix(scheduler): タイムゾーン設定バグを修正 |
| `docs` | **ドキュメント** | docs(README): セットアップ手順を追加 |
| `style` | **コード整形** | style: インポートのソート |
| `refactor` | **リファクタリング** | refactor(models): 予測モデル構造を整理 |
| `perf` | **パフォーマンス改善** | perf(db): クエリを最適化 |
| `test` | **テスト追加・修正** | test: データパイプラインテストを追加 |
| `ci` | **CI/CD設定** | ci: GitHub Actionsワークフロー追加 |
| `chore` | **その他** | chore: 依存パッケージを更新 |
| `revert` | **コミット取り消し** | revert: "feat: XYZ機能" を取り消し |

### Scope（スコープ）

括弧内に機能領域を指定（推奨）。

| スコープ | 説明 | 対応ファイル |
|---------|------|-----------|
| `data-pipeline` | データ取得・更新 | `src/services/data_pipeline.py`, `src/data/` |
| `scheduler` | スケジューラー定期実行 | `src/services/scheduler_*.py` |
| `models` | AIモデル | `src/models/`, `python/models/` |
| `db` | DuckDB | `src/utils/db.py` |
| `api` | Discord Bot / REST API | `src/api/` |
| `docker` | Docker設定 | `Dockerfile`, `docker-compose.yml` |
| `tests` | テスト | `tests/` |
| `docs` | ドキュメント | `docs/`, README |

### Subject（主題）

- 50～100文字
- 命令形で簡潔に（「〜する」ではなく「〜した」）
- 大文字で開始（推奨）
- ピリオドで終わらない

### Body（本文）

詳細な説明が必要な場合のみ記述。

```
feat(data-pipeline): 差分更新時のDB書き込みを逐次化

コミット前の自動レビューを導入し、並列フェーズでの
DB書き込み競合を回避した。

- フェーズ1（並列）: データ取得・特徴量生成のみ
- フェーズ2（逐次）: market_data_raw と stock_features 保存
```

### 良い例

```
feat(data-pipeline): 差分更新時のDB書き込みを逐次化
fix(scheduler): タイムゾーン設定バグを修正
docs: README のセットアップ手順を追加
refactor(models): 予測モデルクラスを整理
test: 統合テストとユニットテストを分離
```

### 悪い例

```
update code            # ❌ 曖昧
Fix bug               # ❌ 型・スコープなし
データ処理修正         # ❌ 英語ではない
fix #123              # ❌ コミットメッセージがない
```

---

## トラブルシューティング

### Q: 「hook file or executable not found」エラー
```
Error: hook id 'check-commit-message' hook file or executable not found
```

**原因**: `.github/hooks/check_commit_message.py` がない

**対策**:
```powershell
# ファイルが存在するか確認
ls .github\hooks\

# なければ作成（既に配置済みなら不要）
```

### Q: mypy のエラーが出続ける

**原因**: 型情報が不完全

**対策**:
```powershell
# 緩和版で実行
pre-commit run mypy --all-files -- --ignore-missing-imports

# または .pre-commit-config.yaml の mypy セクションを調整
```

### Q: 「フォーマット修正が多すぎてコミット困難」

**原因**: prototypeコードで型やスタイルが不統一

**対策**:
```powershell
# 1. すべてを一度修正
pre-commit run --all-files

# 2. 修正内容を確認
git diff

# 3. 修正変更をステージング
git add <modified_files>

# 4. コミット
git commit -m "style: コード整形"
```

### Q: 特定のテストをスキップしたい

`.pre-commit-config.yaml` の該当フックの `exclude` を調整:

```yaml
- id: mypy
  exclude: ^(tests/|.*_test\.py$|prototypes/)
```

### Q: 「Pre-commit hooksをリセットしたい」

```powershell
# アンインストール
pre-commit uninstall

# 再インストール
pre-commit install
pre-commit install --hook-type commit-msg
```

---

## 設定ファイル参照

### `.pre-commit-config.yaml`
プロジェクトルートに配置。各フックの設定・バージョン・条件を定義。

### `.github/hooks/check_commit_message.py`
コミットメッセージフォーマット検証スクリプト。

### `python/requirements.txt`
`pre-commit`, `black`, `flake8`, `mypy` 等を列挙。

---

## 運用ポリシー

### 開発時
- ローカルでコミット前にhooksが実行される
- エラーが出ても修正・再コミットで対応可能

### PR時
- GitHub Actions で再度 pre-commit を実行（オプション）
- チェック漏れを防止

### 本番デプロイ前
- すべてのチェックをパスしたコードのみをマージ

---

## 参考資料

- [Pre-commit 公式ドキュメント](https://pre-commit.com/)
- [Black - Python コードフォーマッター](https://github.com/psf/black)
- [Flake8 - Python リンター](https://flake8.pycqa.org/)
- [mypy - Python 型チェッカー](http://mypy-lang.org/)
- [Conventional Commits](https://www.conventionalcommits.org/ja/)

---

*Last updated: 2026-03-02*
