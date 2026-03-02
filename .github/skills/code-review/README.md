# コードレビュー スキル一覧

このディレクトリは StockFixer のコードレビュー関連スキルを管理します。

## 📋 スキル概要

### `code-review` - コードレビュー実行ガイド

**目的**: AI + 自動チェック機構による包括的なコードレビュー

**対象チェック:**
- ✅ コード品質（Black, Flake8, isort, pylint）
- ✅ 型安全性（mypy）
- ✅ DuckDB並行処理問題の検出
- ✅ ファイルロック問題の検出
- ✅ テスト実行とカバレッジ確認
- ✅ セキュリティチェック

**実行方法:**
```bash
# 完全レビュー（品質 + ロック + テスト）
python python/run_code_review.py

# クイックレビュー（品質 + ロック検出のみ）
.venv\Scripts\python -m pre_commit run --all-files

# 特定ファイルのレビュー
python python/run_code_review.py python/src/services/data_pipeline.py

# 自動修正を実行
python python/run_code_review.py --fix

# オプション:
#   --quality : コード品質チェックのみ
#   --locks   : ロック問題検出のみ
#   --tests   : テストのみ実行
#   --full    : 統合テスト + カバレッジを含む
```

**主な検出項目:**
| 検出種別 | ツール | 重大度 | アクション |
|---------|--------|--------|-----------|
| 型エラー | mypy | High | 修正必須 |
| 並行競合 | check-duckdb-concurrency | High | 修正必須 |
| ファイルロック | check-file-lock | High | 修正必須 |
| PEP8違反 | Flake8 | Medium | 修正推奨 |
| 行長超過 | Black | Low | 修正推奨 |

**関連ドキュメント:**
- [SKILL.md](./SKILL.md) - 詳細なコードレビューガイド
- [../../docs/LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) - ロック問題の詳細
- [../../docs/PRE_COMMIT_GUIDE.md](../../docs/PRE_COMMIT_GUIDE.md) - Pre-commit フックガイド

---

## 🚀 クイックスタート

### 1️⃣ 初回セットアップ
```bash
# Pre-commit フックをインストール
.venv\Scripts\python -m pre_commit install
.venv\Scripts\python -m pre_commit install --hook-type commit-msg
```

### 2️⃣ コミット前のレビュー
```bash
# 変更をステージング
git add .

# コードレビュー実行（自動修正を含む）
python python/run_code_review.py --fix

# コミット（フックが自動実行）
git commit -m "feat: <説明>"
```

### 3️⃣ PR 前の確認
```bash
# 完全なレビュー（統合テスト + カバレッジ）
python python/run_code_review.py --full
```

---

## 📊 チェック詳細

### Black/isort（コードフォーマット）
- **対象**: 行長100文字、PEP8準拠
- **自動修正**: ✅あり（`--fix` オプション）
- **失敗時の対策**: 自動修正を実行後、再度チェック

### Flake8（PEP8準拠チェック）
- **対象**: スタイル違反、未使用インポート等
- **自動修正**: ❌なし（手動対応必要）
- **失敗時の対策**: エラーメッセージに従い手動修正

### mypy（型安全性チェック）
- **対象**: 型ヒントの不整合、Optional型の未指定
- **自動修正**: ❌なし（型ヒント追加が必要）
- **失敗時の対策**: [LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) の「mypy対策」を参照

### check-duckdb-concurrency（DB並行問題）
- **対象**: ThreadPoolExecutor + execute()、DELETE + INSERT等
- **自動修正**: ❌なし（アーキテクチャ改善が必要）
- **失敗時の対策**: [LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) の「修正テンプレート」を参照

### check-file-lock（ファイルロック問題）
- **対象**: 並列ファイル操作、to_csv()競合等
- **自動修正**: ❌なし（ロック機構追加が必要）
- **失敗時の対策**: ファイル操作を順序実行に変更、またはLock機構を追加

### pytest（テスト実行）
- **対象**: ユニットテスト、統合テスト
- **カバレッジ**: 80% 以上推奨
- **失敗時の対策**: test ファイルを修正、または実装コードの修正

---

## 🔍 レビュー結果の解釈

### ✅ PASS（グリーン）
全チェックが正常に完了。PR作成可能です。

### ❌ FAIL（赤）
以下のいずれかの対応が必要：

**High重大度の場合（修正必須）:**
```bash
# 1. 自動修正が可能な場合
python python/run_code_review.py --fix
git add .
git commit -m "style: 自動修正"

# 2. 手動修正が必要な場合
# → [LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) を参照
git add .
git commit -m "fix: ロック問題を修正"
```

**Medium以下の場合（修正推奨）:**
修正するか、やむを得ずスキップ可能：
```bash
# スキップしてコミット（非推奨）
SKIP=check-duckdb-concurrency git commit -m "feat: ..."
```

---

## 🎯 コードレビュー チェックリスト

コミット時の確認項目：

```
品質チェック
☐ Black/isort/Flake8 パス
☐ mypy 型チェック パス
☐ pylint 品質スコア確認

ロック問題
☐ DuckDB並行チェック パス
☐ ファイルロック検出 パス

テスト
☐ ユニットテスト 実行完了
☐ テストカバレッジ 80% 以上
☐ 統合テスト 実行完了（PR前）

セキュリティ
☐ .env / secrets.* をコミットしない
☐ 機密情報（APIキー）はハードコーディングしない
☐ ユーザー入力を検証している

アーキテクチャ
☐ レイヤー構造（run → api → services → models → features → data → utils）を遵守
☐ 上位層が下位層のみを参照
☐ 循環依存がない

コミットメッセージ
☐ feat/fix/docs/style/refactor/test/chore のいずれかで開始
☐ 詳細な変更内容を記載
☐ 可能であれば Closes #issue_number を記載
```

---

## 🛠️ トラブルシューティング

**Q: "mypy: Unresolved import" エラー**
```bash
# 型スタブをインストール
.venv\Scripts\python -m pip install types-requests types-PyYAML pandas-stubs
```

**Q: Flake8 の line-too-long (E501) 警告**
```python
# 関数を複数行に分割
result = process_data(
    param1, param2, param3,  # 100文字以内
)
```

**Q: pytest でカバレッジが 80% 未満**
```bash
# カバレッジレポートを確認
.venv\Scripts\python -m pytest tests/ --cov=python/src --cov-report=html
# htmlcov/index.html をブラウザで開く
```

**Q: DuckDB並行チェック High警告**
→ [LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) の「修正テンプレート」を参照

---

## 📚 参考リソース

| リソース | 説明 |
|---------|------|
| [SKILL.md](./SKILL.md) | コードレビューの詳細ガイド |
| [LOCK_DETECTION_GUIDE.md](../../docs/LOCK_DETECTION_GUIDE.md) | ロック問題の検出・修正方法 |
| [PRE_COMMIT_GUIDE.md](../../docs/PRE_COMMIT_GUIDE.md) | Pre-commit フックの全体ガイド |
| [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) | システムアーキテクチャ |
| [copilot-instructions.md](../copilot-instructions.md) | コーディング標準・禁止事項 |
| [run_code_review.py](../../python/run_code_review.py) | コードレビュー実行スクリプト |

---

## 🎓 次のステップ

✅ コードレビューが完了した後：

1. **データパイプラインの実行**: `data-pipeline` スキル
2. **モデル学習**: `model-training` スキル
3. **バックテスト**: `backtest` スキル
4. **Git操作・PR作成**: `git-ops` スキル

---

**最終更新**: 2026年3月2日
