---
name: model-training
description: "AIモデルの学習・保存を行う。銘柄別モデル学習と統合モデル学習の両方に対応。モデル学習、モデル作成、XGBoost、LightGBM、統合モデル、unified model、train、学習の話題で使用する。"
metadata:
  author: StockFixer
  version: "1.0"
compatibility: "Python 3.10+, XGBoost, LightGBM, joblib。python/ ディレクトリで実行。"
---

## Goal
XGBoost・LightGBMモデルを学習し、joblib形式で保存する。

## Procedure

### 銘柄別モデル学習
```bash
cd python
# 単一銘柄
py run_model_creation.py --market jp --symbol 7203

# ウォッチリスト全銘柄バッチ
py run_model_creation.py --batch
```
- XGBoost + LightGBM 両方を自動学習
- 保存先: `python/models/[market]_[symbol]/Stock{XGBoost,LightGBM}Model.joblib`
- バッチ処理: フェーズ1でDB読み込み（並列）、フェーズ2で学習・保存（逐次）

### 統合モデル学習
```bash
cd python
# XGBoost + LightGBM 両方（デフォルト）
py run_unified_model_training.py

# 片方のみ
py run_unified_model_training.py --model-type LightGBMModel --no-both
```
- 全銘柄のデータを結合して1つのモデルを学習（汎化性能重視）
- 保存先: `python/models/unified/UnifiedStock{XGBoost,LightGBM}.joblib`

### 使い分け基準
| 方式 | 適用場面 | メリット | デメリット |
|------|---------|---------|-----------|
| 統合モデル | 通常運用（デフォルト） | 汎化性能が高い、管理が容易 | 個別銘柄特性の捕捉が弱い |
| 銘柄別モデル | 特定銘柄の精度最大化 | 銘柄固有パターン学習 | 管理コスト大、データ不足リスク |

### 内部処理フロー
1. `load_features_for_training()` でDB読み込み＋特徴量/ターゲット分離
2. `ModelManager` でモデルインスタンス生成
3. `.fit(X_train, y_train)` でモデル学習
4. `joblib.dump()` でモデル保存

## Key Functions
- `train_models_for_symbol(market, symbol)` — 銘柄別のXGBoost＋LightGBM学習・保存
- `train_unified_model(model_type, model_name)` — 統合モデル学習・保存
- `run_model_batch()` — ウォッチリスト全銘柄モデル作成

## References
- [model_training_pipeline.py](../../../python/src/services/model_training_pipeline.py)
- [unified_model_pipeline.py](../../../python/src/services/unified_model_pipeline.py)
- [model_manager.py](../../../python/src/models/model_manager.py)
