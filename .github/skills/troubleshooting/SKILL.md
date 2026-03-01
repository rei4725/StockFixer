---
name: troubleshooting
description: "よくあるエラーの原因と対処法を提供する。エラー、トラブル、障害、バグ、動かない、失敗、タイムアウト、DuckDBロック、yfinanceエラー、モデルが見つからない、Discord送信エラー、メモリ不足の話題で使用する。"
metadata:
  author: StockFixer
  version: "1.0"
---

## Goal
StockFixerで発生しやすいエラーの原因特定と解決を迅速に行う。

## Troubleshooting Guide

### DuckDB書込エラー（ロック競合）
**症状**: `IOException: Could not set lock on file` 等
**原因**: 複数プロセス/スレッドから同時にDB書込を試みている
**対処**:
1. DB書込は必ず逐次実行にする（並列NG）
2. バッチ処理のフェーズ2（DB書込）は逐次設計になっているか確認
3. 別プロセスからの読み取りは `get_readonly_connection()` を使用
4. 書込中のプロセスが残っていないか確認してから再実行

### yfinanceエラー
**症状**: `No data found` / `Ticker not found` / `Too Many Requests` / `Rate limited` 等
**原因**: ティッカー不正、API制限、上場廃止、ネットワークエラー
**対処**:
1. 日本株は `.T` サフィックスが必要 → `get_ticker()` で自動補正される
2. 二重サフィックス（`7203.T.T`）になっていないか確認
3. **レート制限エラー**（`Too Many Requests`, `Rate limited`）
   - **自動対応**: リトライロジックが指数バックオフで自動的に待機・再試行
   - **待機時間**: 初期2秒から最大60秒まで段階的に増加
   - **最大リトライ**: 5回まで自動リトライ
   - コンソール出力で待機内容を表示 → 「待機中: 10秒... 9秒...」のようにカウントダウン
   - 複数銘柄取得時は各銘柄のリトライが独立（全体が止まらない）
4. 上場廃止銘柄は `データ取得対象.csv` から削除

### モデルが見つからない
**症状**: `FileNotFoundError` / モデルロード失敗
**原因**: モデルが未学習、パスが不正
**対処**:
1. 銘柄別モデル: `python/models/[market]_[symbol]/` にjoblibファイルがあるか確認
2. 統合モデル: `python/models/unified/` にUnifiedStock{XGBoost,LightGBM}.joblibがあるか確認
3. モデルが存在しない場合は学習を実行:
   - 銘柄別: `py run_model_creation.py --market jp --symbol 7203`
   - 統合: `py run_unified_model_training.py`

### Discord送信エラー
**症状**: メッセージ送信失敗、文字数超過
**原因**: Webhook URL未設定、メッセージ長超過
**対処**:
1. Webhook URLが環境変数に正しく設定されているか確認
2. メッセージ長制限: 2000文字 → 分割送信処理あり
3. BOTトークンは `DISCORD_BOT_TOKEN` 環境変数

### メモリ不足
**症状**: `MemoryError` / プロセスキル
**原因**: 統合モデル学習時に全銘柄データを結合するため
**対処**:
1. DuckDB接続のmemory_limitを確認（デフォルト2GB）
2. 監視銘柄数を削減して再実行
3. Docker環境ではコンテナのメモリ制限を増加

### importエラー
**症状**: `ModuleNotFoundError`
**原因**: importパスの不一致、`__init__.py`の欠落
**対処**:
1. 実行は `python/` ディレクトリから行う
2. importパスは `python/` からの絶対パスで統一（例: `from src.utils.db import ...`）
3. 各ディレクトリに `__init__.py` が配置されているか確認

## References
- [db.py](../../../python/src/utils/db.py)
- [data_path_utils.py](../../../python/src/utils/data_path_utils.py)
