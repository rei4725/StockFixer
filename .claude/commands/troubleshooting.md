StockFixer で発生しやすいエラーの原因特定と解決を迅速に行う。

## DuckDB 書込エラー（ロック競合）
**症状**: `IOException: Could not set lock on file`
**原因**: 複数プロセス/スレッドから同時に DB 書込を試みている
**対処**:
1. DB書込は必ず逐次実行にする（並列NG）
2. バッチ処理のフェーズ2（DB書込）は逐次設計になっているか確認
3. 別プロセスからの読み取りは `get_readonly_connection()` を使用
4. 書込中のプロセスが残っていないか確認してから再実行

## yfinance エラー
**症状**: `No data found` / `Ticker not found` / `Too Many Requests`
**対処**:
1. 日本株は `.T` サフィックスが必要 → `get_ticker()` で自動補正される
2. 二重サフィックス（`7203.T.T`）になっていないか確認
3. レート制限エラーは自動対応（指数バックオフ、最大5回リトライ）
4. 上場廃止銘柄は `データ取得対象.csv` から削除

## モデルが見つからない
**症状**: `FileNotFoundError` / モデルロード失敗
**対処**:
1. 銘柄別モデル: `python/models/[market]_[symbol]/` に joblib ファイルがあるか確認
2. 統合モデル: `python/models/unified/UnifiedStock{XGBoost,LightGBM}.joblib` があるか確認
3. モデルが存在しない場合は学習を実行:
   ```bash
   py run_model_creation.py --market jp --symbol 7203  # 銘柄別
   py run_unified_model_training.py                     # 統合
   ```

## Discord 送信エラー
**症状**: メッセージ送信失敗、文字数超過
**対処**:
1. `DISCORD_WEBHOOK_URL` / `DISCORD_BOT_TOKEN` が環境変数に正しく設定されているか確認
2. メッセージ長制限 2000文字 → 分割送信処理あり

## メモリ不足
**症状**: `MemoryError` / プロセスキル
**原因**: 統合モデル学習時に全銘柄データを結合するため
**対処**:
1. DuckDB 接続の memory_limit を確認（デフォルト 2GB）
2. 監視銘柄数を削減して再実行
3. Docker 環境ではコンテナのメモリ制限を増加

## import エラー
**症状**: `ModuleNotFoundError`
**対処**:
1. 実行は `python/` ディレクトリから行う
2. import パスは `python/` からの絶対パスで統一（例: `from src.utils.db import ...`）
3. 各ディレクトリに `__init__.py` が配置されているか確認

## pip-audit CI FAIL（依存脆弱性スキャン）
**症状**: `Dependency Vulnerability Scan (pip-audit)` が失敗する
**対処**:
```bash
cd C:\src\StockFixer\python
pip install pip-audit
pip-audit -r requirements.txt
pip-audit -r requirements-dev.txt
```
報告されたパッケージを修正バージョン以上に更新してから push する。

## bandit CI FAIL（SAST スキャン）
**症状**: `SAST Scan (bandit)` が失敗する（HIGH severity 検出）
**対処**:
```bash
cd C:\src\StockFixer\python
pip install bandit
bandit -r src/ --exclude src/_deprecated -ll
```
HIGH 箇所を修正する。誤検知の場合は `# nosec B602 - 理由` でコメント抑制する。
