# 障害対応フロー（INCIDENT_RESPONSE）

StockFixer における障害レベルの定義・初動対応・エスカレーション・ポストモーテムテンプレートをまとめる。

---

## 障害レベル定義

| レベル | 名称 | 定義 | 検知目安 |
|---|---|---|---|
| **P1** | 取引停止 | 自動発注・約定処理が完全に停止し、注文の送受信ができない状態 | Discord 通知なし + ジョブ完了ログが 10 分以上途絶 |
| **P2** | 予測異常 | 予測パイプラインが失敗・異常値を返し、Top10/Worst10 が生成されない状態 | Discord の予測通知なし、または全銘柄が同一スコア |
| **P3** | 通知遅延・軽微障害 | Discord 通知の遅延・欠落、個別銘柄の取得失敗など部分的な機能劣化 | スケジュール時刻から 30 分超過で通知なし |

---

## P1: 取引停止

### 症状
- `daily_auto_order` / `daily_settle_orders` ジョブが完了しない
- Discord の発注通知・約定通知が来ない
- `Logs/stockfixer_error.log` に致命的なエラーが記録されている

### 初動対応（5 分以内）

1. ログ確認
   ```powershell
   docker compose logs --tail 100
   Get-Content Logs\stockfixer_error.log -Tail 50
   ```

2. コンテナ状態確認
   ```powershell
   docker ps --filter name=stockfixer
   ```

3. コンテナが停止している場合は再起動
   ```powershell
   docker compose restart
   ```

4. 再起動後も回復しない場合は手動停止（損失拡大防止）
   ```powershell
   docker compose stop
   ```

### エスカレーション
- **15 分以内**: Discord `#alerts` チャンネルに手動で「発注停止中」を通知する
- **30 分以内**: 原因が特定できない場合は `docker compose stop` でシステム全停止し、手動監視に切り替える

### 復旧確認
```powershell
# ペーパートレード状態確認
docker exec stockfixer python run_predict.py --mode watchlist

# 発注ジョブの手動実行テスト
docker exec stockfixer python run_scheduler.py --run-now paper
```

---

## P2: 予測異常

### 症状
- `daily_pipeline` が失敗またはタイムアウト（通常 20〜50 分）
- Discord の Top10/Worst10 通知が来ない
- 予測値がすべて同じ値、または極端な値（±99% など）

### 初動対応（15 分以内）

1. パイプラインログ確認
   ```powershell
   Get-Content Logs\stockfixer.log -Tail 100
   ```

2. DuckDB 接続確認（ロック障害の疑い）
   ```powershell
   Test-Path python\data\stockfixer.duckdb.lock
   ```
   ロックファイルが残存している場合 → 「DuckDB ロック競合」の対処を適用

3. yfinance 疎通確認
   ```powershell
   docker exec stockfixer python -c "import yfinance as yf; print(yf.Ticker('AAPL').fast_info)"
   ```
   タイムアウト・エラーの場合 → yfinance 側の一時的な障害と判断し、次回実行を待つ

4. 単一銘柄で再実行し、個別銘柄の問題かパイプライン全体の問題かを切り分け
   ```powershell
   docker exec stockfixer python run_data_creation.py --market us --symbol AAPL
   docker exec stockfixer python run_predict.py --mode single --market us --symbol AAPL
   ```

### エスカレーション
- **30 分以内**: 全銘柄で予測不能な場合は Discord に手動通知し、当日の自動取引を停止する
- **翌日まで**: モデルファイル (`python/models/`) が破損していないか確認する

### 復旧確認
```powershell
# 日次パイプラインを即時実行
docker exec stockfixer python run_scheduler.py --run-now daily
```

---

## P3: 通知遅延・軽微障害

### 症状
- Discord 通知が遅延（スケジュール時刻から 30 分超）
- 一部銘柄のデータ取得失敗（全体は動作している）
- ログにWARNING が多数出力されている

### 初動対応（当日中）

1. スケジューラーのジョブ状態確認
   ```powershell
   Get-Content Logs\stockfixer.log -Tail 200
   ```

2. Discord Bot の接続状態確認（Bot がオフライン表示の場合）
   ```powershell
   docker compose logs stockfixer | Select-String "discord"
   ```

3. 失敗した銘柄の個別再取得
   ```powershell
   docker exec stockfixer python run_data_creation.py --market jp --symbol <SYMBOL>
   ```

### エスカレーション
- **翌営業日まで**: 障害内容と影響範囲をログから確認し、ポストモーテムを作成する（軽微な場合は省略可）

---

## 既知の障害パターン

### DuckDB ロック競合

**症状**
```
IOException: Could not set lock on file 'python/data/stockfixer.duckdb'
RuntimeError: DuckDB書き込みロック取得タイムアウト
```

**原因**  
複数の Python プロセスが同時に DuckDB への書き込み接続を取得しようとしている。スケジューラー稼働中に手動で `run_*.py` を実行した場合に発生しやすい。

**復旧手順**
1. 手動実行中のスクリプトを中断する（Ctrl+C）
2. スケジューラーのジョブ完了を待つ（`docker compose logs -f` で確認）
3. ロックファイルが残存している場合は削除する
   ```powershell
   Remove-Item python\data\stockfixer.duckdb.lock -ErrorAction SilentlyContinue
   ```
4. 操作を再実行する

詳細は `docs/OPERATIONS.md` の「DuckDB ロック競合エラー」および `docs/LOCK_DETECTION_GUIDE.md` を参照。

---

### yfinance データ取得障害

**症状**
```
YFRateLimitError: Too many requests
ConnectionError: HTTPSConnectionPool... Max retries exceeded
```

**原因**  
Yahoo Finance API の一時的な障害またはレート制限。

**復旧手順**
1. 30 分〜1 時間待機後に再実行する
2. 単一銘柄でのテスト実行で疎通を確認する
   ```powershell
   docker exec stockfixer python -c "import yfinance as yf; print(yf.Ticker('AAPL').fast_info)"
   ```
3. 疎通が回復したら日次パイプラインを即時実行する
   ```powershell
   docker exec stockfixer python run_scheduler.py --run-now daily
   ```

---

### モデルファイル不整合

**症状**
```
FileNotFoundError: No such file or directory: 'python/models/...'
joblib.externals.loky.process_executor.TerminatedWorkerError
```

**原因**  
モデルファイルが存在しない、または破損している。週次モデル学習が途中で失敗した場合に発生することがある。

**復旧手順**
1. モデルファイルの存在確認
   ```powershell
   Get-ChildItem python\models -Recurse -Filter "*.joblib"
   ```
2. 不足・破損している銘柄のモデルを再学習する
   ```powershell
   docker exec stockfixer python run_model_creation.py --market jp --symbol <SYMBOL>
   ```
3. 統合モデルが必要な場合は再学習する
   ```powershell
   docker exec stockfixer python run_unified_model_training.py
   ```

---

### コンテナ起動失敗

**症状**  
`docker compose up -d` 後にコンテナが即座に終了する。

**原因**  
`.env` ファイルの欠落・誤記、または依存ライブラリのインストール失敗。

**復旧手順**
1. ログで原因を確認する
   ```powershell
   docker compose logs
   ```
2. `.env` ファイルの存在と必須キーを確認する
   ```powershell
   Test-Path python\.env
   ```
3. イメージをキャッシュなしで再ビルドする
   ```powershell
   docker compose build --no-cache
   docker compose up -d
   ```

---

### 日次損失ガード誤発動

**症状**  
発注ジョブが実行されるが、注文が送信されない。ログに「日次損失上限に達しました」と記録されている。

**原因**  
`MAX_DAILY_LOSS_RATE` で設定した閾値を超えた、または設定値が意図せず低くなっている。

**復旧手順**
1. 損失額と設定値を確認する
   ```powershell
   Get-Content Logs\stockfixer.log | Select-String "損失"
   ```
2. 誤発動と判断した場合は一時的に無効化して再実行する
   ```powershell
   docker exec -e DISABLE_DAILY_LOSS_GUARD=1 stockfixer python run_scheduler.py --run-now paper
   ```
3. 再実行後は環境変数を削除し、恒久的な無効化を残さない

---

## ポストモーテムテンプレート

障害収束後、P1・P2 については必ずポストモーテムを作成する。P3 は任意。

```markdown
## ポストモーテム: <タイトル>

**作成日**: YYYY-MM-DD
**作成者**: <担当者>
**レベル**: P1 / P2 / P3

---

### 発生時刻・収束時刻

| 項目 | 時刻（JST） |
|---|---|
| 検知時刻 | YYYY-MM-DD HH:MM |
| 収束時刻 | YYYY-MM-DD HH:MM |
| 総障害時間 | X 時間 Y 分 |

---

### 影響範囲

- 影響を受けたジョブ:
- 影響を受けた銘柄・市場:
- 未送信の Discord 通知:
- 未発注の注文件数（ペーパートレード）:

---

### タイムライン

| 時刻（JST） | 出来事 |
|---|---|
| HH:MM | 障害発生 |
| HH:MM | 検知 |
| HH:MM | 初動対応開始 |
| HH:MM | 原因特定 |
| HH:MM | 復旧措置実施 |
| HH:MM | 収束確認 |

---

### 根本原因

（技術的な根本原因を 1〜3 段落で記述する）

---

### 再発防止策

| 対策 | 担当 | 期限 | 関連チケット |
|---|---|---|---|
| | | | |

---

### 教訓・補足

（今回の障害から得られた教訓、注意事項など）
```

---

## 関連ドキュメント

- [OPERATIONS.md](OPERATIONS.md) — 日常運用手順・DuckDB ロック対処
- [RUNBOOK_DEPLOY.md](RUNBOOK_DEPLOY.md) — デプロイ手順・単一プロセス制約
- [LOCK_DETECTION_GUIDE.md](LOCK_DETECTION_GUIDE.md) — DuckDB ロック問題の自動検出

---

*Last updated: 2026-05-10*
