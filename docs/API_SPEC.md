# API 仕様書

StockFixer の Flask REST API エンドポイントおよび Discord スラッシュコマンドの仕様をまとめたドキュメント。

> **依存関係**: `/health` エンドポイントは NF-301 完了後に利用可能。  
> **後続**: R-304（外部提供 API 設計）の前提資料として使用する。

---

## 公開範囲の区別

| 区分 | 説明 |
|------|------|
| **外部公開可能** | Docker HEALTHCHECK・監視システム・外部モニタリングから呼び出し可能 |
| **内部限定** | コンテナ内部またはローカルネットワーク内部からのみアクセスを想定 |

---

## Flask REST API

ポート: `5100`（環境変数 `HEALTH_PORT` で変更可能）  
実装: `python/src/api/health.py`, `python/src/api/metrics.py`

### エンドポイント一覧

| エンドポイント | メソッド | 公開範囲 | 説明 |
|----------------|----------|----------|------|
| `/health` | GET | **外部公開可能** | ヘルスチェック（DB・スケジューラ・予測） |
| `/metrics` | GET | 内部限定 | Prometheus 形式メトリクス |

---

### GET /health

システム全体の稼働状態を返す。Docker `HEALTHCHECK` から呼び出される。

#### リクエスト

```
GET /health HTTP/1.1
Host: localhost:5100
```

パラメータ・リクエストボディなし。

#### レスポンス

**200 OK** — DB が正常な場合

```json
{
  "status": "ok",
  "db": "ok",
  "scheduler_last_runs": {
    "daily_pipeline": "2026-05-10T03:00:00+00:00",
    "weekly_model_training": "2026-05-05T02:00:00+00:00"
  },
  "last_prediction_at": "2026-05-10T03:05:23.412000+00:00",
  "checked_at": "2026-05-10T04:00:00.000000+00:00"
}
```

**503 Service Unavailable** — DB 接続失敗時

```json
{
  "status": "degraded",
  "db": "error: unable to open database file",
  "scheduler_last_runs": {},
  "last_prediction_at": null,
  "checked_at": "2026-05-10T04:00:00.000000+00:00"
}
```

#### レスポンスフィールド

| フィールド | 型 | 説明 |
|------------|----|------|
| `status` | string | `"ok"` または `"degraded"` |
| `db` | string | `"ok"` または `"error: <メッセージ>"` |
| `scheduler_last_runs` | object | `{job_id: ISO8601タイムスタンプ}` の辞書 |
| `last_prediction_at` | string \| null | 最新予測の実行時刻（UTC ISO8601）、未実行時は `null` |
| `checked_at` | string | このレスポンス生成時刻（UTC ISO8601） |

#### ヘルスチェック内容

1. DuckDB への `SELECT 1` 接続確認
2. `scheduler_queue_state.json` からスケジューラ最終実行時刻の読み込み
3. `prediction_results` テーブルの `MAX(predicted_at)` 取得

---

### GET /metrics

Prometheus テキスト形式でアプリケーションメトリクスを返す。

#### リクエスト

```
GET /metrics HTTP/1.1
Host: localhost:5100
```

#### レスポンス

**200 OK** — Content-Type: `text/plain; version=0.0.4; charset=utf-8`

```
# HELP pipeline_duration_seconds Pipeline execution duration in seconds
# TYPE pipeline_duration_seconds histogram
pipeline_duration_seconds_bucket{pipeline="daily_pipeline",le="1.0"} 0.0
pipeline_duration_seconds_bucket{pipeline="daily_pipeline",le="5.0"} 0.0
...
# HELP pipeline_runs_total Total number of pipeline runs
# TYPE pipeline_runs_total counter
pipeline_runs_total{pipeline="daily_pipeline",status="success"} 42.0
pipeline_runs_total{pipeline="daily_pipeline",status="fail"} 1.0
# HELP duckdb_query_duration_seconds DuckDB query duration in seconds
# TYPE duckdb_query_duration_seconds histogram
...
```

#### 公開メトリクス

| メトリクス名 | 型 | ラベル | 説明 |
|---|---|---|---|
| `pipeline_duration_seconds` | Histogram | `pipeline` | パイプライン実行時間（秒） |
| `pipeline_runs_total` | Counter | `pipeline`, `status` | パイプライン実行回数（`status`: `success`/`fail`） |
| `duckdb_query_duration_seconds` | Histogram | なし | DuckDB クエリ実行時間（秒） |

**内部限定の理由**: メトリクスは内部監視（Prometheus スクレイピング）専用。外部公開するとシステム内部情報が漏洩する可能性がある。

---

## Discord コマンド

実装: `python/src/reporting/discord/discord_bot.py`  
コマンドはすべてメッセージテキストによるプレフィックスコマンド（スラッシュコマンド形式の文字列）。  
**公開範囲**: すべて内部限定（認証済み Discord サーバー内のみ）。

### コマンド一覧

| コマンド | パラメータ | 説明 |
|----------|------------|------|
| `/forecast` | なし | 最新予測結果の Top10 / Worst10 一覧 |
| `/WatchNext` | なし | 監視リスト銘柄の予測サマリ |
| `/signal <symbol> [market] [--explain]` | symbol (必須), market (任意), --explain (任意) | 単一銘柄シグナル |
| `/status` | なし | スケジューラ稼働状態 |
| `/monthlyreport [YYYY-MM]` | 対象月 (任意) | 月次 KPI レポート |

---

### /forecast

最新の予測結果から市場別 Top10（上昇期待）・Worst10（下落期待）を表示する。

#### パラメータ

なし

#### レスポンス例

```
【US市場】差異割合上位10銘柄
symbol    現在値      予想終値     予想変化率
NVDA      875.234    884.123     +1.015%
AAPL      189.456    191.234     +0.939%
...

【JP市場】差異割合ワースト10銘柄
symbol    現在値      予想終値     予想変化率
7203      2500.000   2462.500    -1.500%
...
```

#### エラーレスポンス

- `予測結果が見つかりませんでした。` — DB に予測データなし

---

### /WatchNext

監視リスト（`monitor_list.csv`）に登録された銘柄の予測サマリを表示する。

#### パラメータ

なし

#### レスポンス例

```
【監視対象銘柄】
symbol    現在値      予想終値     予想変化率
AAPL      189.456    191.234     +0.939%
MSFT      420.123    423.567     +0.819%
...
```

---

### /signal

単一銘柄のシグナル（Buy / Hold / Sell）と予測価格を表示する。

#### パラメータ

| パラメータ | 必須 | デフォルト | 説明 |
|------------|------|-----------|------|
| `<symbol>` | 必須 | — | 銘柄コード（大文字変換される） |
| `[market]` | 任意 | `"us"` | 市場コード（`us`, `jp` 等） |
| `[--explain]` | 任意 | なし | SHAP 寄与度 Top5 を表示するフラグ |

#### コマンド例

```
/signal AAPL
/signal AAPL us
/signal 7203 jp
/signal NVDA us --explain
```

#### レスポンス例（--explain なし）

```
=== US / AAPL ===
現在値    : 189.456
予想終値  : 191.234
予想変化率: +0.939%
シグナル  : ⬆️ Buy
使用モデル: 3モデルの平均
```

#### レスポンス例（--explain あり）

```
=== US / AAPL ===
現在値    : 189.456
予想終値  : 191.234
予想変化率: +0.939%
シグナル  : ⬆️ Buy
使用モデル: 3モデルの平均

[SHAP 寄与度 Top5 - 方向:上昇]
  ▲ rsi_14              +0.00312
  ▲ close_lag_1         +0.00287
  ▼ volume_ma_5         -0.00134
  ▲ macd_signal         +0.00098
  ▼ bb_upper            -0.00076
```

#### シグナル判定基準

| 予想変化率 | シグナル |
|-----------|----------|
| > +0.5% | ⬆️ Buy |
| -0.5% 〜 +0.5% | ⏺️ Hold |
| < -0.5% | ⬇️ Sell |

#### エラーレスポンス

- `モデルが見つかりませんでした: us/AAPL` — 指定銘柄のモデル未学習

---

### /status

APScheduler の日次・週次ジョブの最終実行状態を表示する。

#### パラメータ

なし

#### レスポンス例

```
=== スケジューラ状態 ===
日次 (daily_pipeline)
  最終実行: 2026-05-10 12:00:00 JST  [状態: success]
週次 (weekly_model_training)
  最終実行: 2026-05-05 02:00:00 JST  [状態: success]
```

#### エラーレスポンス

- `スケジューラ状態ファイルが見つかりません。` — `scheduler_queue_state.json` が存在しない

---

### /monthlyreport

指定月（デフォルト: 当月）の月次 KPI レポートを表示する。

#### パラメータ

| パラメータ | 必須 | デフォルト | 説明 |
|------------|------|-----------|------|
| `[YYYY-MM]` | 任意 | 当月 | 対象年月（例: `2026-04`） |

#### コマンド例

```
/monthlyreport
/monthlyreport 2026-04
```

#### レスポンス例

```
=== 月次KPIレポート 2026-04 ===

【最重要KPI】
  Net Return   : +2.34%
  Max Drawdown : 8.45%  (目標: 15%以下  → 達成)
  Sharpe Ratio : 1.2345  (目標: 1.0以上  → 達成)

【補助KPI】
  Hit Rate     : 54.32%  (直近30日)
  Avg Slippage : 0.12%   (直近30日)

【メタ情報】
  対象銘柄数   : 25
  WFスナップ   : wf_snapshot_2026-04.json
  生成日時     : 2026-05-01T02:00:00
```

#### エラーレスポンス

- `月次レポートの取得に失敗しました: <エラー内容>` — データ取得・処理エラー時

---

## メッセージフォーマット共通仕様

実装: `python/src/reporting/discord/discord_formatters.py`, `python/src/reporting/discord/discord_text.py`

### 列名マッピング

| DB カラム名 | Discord 表示名 |
|-------------|----------------|
| `symbol` | `シンボル` |
| `current_price` | `現在値` |
| `avg_pred_price` | `予想終値` |
| `diff_ratio` | `予想変化率` |

### 数値フォーマット

| データ種別 | フォーマット |
|------------|------------|
| 価格 | 小数点以下3桁で切り捨て |
| 変化率 | `+X.XXg%` / `-X.XXg%` 形式 |

### メッセージ分割

| 設定 | 上限 |
|------|------|
| 標準メッセージ | 1,900 文字 |
| ワイドメッセージ | 3,800 文字 |

1 メッセージが上限を超える場合、行単位で複数メッセージに分割して送信する。
