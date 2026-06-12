---
name: scheduler-ops
description: "スケジューラーの起動・管理・即時実行を行う。スケジューラー・定期実行・日次パイプライン・週次学習の話題では必ずこのスキルを使用する。cron・daily・weeklyジョブの設定や即時テスト実行が絡む場面でも使用する。"
compatibility: "Python 3.10+, APScheduler。python/ ディレクトリで実行。"
---

## Goal
APSchedulerベースのジョブスケジューラーを適切に運用する。

## Procedure

### スケジューラー起動
```bash
cd python
# 通常起動（BlockingScheduler）
py run_scheduler.py

# Discord Bot と同時起動（BackgroundScheduler）
py run_scheduler.py --with-bot
```

### パイプライン即時実行（テスト用）
```bash
py run_scheduler.py --run-now daily
py run_scheduler.py --run-now weekly
py run_scheduler.py --run-now optimization
```

### スケジュール定義
| ジョブID | スケジュール | 処理内容 |
|----------|-------------|----------|
| `daily_pipeline` | 月〜金 07:30 JST | データバッチ取得 → 統合モデル予測 → Discord通知 |
| `daily_auto_order` | 月〜金 08:50 JST | ペーパートレード注文発注 |
| `daily_settle_orders` | 月〜金 09:05 JST | pending 注文の約定処理 |
| `daily_paper_trade_report` | 月〜金 15:30 JST | ペーパートレード損益レポート Discord 送信 |
| `daily_drift_check` | 月〜金 16:00 JST | ドリフト監視と閾値超過銘柄の再学習 |
| `weekly_model_training` | 毎週火曜 01:00 JST | XGBoost + LightGBM 統合モデル再学習 |
| `weekly_optimization` | 毎週水曜 01:00 JST | 全銘柄バックテスト最適化 → `config/optimal_params.json` 更新 |
| `weekly_walk_forward_report` | 毎週木曜 01:00 JST | Walk-Forward 比較レポート生成 |
| `weekly_report` | 毎週木曜 02:00 JST | パフォーマンスレポート Discord 送信 |
| `weekly_watchlist_refresh` | 毎週金曜 01:00 JST | ウォッチリスト自動更新（S&P500 / 日経225 差分同期） |

### スケジュール運用方針
> メインPCで稼動するため、**平日夜〜休日は負荷の高い処理を入れない**。重い週次処理（学習・最適化）は平日深夜（01:00〜）に曜日分散、土日は完全フリー。
- **週次再デプロイ（`weekly_redeploy.ps1`）**: 毎週月曜 02:00 にgit pull → テスト → Dockerリビルドを自動実行

### 共通設定
- `misfire_grace_time=3600` — 1時間以内の遅延なら実行
- `coalesce=True` — 複数回分溜まっても1回だけ実行

### 日次パイプラインの処理フロー
1. `run_data_batch()` — ウォッチリスト全銘柄のデータ更新
2. `predict_all_unified()` — 統合モデルで全銘柄予測
3. Discord Webhook で Top10/Worst10 通知

### 週次学習の処理フロー
1. XGBoost 統合モデル再学習
2. LightGBM 統合モデル再学習
3. 保存先: `python/models/unified/`

## Key Functions
- `run_daily_pipeline()` — 日次パイプライン（データ→予測→通知）
- `run_weekly_training()` — 週次統合モデル再学習
- `run_weekly_optimization()` — 週次全銘柄バックテスト最適化（`config/optimal_params.json` 更新）

## References
- [scheduler.py](../../../python/src/orchestration/scheduler.py)
- [run_scheduler.py](../../../python/run_scheduler.py)

## 運用メモ（デプロイ連携）
- Windows タスク `StockFixer Weekly Redeploy`（`weekly_redeploy.ps1`）は、再デプロイ前に `python -m pytest tests/unit -v` を実行する
- UnitTest が失敗した場合、再デプロイは中断される（`exit 1`）
