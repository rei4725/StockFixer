APScheduler ベースのジョブスケジューラーを適切に運用する。

## 実行コマンド

```bash
cd python
py run_scheduler.py              # 通常起動（BlockingScheduler）
py run_scheduler.py --with-bot   # Discord Bot と同時起動（BackgroundScheduler）

# パイプライン即時実行（テスト用）
py run_scheduler.py --run-now daily
py run_scheduler.py --run-now weekly
py run_scheduler.py --run-now optimization
```

## スケジュール定義
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
| `weekly_watchlist_refresh` | 毎週金曜 01:00 JST | ウォッチリスト自動更新（S&P500/日経225 差分同期） |

## 共通設定
- `misfire_grace_time=3600` — 1時間以内の遅延なら実行
- `coalesce=True` — 複数回分溜まっても1回だけ実行

## 運用方針
平日夜〜休日は負荷の高い処理を入れない。週次処理（学習・最適化）は平日深夜（01:00〜）に曜日分散、土日は完全フリー。

週次自動デプロイ（`weekly_redeploy.ps1`）は毎週月曜 02:00 に git pull → テスト → Docker リビルドを自動実行。UnitTest 失敗時はデプロイ中断（`exit 1`）。

## Key Functions
- `run_daily_pipeline()` — 日次パイプライン（データ→予測→通知）
- `run_weekly_training()` — 週次統合モデル再学習
- `run_weekly_optimization()` — 週次全銘柄バックテスト最適化
