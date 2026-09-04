# 設計: レジームレバレッジ戦略ペーパートレードボット

日付: 2026-09-03
関連: `trading-strategy/STRATEGY.md` 7章「強気相場・買い持ち型戦略」

## 1. 背景・目的

`C:\src\trading-strategy` で信用取引戦略（STRATEGY.md）を管理している。2〜6章
（ブレイクアウト型ロング）・5章（空売り）・7章（強気相場・レバレッジ買い持ち）の
3サブ戦略を、既存のスタンドアロンPythonスクリプト（`backtest.py` 系列）で
最新データ（直近5年〜20年）で再検証した結果:

- 2〜6章: 平均+0.47R・勝率56.6%（正のエッジは維持。ただし強気相場では単純買い持ちに劣後）
- 5章: 平均-0.10R・勝率49.4%（**負けエッジ、成長株ユニバースには不向き**）
- 7章: マージンコール現実モデルで、単純な信用買い持ちが20年で16/20銘柄破綻する一方、
  レジームフィルター運用は生存。SPYはレバレッジ2.5〜3倍で現物買い持ちに匹敵

この結果を受け、**7章をSPY単独・レバレッジ2.0倍・円建て（為替リスク込み）**で
StockFixerのペーパートレードボットとして先行実装する。5章は不採用、2〜6章は
将来の別フェーズ。

## 2. 参照実装

- `trading-strategy/backtest/backtest_regime_leverage.py` — レバレッジ・複利・週次マージンコール判定のロジック本体
- `trading-strategy/backtest/backtest_regime_fx.py` — 円建て換算・為替込み検証（レジーム判定はドル建てSPYの200日線を使う、7.4節の記述どおり。円建て判定は検証済みで不採用）

## 3. 実装方針: 自己完結モジュール（allocation_strategy パターン踏襲）

`src/trading/allocation_strategy/`（TQQQ/SHY配分戦略ボット、PR #679/680）と同じ設計:

- 新規BC `src/trading/regime_leverage_strategy/` を新設。既存の `PaperBroker`・`allocation_strategy` には一切手を入れない
- 状態は追記専用ログテーブル `regime_leverage_log`（`allocation_rebalance_log` と同型）
- レバレッジ分の「借入」は仮想的な負債として記録するのみ（実際の信用取引APIは使わない）
- 価格取得は `MarketDataPort` をDI注入（BCはinfrastructure/他BCを直接importできないレイヤー規約のため、具象アダプタの生成と注入はorchestration層の責務——`allocation_strategy/service.py` と同じパターン）
- ドル円レートは `MarketDataPort.get_forex_data("JPY=X", start, end)` で取得（要実装確認: `YFinanceMarketDataAdapter.get_forex_data` の対応状況を最初のタスクで検証する）

## 4. テーブルスキーマ

```sql
CREATE TABLE IF NOT EXISTS regime_leverage_log (
    id                    BIGSERIAL PRIMARY KEY,
    executed_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action                VARCHAR NOT NULL,   -- 'entry' | 'exit' | 'noop'
    reason                VARCHAR,            -- 'regime_entry' | 'regime_flip' | 'initial_stop' | 'margin_call' | 'weekly_noop' | 'daily_noop'
    spy_price_usd         DOUBLE PRECISION,
    usdjpy_rate           DOUBLE PRECISION,
    shares                DOUBLE PRECISION NOT NULL,  -- 現在の保有株数（0なら未保有）
    entry_date            TIMESTAMP,                  -- 保有中のみ意味を持つ（金利の日数計算用）
    entry_price_jpy       DOUBLE PRECISION,           -- 保有中のみ意味を持つ
    entry_commission_jpy  DOUBLE PRECISION,           -- エントリー時に確定する手数料（固定値）
    equity_at_entry_jpy   DOUBLE PRECISION,           -- 保有開始時点の自己資金（複利計算の基準）
    stop_price_jpy        DOUBLE PRECISION,           -- entry - 3.0*ATR（円建て、エントリー時に固定）
    equity_now_jpy        DOUBLE PRECISION NOT NULL,  -- このレコード時点の評価額（未保有なら現金相当）
    maintenance_ratio     DOUBLE PRECISION            -- 保有中のみ意味を持つ実測維持率
);
```

`allocation_rebalance_log` と異なり、7章は「週次のエントリー/エグジット判定」と
「日次のマージンコール判定」の2種類のジョブが同じ状態を読み書きするため、
`action`/`reason` で発生源を区別できるようにする。

**評価額の再計算方式**: バックテスト（`backtest_regime_leverage.py`）は保有期間中の
含み損益・金利をポジション辞書に累積して持ち回るが、ペーパートレードでは
`entry_date`・`entry_price_jpy`・`equity_at_entry_jpy`・`entry_commission_jpy`
の4値だけを保持し、評価が必要になるたびに以下を都度再計算する（累積値を毎回
更新して持ち回らない、バグりにくい方式）:

```
unrealized_pnl_jpy   = (current_price_jpy - entry_price_jpy) * shares
days_held            = (today - entry_date).days
interest_accrued_jpy = entry_price_jpy * shares * INTEREST_ANNUAL["JPY"] / 365 * days_held
equity_now_jpy       = equity_at_entry_jpy + unrealized_pnl_jpy - interest_accrued_jpy - entry_commission_jpy
maintenance_ratio    = equity_now_jpy / (current_price_jpy * shares)
```

## 5. ジョブ設計

バックテスト（`run_levered_regime`）は、マージンコール・初期損切りの両方を
**週中の安値**で判定し、レジーム転換のみ週足終値で判定している。これをそのまま
週次バッチに落とすと「週の半ばの急落を金曜まで検知できない」——まさに最初に
日次ジョブを作る根拠にした問題が、初期損切りにも同じ形で残ってしまう。
したがって**初期損切りとマージンコールは日次ジョブが安値ベースで判定し、
週次ジョブはレジーム転換の判定と新規エントリーだけを担当**する設計にする。

### 5.1 週次ジョブ（毎週金曜、市場引け後）

`run_regime_leverage_weekly_check(market_data_port)`

1. 直近の `regime_leverage_log` から現在の状態（保有中か否か、保有中なら entry_price/stop/shares）を復元
2. SPYの週足終値・200日線（日足からrollingで算出）を取得
3. **未保有の場合**:
   - 週足終値 > 200日線（レジーム上昇）なら新規エントリー
     - 建玉USD時価 = 直前のequity_after_jpy（円）÷ ドル円レート × 2.0（レバレッジ）
     - shares = floor(建玉USD時価 ÷ 週足終値)
     - stop_price = entry_price - 3.0 × ATR(14)（円建てに換算して保存)
   - 上昇でなければ `noop`
4. **保有中の場合**:
   - 週足終値 ≤ 200日線 → `regime_flip` でエグジット（全量売却）
   - 上回っていれば `noop`（保有継続。初期損切り判定はこのジョブでは行わない——5.2参照）

### 5.2 日次ジョブ（毎営業日、市場引け後）

`run_regime_leverage_daily_margin_check(market_data_port)`

1. 直近の状態を復元。**未保有なら何もしない**
2. 当日のSPY安値・ドル円レートを取得し、円建て評価額を計算
3. 判定は以下の優先順位で行う（バックテストの `run_levered_regime` と同じ順序）:
   a. 維持率 = (equity_at_entry + (当日安値換算 − entry_price) × shares) ÷ (当日安値換算 × shares)。
      これが 0.20（`MARGIN_MAINTENANCE["JPY"]`）を下回れば即時強制決済（`margin_call`）
   b. 上記に該当せず、当日安値換算が stop_price（エントリー時に固定した entry − 3.0×ATR）を
      下回れば `initial_stop` で決済
   c. どちらでもなければ `daily_noop`（`maintenance_ratio` は監視のため毎回記録する）

**処理順序の注意**: 金曜は市場引け後に日次ジョブ→週次ジョブの順で実行する
（日次ジョブが先に決済していれば、週次ジョブは「未保有」から判定を始める）。

## 6. モジュール構成

```
src/trading/regime_leverage_strategy/
  __init__.py
  types.py        # RegimeLeverageSnapshot データクラス
  repository.py   # get_latest_snapshot / list_snapshots / insert_snapshot
  service.py      # run_regime_leverage_weekly_check / run_regime_leverage_daily_margin_check
```

`allocation_strategy` との違い: サービス関数が2つ（週次・日次）に分かれる点、
`equity.py` 的な「エクイティ曲線の再構成」は今回のスコープでは作らない
（ユーザー合意により月次損益グラフへの統合は見送り、後日追加）。

## 7. スケジューラ登録

`run_scheduler.py` の `SCHEDULE_CONFIG` に2エントリを追加:

```python
"regime_leverage_weekly": {"day_of_week": "fri", "hour": 6, ...},   # 週次、6章の配分戦略と同じ06:00枠を踏襲
"regime_leverage_daily_margin": {"hour": 6, ...},                    # 日次
```

（具体的な時刻・cron式は既存の `allocation_rebalance` ジョブの登録パターンに合わせる）

## 8. テスト方針（TDD）

- `repository.py`: 既存 `test_allocation_repository.py` と同型（モックDB接続）
- `service.py`:
  - 週次: レジーム上昇/下降・保有あり/なし・初期損切り到達、の組み合わせを純粋ロジックとしてテスト（MarketDataPortはモック）
  - 日次: 維持率が閾値を上回る/下回るケースをテスト
- 実データでの動作確認: `advisor` 指摘で毎回発覚している「実データで一度動かす」を今回も徹底する（モックだけで満足しない）

## 9. 既知の限界（バックテストからの継承）

- カタリスト・イベント前クローズ判定は対象外（指数連動のためそもそも7.1節で個別株のカタリスト条件は適用外）
- 手数料・金利モデルは `trading-strategy/backtest.py` の `INTEREST_ANNUAL["JPY"]=0.030`・`COMMISSION_PCT` をそのまま踏襲
- 為替レート取得元（`YFinanceMarketDataAdapter.get_forex_data` の実装状況）は未確認。実装タスクの最初に検証すること
