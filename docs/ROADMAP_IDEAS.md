# StockFixer 次期機能アイデア・ロードマップ

> 作成日: 2026-03-15  
> 目的: 予測精度の向上 × マネタイズ実現に向けた実装候補の整理

---

## 現状サマリー

| カテゴリ | 現在の実装 |
|---------|-----------|
| データ取得 | yfinance（OHLCV・日足） |
| 特徴量 | テクニカル指標（RSI/MACD/EMA/ATR/BB/Stoch）+ ラグ特徴量 |
| モデル | XGBoost / LightGBM（翌日終値変化率を回帰予測） |
| シグナル | Buy/Sell/Hold（予測閾値 ±0.5% + RSI補強） |
| バックテスト | Walk-Forward + パラメータ最適化 |
| 通知 | Discord Bot（Top10/Worst10） |
| 自動売買 | SBI Webスクレイピング（Selenium実装途中）※公式APIへ移行予定 |

---

## Part 1｜精度向上アイデア

### 🥇 Priority A — 効果大・実装コスト中

---

#### A-1. 外部マクロ指標の特徴量化
**概要**: 個別株の予測に、市場全体の状況を示すマクロ指標を追加する。

| 追加する指標 | 取得元 | 用途 |
|------------|--------|------|
| USD/JPY 為替レート | yfinance (`JPY=X`) | 輸出株への影響補正 |
| VIX（恐怖指数） | yfinance (`^VIX`) | リスクオン/オフ判定 |
| 米10年金利 | yfinance (`^TNX`) | グロース株バリュエーション |
| セクターETF（XLK, XLF等） | yfinance | セクターローテーション検出 |
| 日経225 / SP500 | yfinance | 指数連動性の特徴量 |

**実装ポイント**:
- `data_loader.py` に `load_macro_data()` を追加
- `technical_analysis.py` でメイン株価DFにjoin
- 既存の `create_basic_lag_features()` にそのまま流せる

---

#### A-2. 市場レジーム検出（Market Regime Detection）
**概要**: 相場がトレンド相場・レンジ相場・高ボラティリティのどの状態かを検出し、モデルの予測に重みづけまたはフィルタリングを行う。

```
相場レジーム = HMM(隠れマルコフモデル) or ルールベース（ADX/ボリンジャー幅）
          ↓
レジームを one-hot 特徴量 or モデル切り替えのゲートとして利用
```

**効果**: トレンド相場で正確なモデルがレンジ相場で誤シグナルを出すのを防ぐ。
**実装**: `features/market_regime.py` を新規作成。`hmmlearn` または ADX閾値による3値分類。

---

#### A-3. マルチタイムフレーム確認シグナル
**概要**: 日足シグナルを週足・月足トレンドと照合し、方向が一致するときのみシグナルを採用する。

```
週足トレンド UP + 日足 Buy シグナル → 採用
週足トレンド DOWN + 日足 Buy シグナル → 棄却（Hold へ格下げ）
```

**実装**:
- `data_loader.py` に `interval="1wk"` / `"1mo"` 取得を追加
- `signal_generator.py` に `confirm_with_higher_tf()` メソッドを追加
- バックテスト結果でシグナル数と精度のトレードオフを検証

---

#### A-4. アンサンブル強化（スタッキング）
**概要**: XGBoost と LightGBM の予測値の単純平均（現状）を、メタモデル（LinearRegression or Ridge）でスタッキングする。

```
XGBoost予測
LightGBM予測  } → メタモデル(Ridge) → 最終予測値
（過去バックテスト実績）
```

**実装**: `models/ensemble_model.py` を新規作成。`model_manager.py` で統合管理。

---

#### A-5. Optuna によるハイパーパラメータ自動最適化
**概要**: 現在固定のモデルパラメータをバックテスト Sharpe Ratio を目的関数として Optuna で最適化する。

**現状の課題**: グリッドサーチ（`backtest_optimize_pipeline.py`）は組み合わせ爆発が起きやすい。Optuna(TPE)なら試行数を大幅削減できる。

**実装**:
```python
# services/optuna_optimizer.py
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }
    # バックテスト実行 → Sharpeを返す
    return sharpe_ratio

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

---

### 🥈 Priority B — 効果大・実装コスト高

---

#### B-1. ニュース・センチメント特徴量
**概要**: 株価ニュースのテキストをNLP解析し、感情スコア（ポジティブ/ネガティブ）を特徴量に追加する。

| ソース | 取得方法 | 特徴量 |
|--------|---------|--------|
| Yahoo Finance ニュース | `yfinance.Ticker.news` | 直近N件の平均センチメントスコア |
| Bloomberg / Reuters RSS | feedparser | 銘柄名マッチング + VADER/FinBERT |
| X（旧Twitter）| Twitter API v2 | $ティッカーのポジティブ率 |

**推奨ライブラリ**: `transformers` (FinBERT) or `vaderSentiment`（軽量・高速）

**注意**: APIコスト・レート制限に注意。まず `yfinance.Ticker.news` から無料で始めるのが現実的。

---

#### B-2. LSTM / Transformer モデルの追加
**概要**: 時系列の長期依存性を学習できるニューラルネットワークモデルを追加し、アンサンブルに組み込む。

```
既存: XGBoost + LightGBM（表形式特徴量）
追加: LSTM or Temporal Fusion Transformer（時系列順序を保持）
統合: スタッキングアンサンブル
```

**実装**: `models/lstm_model.py`。`PyTorch` または `TensorFlow/Keras`。
**注意**: 学習時間・GPU要件が増えるため、週次バッチ学習に限定推奨。

---

#### B-3. 決算・イベントカレンダー特徴量
**概要**: 決算発表日・配当落ち日・株式分割・日銀会合・FOMC等のイベントを特徴量に組み込む。

```
決算発表3日前: event_pre_earnings = 1
決算発表当日:   event_earnings = 1
決算後5日:     event_post_earnings = 1
```

**効果**: 決算前後のボラティリティ急増期をモデルが認識できるようになる。
**取得先**: `yfinance.Ticker.calendar`、日本株は `kabutan.jp` スクレイピング。

---

#### B-4. モデルドリフト検出 + 自動再学習
**概要**: 本番予測の精度を継続監視し、精度劣化（ドリフト）を検出したら自動で再学習をトリガーする。

```
毎日: 予測結果 vs 実際の終値 → 予測精度ログ
週次: 直近4週のSharpeが基準値を下回ったら → retrain_trigger()
      → model_training_pipeline.run() 自動実行
```

**実装**: `services/drift_monitor.py`。DuckDB の `prediction_results` テーブルを参照。

---

## Part 2｜マネタイズアイデア

### 💰 Priority A — 早期収益化・実装コスト低〜中

---

#### M-1. 自動売買の完成（公式取引API への移行）
**概要**: 現在の Selenium スクレイピング実装（`src/sbi/`）を廃止し、**公式取引APIを持つ証券会社** に乗り換えてシグナルに基づく自動注文実行を実現する。**これが最も直接的な利益創出手段**。

---

**推奨: auカブコム証券 — kabu STATION® API**

| 項目 | 内容 |
|------|------|
| API種別 | REST API（ローカルホスト経由） |
| 提供形態 | kabu STATION® アプリを常駐させた上でローカルポートにHTTPリクエスト |
| 費用 | 口座開設・API利用ともに無料 |
| 対象市場 | 東証（現物・信用）、先物・オプション |
| ドキュメント | 公式OpenAPI仕様書あり（Swagger UI） |

```
kabu STATION® (常駐アプリ)
  ↑ REST API (localhost:18080)
Python (StockFixer)
  ├── POST /kabusapi/sendorder  → 注文発注
  ├── GET  /kabusapi/positions  → 保有ポジション照会
  ├── GET  /kabusapi/wallet/cash → 余力照会
  └── WebSocket /kabusapi/push  → リアルタイム価格受信
```

**実装すべき機能**:
- [ ] `src/kabu/kabu_api.py` を新規作成（kabu STATION® API クライアント）
- [ ] シグナルに基づく成行/指値注文の自動発注（`POST /sendorder`）
- [ ] 保有ポジション・余力の取得（`GET /positions`, `/wallet/cash`）
- [ ] 損切り・利確ラインに達したら自動決済注文
- [ ] 1銘柄あたり最大投資額の上限設定（リスク管理）
- [ ] 注文履歴のDuckDB保存
- [ ] WebSocket でリアルタイム株価を受信し、日中監視を実現

```python
# src/kabu/kabu_api.py（実装イメージ）
import httpx

KABU_BASE = "http://localhost:18080/kabusapi"

def get_token(api_password: str) -> str:
    res = httpx.post(f"{KABU_BASE}/token", json={"APIPassword": api_password})
    return res.json()["Token"]

def send_order(token: str, symbol: str, side: int, qty: int, price: float = 0):
    """side: 1=買 2=売, price=0 で成行"""
    headers = {"X-API-KEY": token}
    body = {
        "Password": "",  # 取引パスワード（環境変数から取得）
        "Symbol": symbol,
        "Exchange": 1,        # 1=東証
        "SecurityType": 1,    # 1=株式
        "Side": str(side),
        "CashMargin": 1,      # 1=現物
        "DelivType": 2,
        "FundType": "AA",
        "Qty": qty,
        "FrontOrderType": 10 if price == 0 else 20,  # 10=成行 20=指値
        "Price": price,
        "ExpireDay": 0,
    }
    return httpx.post(f"{KABU_BASE}/sendorder", headers=headers, json=body).json()
```

---

**代替案: Interactive Brokers — Client Portal API**

| 項目 | 内容 |
|------|------|
| API種別 | REST API（Client Portal Gateway 経由） |
| 費用 | 口座開設要・月次アクティビティ要件あり |
| 対象市場 | 東証を含むグローバル全市場 |
| 特徴 | 米国株・ETF・先物も同一APIで取引可能 |

米国株への展開を視野に入れる場合に選択する。

---

**リスク管理（必須）**:
```
1日の最大損失額 = 口座残高の2%まで
1銘柄の最大ポジション = 口座残高の10%まで
連続損失3回でその日の取引停止
```

---

#### M-2. Discord プレミアム配信（有料ティア）
**概要**: 現在の無料Discord Bot を拡張し、より詳細な分析レポートを有料会員に提供する。

| ティア | 価格 | 提供内容 |
|--------|------|---------|
| 無料 | 0円 | Top10/Worst10 + 予想変化率のみ |
| スタンダード | 980円/月 | 上記 + エントリー根拠（指標詳細）・推奨TP/SL |
| プレミアム | 2,980円/月 | 上記 + リアルタイムアラート・バックテスト詳細・マクロ分析 |

**実装**:
- Discord ロールによるアクセス制御
- Stripe or FANBOX による課金管理
- 有料コマンド (`/detail`, `/alert`, `/backtest`) を追加

---

#### M-3. パフォーマンスレポート自動生成
**概要**: 月次・週次でシステムの予測精度と仮想損益をPDFレポートとして自動生成・配信する。

**内容**:
- 予測精度（Hit Rate, Sharpe Ratio, Maximum Drawdown）
- Top10銘柄の実際の値動きvs予測
- Strategy別パフォーマンス比較

**実装**: `reportlab` または `WeasyPrint` でPDF生成。Discord DM または Email で配信。
**用途**: 自己検証用 + 有料サービスの信頼性証明として活用。

---

#### M-4. ポジションサイジング最適化（Kelly基準）
**概要**: 確率論的に最適な投資額を計算し、資本効率を最大化する。

```python
# Kelly Criterion（ケリー基準）
kelly_fraction = (win_rate * avg_win_size - (1 - win_rate) * avg_loss_size) / avg_win_size
invest_amount = capital * kelly_fraction * 0.5  # ハーフケリーで安全側に調整
```

**効果**: 高自信シグナルに多く賭け、低自信シグナルには少額投資。長期的な資産成長率が理論最大化される。

---

### 💰 Priority B — 中期的収益化

---

#### M-5. 複数証券会社対応（公式API 横断対応）
**概要**: kabu STATION® API（auカブコム）を主軸としつつ、公式APIを持つ他の証券会社・プロバイダーにも対応し、障害リスク分散と対象市場の拡大を図る。

| 証券会社 | API名称 | 対象市場 | 費用 |
|---------|---------|---------|------|
| auカブコム証券 | kabu STATION® API | 東証（現物・信用） | 無料 |
| Interactive Brokers | Client Portal API / TWS API | 東証・米国・グローバル | 口座要件あり |
| GMOクリック証券 | システムトレードAPI | 東証（現物・信用） | 無料 |

> ※ SBI証券・楽天証券・松井証券はリテール向け公式自動売買APIを提供していないため対象外とする。

**実装方針**:
```python
# 抽象化インターフェース（各社のAPI差異を吸収）
class BrokerBase(ABC):
    @abstractmethod
    def get_token(self) -> str: ...
    @abstractmethod
    def send_order(self, symbol, side, qty, price) -> dict: ...
    @abstractmethod
    def get_positions(self) -> list[dict]: ...
    @abstractmethod
    def get_balance(self) -> float: ...

class KabuBroker(BrokerBase): ...       # src/kabu/   ← Phase 1 で実装
class IBKRBroker(BrokerBase): ...       # src/ibkr/   ← 米国株対応時
class GMOBroker(BrokerBase): ...        # src/gmo/    ← バックアップ口座
```

**用途**: `BrokerBase` を介することで、証券会社を切り替えても `services/` 以上のロジックは無修正。複数口座への同時発注や、API障害時の自動フェイルオーバーにも対応できる設計にする。

---

#### M-6. Web ダッシュボード
**概要**: 現在のDiscord通知をWebUIに昇格させ、ブラウザから予測・ポジション・パフォーマンスをリアルタイム確認できるようにする。

**技術スタック**:
- Backend: Flask (既存APIを拡張)
- Frontend: Streamlit（実装コスト最小） or React（将来拡張性）
- Hosting: Fly.io / Render（無料枠あり）

**主要画面**:
1. ダッシュボード: 本日のTop10/Worst10 + 保有ポジション
2. 銘柄詳細: チャート + テクニカル指標 + AI予測根拠
3. バックテスト結果閲覧
4. パフォーマンス履歴

---

#### M-7. ペアトレード戦略
**概要**: 相関が高い2銘柄のスプレッド平均回帰を利用したマーケットニュートラル戦略を追加する。個別株の方向性リスクを取らず安定した収益を狙う。

```
例: 7203.T(トヨタ) と 7267.T(ホンダ) の価格スプレッドが
2σ以上乖離した場合: 割安側Buy + 割高側Sell
スプレッドが収束したら決済
```

**実装**: `strategy/pairs_trading.py`。`statsmodels` の共和分検定（Engle-Granger法）を利用。
**優位性**: 相場全体の騰落に依存しない。Sharpe比が高い傾向。

---

#### M-8. 予測API の有料公開（SaaS化）
**概要**: 既存の Flask API を整備し、APIキー認証付きで外部に有料提供する。

```
GET /api/v1/forecast?symbol=7203.T
Authorization: Bearer {api_key}

Response: {
  "symbol": "7203.T",
  "predicted_change": 0.018,
  "confidence": 0.72,
  "signals": {"RSI": "oversold", "MACD": "bullish_cross"},
  "updated_at": "2026-03-15T06:00:00+09:00"
}
```

**課金モデル**:
- 無料: 10リクエスト/日
- ライト: 1,980円/月・500リクエスト/日
- プロ: 9,800円/月・無制限

**実装**: `api/api_server.py` にAPIキー認証ミドルウェア追加。RateLimiting は `Flask-Limiter`。

---

## Part 3｜実装優先度マトリックス

```
高インパクト
     ↑
     │  [B-1 センチメント]  [M-1 自動売買完成] ← ★最優先
     │  [A-4 アンサンブル]  [M-2 有料Discord]
     │  [A-1 マクロ指標]   [A-5 Optuna]
     │
     │  [B-2 LSTM]          [M-6 Webダッシュボード]
     │  [B-3 イベント]      [M-7 ペアトレード]
     │
     │  [A-2 レジーム検出]  [M-3 レポート自動生成]
     │  [A-3 マルチTF]      [M-4 Kelly基準]
     ↓
低インパクト
     ←───────────────────────────────────────→
   低コスト                              高コスト
```

### 推奨実装順序

| フェーズ | 期間目安 | 実装内容 | 期待効果 |
|---------|---------|----------|---------|
| **Phase 1** | 〜1ヶ月 | A-1(マクロ指標) + A-5(Optuna) + M-1(自動売買完成) | 精度+5〜10%・実益創出 |
| **Phase 2** | 1〜2ヶ月 | A-2(レジーム検出) + A-4(アンサンブル) + M-4(Kelly基準) | ドローダウン削減・資本効率向上 |
| **Phase 3** | 2〜3ヶ月 | M-2(有料Discord) + M-3(レポート) + M-8(有料API) | 月次収益の確立 |
| **Phase 4** | 3ヶ月〜 | B-1(センチメント) + B-2(LSTM) + M-6(Web UI) | プレミアムサービス化 |

---

## 参考：現在の予測精度ベースライン

Phase 1 開始前にベースラインを計測しておくことを推奨:

```bash
# バックテストで現状精度を計測
python run_backtest.py --market jp --symbol 7203.T --start 2023-01-01 --end 2025-12-31
```

改善施策を追加するたびに同じコマンドで比較し、**精度の向上幅を数値で追跡**する。

---

*このドキュメントは実装アイデアの一覧です。各機能の詳細設計は対応する SKILL.md または設計ドキュメントで管理してください。*
