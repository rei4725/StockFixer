# プロジェクト：CuteStock - 株式自動売買システム

## 1. 概要
- **目的**: Pythonによる戦略ロジックとAI予測、C#による注文実行とUI表示を組み合わせたリアルタイム自動売買システムを構築し、利益を最大化する。
- **技術スタック**: Python (分析・AI), C# (注文・UI)
- **制約**: (ここに制約を記述)
- **関連ドキュメント**: (ここに関連ドキュメントを記述)

---

## 2. システム構成

### 2.1. Python側（戦略・AI・分析）
- **データ取得**: `yfinance` を利用した株価、為替、ETFデータの取得
- **テクニカル分析**: `ta`ライブラリなどを用いたテクニカル指標（RSI, MACD, EMA, ATR）の生成
- **AIモデル**: XGBoost, LightGBM を用いた価格予測モデルの構築
- **シグナル生成**: 分析結果に基づき、売買シグナル（Buy/Sell/Hold）を生成
- **APIサーバー**: Flask を用いたREST APIを構築し、C#側との連携を実現
- **Discord連携** : Discordサーバーでのメッセージを受けて、値上がり予想をDiscordに投稿する

### 2.2. C#側（注文・UI・運用）
- **証券会社連携**: 楽天証券、SBI証券などのAPIと連携し、注文実行とポジション管理を行う
- **UI**: WPF を用いたリアルタイムUIを開発し、運用状況を可視化
- **運用管理**: ログ管理、Slack通知、データベースへの記録
- **通信**: Python側APIとの通信（REST API）

---

## 3. アーキテクチャ図

```mermaid
graph TD
    subgraph Python: Analysis & Strategy
        A[Data Acquisition (yfinance, ccxt)] --> B[Technical Analysis (ta)];
        B --> C[AI Price Prediction (XGBoost)];
        C --> D[Signal Generation];
        D --> E[REST API Server (Flask)];
    end

    subgraph C#: Execution & UI
        F[Order Execution (Broker API)]
        G[UI Display (WPF)]
        H[Logging & Notifications]
    end

    E -- REST API --> F;
    F --> G;
    F --> H;
