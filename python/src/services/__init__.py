"""
services 層

ウォッチリスト CRUD は src.services.watchlist に実装されています。
他の Bounded Context は各トップレベルパッケージを直接参照してください:

    src.backtest       — バックテスト・最適化
    src.market_data    — データ取得・特徴量生成
    src.orchestration  — スケジューラ
    src.prediction     — 予測・学習
    src.reporting      — レポート・クエリ
    src.trading        — 注文・リスク管理
    src.watchlist      — ウォッチリスト（バッチ管理）
"""
