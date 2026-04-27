"""
yfinance クライアントラッパー（後方互換 re-export）

実装は src.market_data.yf_client に移動しました。
このモジュールは移行期間中の後方互換のために残しています。

    # 新しい import パス（推奨）
    from src.market_data import yf_client

    # 旧パス（後方互換・フェーズ4で削除予定）
    from src.utils import yf_client
"""

# re-export: 実装は market_data.yf_client に移行済み
from src.market_data.yf_client import download, ticker_history  # noqa: F401
