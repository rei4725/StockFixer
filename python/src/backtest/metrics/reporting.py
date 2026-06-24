"""バックテスト結果の可視化・ベンチマーク取得

equity curve / drawdown のグラフ化と、ベンチマーク指数の期間リターン取得。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_backtest(
    trade_log: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: str,
    market: str = "",
    symbol: str = "",
    initial_cash: float = 1_000_000,
) -> str:
    """
    バックテスト結果をグラフ化して PNG として保存する。

    Args:
        trade_log: Backtester.simulate_trading が返す取引ログ DataFrame
        metrics: compute_metrics が返す指標辞書
        output_dir: PNG 保存先ディレクトリ
        market: マーケット識別子 (ファイル名用)
        symbol: 銃柔シンボル (ファイル名用)
        initial_cash: 初期資金 (ベースライン表示用)

    Returns:
        保存した PNG ファイルの絶対パス。
        失敗時は空文字列を返す。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 非インタラクティブ環境向け
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib がインストールされていません。`pip install matplotlib` を実行してください。"
        )

    if trade_log is None or trade_log.empty:
        return ""

    # --- equity curve 構築 ---
    sell_actions = [
        "sell",
        "final_sell",
        "stop_loss",
        "take_profit",
        "short_cover",
        "final_short_cover",
    ]
    sell_log = (
        trade_log[trade_log["action"].isin(sell_actions)]
        if "action" in trade_log.columns
        else trade_log
    )
    if sell_log.empty or "date" not in sell_log.columns:
        return ""

    equity = pd.concat(
        [
            pd.Series([initial_cash], index=[sell_log.iloc[0]["date"]]),
            sell_log.set_index("date")["cash"],
        ]
    )
    equity.index = pd.to_datetime(equity.index)
    equity = equity.sort_index()

    cum_return_pct = (equity / initial_cash - 1) * 100
    roll_max = equity.cummax()
    drawdown_pct = (equity - roll_max) / roll_max * 100

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    title = (
        f"{market.upper()}/{symbol} バックテスト結果" if market or symbol else "バックテスト結果"
    )
    fig.suptitle(title, fontsize=14)

    # 上段: 累積リターン曲線
    ax1.plot(cum_return_pct.index, cum_return_pct.values, color="steelblue", label="戦略リターン")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, label="ベースライン")
    ax1.set_ylabel("累積リターン (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 下段: ドローダウン曲線
    ax2.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="crimson", alpha=0.5)
    ax2.plot(drawdown_pct.index, drawdown_pct.values, color="crimson", linewidth=0.8)
    ax2.set_ylabel("ドローダウン (%)")
    ax2.set_xlabel("日付")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # type: ignore[no-untyped-call]
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # 指標サマリー
    info_parts = [
        f"Total Return: {metrics.get('total_return', 0):.2%}",
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
        f"MaxDD: {metrics.get('max_drawdown', 0):.2%}",
        f"Trades: {metrics.get('num_trades', 0)}",
        f"Win Rate: {metrics.get('win_rate', 0):.1%}",
    ]
    fig.text(0.5, 0.01, "  |  ".join(info_parts), ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{market}_{symbol}_{ts}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def fetch_benchmark_returns(
    ticker: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """
    ベンチマーク指数の期間リターンを取得する。

    Args:
        ticker: Yahoo Finance ティッカー (例: "^N225", "^GSPC")
        start: 開始日 YYYY-MM-DD
        end: 終了日 YYYY-MM-DD

    Returns:
        dict: {"ticker": str, "total_return": float, "start": str, "end": str}
        取得失敗時は total_return=None
    """
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {"ticker": ticker, "total_return": None, "start": start, "end": end}
        close = df["Close"].dropna()
        if len(close) < 2:
            return {"ticker": ticker, "total_return": None, "start": start, "end": end}
        total_return = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
        return {
            "ticker": ticker,
            "total_return": round(total_return, 6),
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
        }
    except Exception:
        logger.warning("ベンチマーク総リターン計算失敗: ticker=%s", ticker, exc_info=True)
        return {"ticker": ticker, "total_return": None, "start": start, "end": end}


BENCHMARK_TICKERS = {
    "n225": "^N225",
    "sp500": "^GSPC",
    "topix": "^TOPX",
    "nasdaq": "^IXIC",
}
