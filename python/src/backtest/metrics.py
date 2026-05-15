"""
バックテスト評価指標

equity_curve (日次・取引ごとの資産曲線 pd.Series) を入力として
各種パフォーマンス指標を計算する関数群。
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    trade_log: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
    cash_column: str = "cash",
) -> dict[str, Any]:
    """
    取引ログから主要メトリクスを一括計算する。

    Args:
        trade_log: Backtester.simulate_trading が返す DataFrame
                   (columns: date, action, price, qty, cash)
        initial_cash: 初期資金
        risk_free_rate: 無リスク金利（年率、小数）
        trading_days_per_year: 1年の取引日数（Sharpe計算用）
        cash_column: equity curve 算出に使う資産列名

    Returns:
        dict: 各指標の辞書
    """
    if trade_log is None or trade_log.empty or cash_column not in trade_log.columns:
        return _empty_metrics(initial_cash)

    # --- equity curve（決済後のキャッシュ推移）---
    # buy直後はキャッシュが激減するため sell 系時点のキャッシュのみを使う
    sell_actions = [
        "sell",
        "final_sell",
        "stop_loss",
        "take_profit",
        "short_cover",
        "final_short_cover",
    ]
    if "action" in trade_log.columns:
        sell_log = trade_log[trade_log["action"].isin(sell_actions)]
    else:
        sell_log = trade_log

    if sell_log.empty or "date" not in sell_log.columns:
        equity = pd.Series([initial_cash])
    else:
        equity = pd.concat(
            [
                pd.Series([initial_cash], index=[sell_log.iloc[0]["date"]]),
                sell_log.set_index("date")[cash_column],
            ]
        )

    final_cash = equity.iloc[-1]
    total_return = (final_cash - initial_cash) / initial_cash

    # 取引ペア（buy/sell）を抽出して勝率・Profit Factor を計算
    wins, losses, win_returns, loss_returns = _extract_trade_pnl(trade_log)
    trade_pnls = wins + losses
    num_trades = len(wins) + len(losses)
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0

    gross_profit = sum(w for w in wins if w > 0)
    gross_loss = abs(sum(loss for loss in losses if loss < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    avg_win = float(np.mean(win_returns)) if win_returns else 0.0
    avg_loss = float(np.mean(loss_returns)) if loss_returns else 0.0

    # --- Sharpe ratio（取引ごとのリターン列から） ---
    sharpe = _sharpe_ratio(trade_pnls, risk_free_rate, trading_days_per_year)

    # --- Maximum Drawdown ---
    max_dd = _max_drawdown(equity)

    buy_log = (
        trade_log[trade_log["action"] == "buy"] if "action" in trade_log.columns else pd.DataFrame()
    )
    position_fractions = pd.Series(dtype=float)
    position_values = pd.Series(dtype=float)
    atr_fallback_trades = 0
    if not buy_log.empty:
        if "position_fraction" in buy_log.columns:
            position_fractions = pd.to_numeric(
                buy_log["position_fraction"], errors="coerce"
            ).dropna()
        if "position_value" in buy_log.columns:
            position_values = pd.to_numeric(buy_log["position_value"], errors="coerce").dropna()
        if "atr_fallback_used" in buy_log.columns:
            atr_fallback_trades = int(
                buy_log["atr_fallback_used"]
                .fillna(False)
                .infer_objects(copy=False)
                .astype(bool)
                .sum()
            )

    return {
        "final_cash": round(final_cash, 2),
        "total_return": round(total_return, 6),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != math.inf else None,
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "avg_position_fraction": round(float(position_fractions.mean()), 6)
        if not position_fractions.empty
        else 0.0,
        "min_position_fraction": round(float(position_fractions.min()), 6)
        if not position_fractions.empty
        else 0.0,
        "max_position_fraction": round(float(position_fractions.max()), 6)
        if not position_fractions.empty
        else 0.0,
        "avg_position_value": round(float(position_values.mean()), 2)
        if not position_values.empty
        else 0.0,
        "atr_fallback_trades": atr_fallback_trades,
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
    }


# --- internal helpers ---


def _empty_metrics(initial_cash: float) -> dict[str, Any]:
    return {
        "final_cash": initial_cash,
        "total_return": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "profit_factor": None,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "avg_position_fraction": 0.0,
        "min_position_fraction": 0.0,
        "max_position_fraction": 0.0,
        "avg_position_value": 0.0,
        "atr_fallback_trades": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
    }


def _extract_trade_pnl(
    trade_log: pd.DataFrame,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Buy → sell のペアから各トレードの損益とリターン率を抽出する。

    Returns:
        (wins, losses, win_returns, loss_returns):
            wins: 正の損益リスト
            losses: 負の損益リスト
            win_returns: 勝ちトレードのリターン率リスト
            loss_returns: 負けトレードのリターン率リスト（絶対値）
    """
    buys: list[dict[str, Any]] = []
    wins: list[float] = []
    losses: list[float] = []
    win_returns: list[float] = []
    loss_returns: list[float] = []

    for _, row in trade_log.iterrows():
        action = row.get("action", "")
        if action == "buy":
            buys.append({"price": row["price"], "qty": row["qty"]})
        elif action in ("sell", "final_sell", "stop_loss", "take_profit") and buys:
            buy = buys.pop(0)
            pnl = (row["price"] - buy["price"]) * buy["qty"]
            ret = (row["price"] - buy["price"]) / buy["price"] if buy["price"] > 0 else 0.0
            if pnl >= 0:
                wins.append(pnl)
                win_returns.append(ret)
            else:
                losses.append(pnl)
                loss_returns.append(abs(ret))

    return wins, losses, win_returns, loss_returns


def compute_cost_comparison_metrics(
    trade_log: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    """控除後（net）と控除前（gross）のKPI比較を返す。"""
    net = compute_metrics(
        trade_log=trade_log,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        cash_column="cash",
    )
    gross = compute_metrics(
        trade_log=trade_log,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        cash_column="cash_gross",
    )

    result = {
        # 互換性維持: 既存キーは net を据え置く
        **net,
        # 追加: 比較用の明示キー
        "gross_final_cash": gross.get("final_cash", initial_cash),
        "gross_total_return": gross.get("total_return", 0.0),
        "gross_sharpe_ratio": gross.get("sharpe_ratio", 0.0),
        "gross_max_drawdown": gross.get("max_drawdown", 0.0),
        "cost_impact_cash": round(
            gross.get("final_cash", initial_cash) - net.get("final_cash", initial_cash), 2
        ),
        "cost_impact_return": round(
            gross.get("total_return", 0.0) - net.get("total_return", 0.0), 6
        ),
    }
    return result


def _sharpe_ratio(
    pnl_list: list[float],
    risk_free_rate: float,
    trading_days_per_year: int,
) -> float:
    """取引単位の損益リストから Sharpe ratio を計算する"""
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    # 年率換算（取引回数ベース）
    trades_per_year = trading_days_per_year  # 近似値
    daily_rf = risk_free_rate / trades_per_year
    return float((mean - daily_rf) / std * math.sqrt(trades_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """資産曲線から最大ドローダウン（負の小数）を計算する"""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


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
        raise ImportError("matplotlib がインストールされていません。`pip install matplotlib` を実行してください。")

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
    title = f"{market.upper()}/{symbol} バックテスト結果" if market or symbol else "バックテスト結果"
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


def compute_metrics_by_regime(
    trade_log: pd.DataFrame,
    price_df: pd.DataFrame,
    initial_cash: float,
    ema_window: int = 200,
    atr_window: int = 14,
) -> dict[str, dict[str, Any]]:
    """
    バックテスト取引ログをレジーム別（bull / bear / range）に分割し、各指標を返す。

    完了条件 (Issue #24):
        レジーム別成績（Net Return / Sharpe / Hit Rate）が Walk-Forward レポートに出力される。

    Args:
        trade_log:    Backtester.simulate_trading が返す DataFrame (columns: date, action, ...)
        price_df:     Close, High, Low を含む OHLCV DataFrame（trade_log と同じ日付範囲）
        initial_cash: 初期資金
        ema_window:   classify_regime に渡す EMA ウィンドウ幅（デフォルト: 200）
        atr_window:   classify_regime に渡す ATR ウィンドウ幅（デフォルト: 14）

    Returns:
        dict: レジームラベル ("bull", "bear", "range", "all") をキーとする metrics dict
    """
    from src.market_data.technical import classify_regime

    results: dict[str, dict[str, Any]] = {}

    # 全件合算メトリクス
    results["all"] = compute_metrics(trade_log, initial_cash)

    if trade_log is None or trade_log.empty or price_df is None or price_df.empty:
        return results

    if "date" not in trade_log.columns:
        return results

    regime_series = classify_regime(price_df, ema_window=ema_window, atr_window=atr_window)
    if regime_series.empty:
        return results

    # DatetimeIndex に統一
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)

    trade_log = trade_log.copy()
    trade_log["_date_dt"] = pd.to_datetime(trade_log["date"])
    trade_log["_regime"] = trade_log["_date_dt"].map(
        lambda d: regime_series.asof(d) if d >= regime_series.index.min() else None
    )

    for label in ("bull", "bear", "range"):
        leg = trade_log[trade_log["_regime"] == label].drop(columns=["_date_dt", "_regime"])
        if leg.empty:
            results[label] = _empty_metrics(initial_cash)
        else:
            results[label] = compute_metrics(leg, initial_cash)

    return results
