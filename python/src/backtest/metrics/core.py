"""バックテスト評価指標（コア計算）

equity_curve (日次・取引ごとの資産曲線 pd.Series) を入力として
主要パフォーマンス指標（リターン・勝率・Sharpe・MaxDD 等）を計算する関数群。
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.backtest.data_port import get_backtest_data_port
from src.backtest.metrics.overfitting import deflated_sharpe_ratio, monte_carlo_equity


def compute_metrics(
    trade_log: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
    cash_column: str = "cash",
    n_trials: int = 0,
    equity_series: Optional[pd.Series] = None,
    include_monte_carlo: bool = False,
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
        equity_series: 各バーの mark-to-market 日次 equity（保有中の含み損益込み）。
                       渡された場合は max_drawdown をこの系列から算出する。None の場合は
                       決済時点キャッシュベースの equity にフォールバックする（後方互換）。
        include_monte_carlo: True の場合、取引損益系列をブートストラップした
                       Monte Carlo リスク統計（mc_ プレフィックス）を結果に追加する。
                       1000回シミュレーションのためグリッドサーチ等の繰り返し呼び出しでは
                       既定で無効（単発のバックテストレポート向け）。

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
    # 符号付き取引リターン率（%）。複数銘柄の取引をプールしてSharpe/DSRを算出する際、
    # 銘柄間で価格スケールが異なる trade_pnls（$建て）をそのまま混ぜると高価格銘柄に
    # 偏るため、スケール非依存なリターン率で揃える（#630）。
    trade_returns = win_returns + [-r for r in loss_returns]

    # --- Sharpe ratio（取引ごとの損益列から） ---
    # 取引単位 Sharpe（非年率）を素に、実取引頻度で年率化する。
    # sharpe_per_trade は DSR 入力（飽和回避）に使う。
    sharpe_per_trade = _sharpe_per_trade(
        trade_pnls, risk_free_rate / max(num_trades, 1) if num_trades > 0 else 0.0
    )
    trades_per_year = _estimate_trades_per_year(trade_log, num_trades, trading_days_per_year)
    sharpe = _annualize_sharpe(sharpe_per_trade, trades_per_year)

    # --- Maximum Drawdown ---
    # mark-to-market 日次 equity が渡された場合はそれを使う（保有中の含み損益を反映）。
    # 渡されない場合は決済時点キャッシュベースの equity にフォールバック（後方互換）。
    dd_equity = equity_series if equity_series is not None and not equity_series.empty else equity
    max_dd = _max_drawdown(dd_equity)

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
                buy_log["atr_fallback_used"].fillna(False).infer_objects().astype(bool).sum()
            )

    result: dict[str, Any] = {
        "final_cash": round(final_cash, 2),
        "total_return": round(total_return, 6),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != math.inf else None,
        "sharpe_ratio": round(sharpe, 4),
        "sharpe_per_trade": round(sharpe_per_trade, 6),
        "max_drawdown": round(max_dd, 6),
        "avg_position_fraction": (
            round(float(position_fractions.mean()), 6) if not position_fractions.empty else 0.0
        ),
        "min_position_fraction": (
            round(float(position_fractions.min()), 6) if not position_fractions.empty else 0.0
        ),
        "max_position_fraction": (
            round(float(position_fractions.max()), 6) if not position_fractions.empty else 0.0
        ),
        "avg_position_value": (
            round(float(position_values.mean()), 2) if not position_values.empty else 0.0
        ),
        "atr_fallback_trades": atr_fallback_trades,
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "trade_returns": trade_returns,
    }
    if n_trials > 0:
        # DSR には非年率の取引単位 Sharpe を渡す（年率化済み Sharpe だと z 値が
        # 巨大化して DSR が 0/1 に飽和し、過学習検知が機能しなくなるため）。
        result["dsr"] = deflated_sharpe_ratio(sharpe_per_trade, n_trials, num_trades)
    if include_monte_carlo:
        mc = monte_carlo_equity(trade_pnls, initial_cash)
        result.update({f"mc_{key}": value for key, value in mc.items()})
    return result


# --- internal helpers ---


def _empty_metrics(initial_cash: float) -> dict[str, Any]:
    return {
        "final_cash": initial_cash,
        "total_return": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "profit_factor": None,
        "sharpe_ratio": 0.0,
        "sharpe_per_trade": 0.0,
        "max_drawdown": 0.0,
        "avg_position_fraction": 0.0,
        "min_position_fraction": 0.0,
        "max_position_fraction": 0.0,
        "avg_position_value": 0.0,
        "atr_fallback_trades": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "trade_returns": [],
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
    equity_net: Optional[pd.Series] = None,
    equity_gross: Optional[pd.Series] = None,
    include_monte_carlo: bool = False,
) -> dict[str, Any]:
    """控除後（net）と控除前（gross）のKPI比較を返す。

    equity_net / equity_gross に各バーの mark-to-market 日次 equity を渡すと、
    max_drawdown を保有中の含み損益込みで算出する（#492）。
    include_monte_carlo: True の場合、net 側に Monte Carlo リスク統計（mc_ プレフィックス）
        を追加する（実運用で晒すコスト後の分布を見たいため gross 側には付与しない）。
    """
    net = compute_metrics(
        trade_log=trade_log,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        cash_column="cash",
        equity_series=equity_net,
        include_monte_carlo=include_monte_carlo,
    )
    gross = compute_metrics(
        trade_log=trade_log,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        cash_column="cash_gross",
        equity_series=equity_gross,
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


def _sharpe_per_trade(pnl_list: list[float], risk_free_per_trade: float = 0.0) -> float:
    """取引単位の Sharpe（mean/std、年率化なし）を返す。

    DSR の入力（López de Prado の式は非年率の per-observation Sharpe を前提）と、
    年率化 Sharpe の素として使う。
    """
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list, dtype=float)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float((arr.mean() - risk_free_per_trade) / std)


def _estimate_trades_per_year(
    trade_log: pd.DataFrame, num_trades: int, trading_days_per_year: int
) -> float:
    """バックテスト期間の暦スパンから実取引頻度（年あたり取引数）を推定する。

    旧実装は一律 252 取引/年と仮定して年率化していたが、これは実取引頻度を無視し
    Sharpe を頻度非整合に膨張させていた。期間が不明な場合のみ従来挙動（252）に
    フォールバックする。
    """
    if num_trades < 1 or "date" not in trade_log.columns:
        return float(trading_days_per_year)
    dates = pd.to_datetime(trade_log["date"], errors="coerce").dropna()
    if dates.empty:
        return float(trading_days_per_year)
    span_days = (dates.max() - dates.min()).days
    if span_days <= 0:
        return float(trading_days_per_year)
    years = span_days / 365.25
    return num_trades / years


def _annualize_sharpe(sharpe_per_trade: float, trades_per_year: float) -> float:
    """取引単位 Sharpe を実取引頻度で年率化する。"""
    if trades_per_year <= 0:
        return 0.0
    return float(sharpe_per_trade * math.sqrt(trades_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """資産曲線から最大ドローダウン（負の小数）を計算する"""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


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
    results: dict[str, dict[str, Any]] = {}

    # 全件合算メトリクス
    results["all"] = compute_metrics(trade_log, initial_cash)

    if trade_log is None or trade_log.empty or price_df is None or price_df.empty:
        return results

    if "date" not in trade_log.columns:
        return results

    regime_series = get_backtest_data_port().classify_regime(
        price_df, ema_window=ema_window, atr_window=atr_window
    )
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
