"""
pair_trading.py — ペアトレード（市場中立戦略）

高相関ペアを自動選定し、Engle-Granger コインテグレーション検定で
スプレッドの統計的有意性を確認した上で、スプレッド収束を狙う
ロング/ショート市場中立戦略を実装する（R-403）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Engle-Granger 残差の ADF 臨界値（5%, k=2 変数, 漸近値 MacKinnon 1994）
_EG_CRITICAL_VALUE_5PCT = -3.0


@dataclass
class PairBacktestResult:
    """ペアトレードのバックテスト結果。"""

    symbol_a: str
    symbol_b: str
    correlation: float
    hedge_ratio: float
    adf_stat: float
    total_return: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float


def select_correlated_pairs(
    prices: pd.DataFrame,
    corr_threshold: float = 0.8,
) -> list[tuple[str, str, float]]:
    """
    価格 DataFrame から高相関ペアを選定する。

    Args:
        prices: 列が銘柄コード、行が日付の終値 DataFrame
        corr_threshold: 相関係数の絶対値の下限（デフォルト 0.8）

    Returns:
        (symbol_a, symbol_b, correlation) のリスト（|相関係数| 降順）
    """
    if prices.shape[1] < 2:
        return []

    corr = prices.corr()
    symbols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            c = float(corr.iloc[i, j])  # type: ignore[arg-type]
            if np.isfinite(c) and abs(c) >= corr_threshold:
                pairs.append((symbols[i], symbols[j], c))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def _adf_statistic(series: np.ndarray, max_lag: int = 1) -> float:
    """
    Augmented Dickey-Fuller 検定統計量を計算する（numpy のみ使用）。

    回帰: ΔY_t = α + β*Y_{t-1} + Σγ_i*ΔY_{t-i} + ε
    H0: β=0（単位根あり）, H1: β<0（定常）
    """
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)
    n = len(dy)

    if n <= max_lag + 2:
        return 0.0

    cols = [y[max_lag:-1]]
    for lag in range(1, max_lag + 1):
        cols.append(dy[max_lag - lag : n - lag])
    cols.append(np.ones(n - max_lag))

    X = np.column_stack(cols)
    Y = dy[max_lag:]

    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        n_obs = len(Y)
        k = X.shape[1]
        sigma2 = np.sum(resid**2) / max(n_obs - k, 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * XtX_inv[0, 0])
        if se < 1e-10:
            return 0.0
        return float(beta[0] / se)
    except np.linalg.LinAlgError:
        return 0.0


def check_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
) -> tuple[float, float, bool]:
    """
    Engle-Granger コインテグレーション検定（2 変数, 5% 水準）。

    Args:
        price_a: 銘柄 A の価格系列
        price_b: 銘柄 B の価格系列

    Returns:
        (hedge_ratio, adf_stat, is_cointegrated) のタプル
    """
    a = np.asarray(price_a.dropna(), dtype=float)
    b = np.asarray(price_b.dropna(), dtype=float)
    n = min(len(a), len(b))

    if n < 30:
        logger.warning("[pair] データ不足のためコインテグレーション検定をスキップ (n=%d)", n)
        return 1.0, 0.0, False

    a, b = a[-n:], b[-n:]

    # Step 1: OLS で hedge ratio を推定 (a = α + β*b + ε)
    X = np.column_stack([b, np.ones(n)])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
    except np.linalg.LinAlgError:
        return 1.0, 0.0, False

    hedge_ratio = float(beta[0])
    spread = a - hedge_ratio * b - float(beta[1])

    # Step 2: ADF 検定
    adf_stat = _adf_statistic(spread, max_lag=1)
    is_cointegrated = adf_stat < _EG_CRITICAL_VALUE_5PCT

    logger.info(
        "[pair] コインテグレーション検定: hedge_ratio=%.4f, ADF=%.4f, critical=%.2f, result=%s",
        hedge_ratio,
        adf_stat,
        _EG_CRITICAL_VALUE_5PCT,
        "共和分あり" if is_cointegrated else "共和分なし",
    )
    return hedge_ratio, adf_stat, is_cointegrated


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """ヘッジ比率を用いてミーン・デミーンドのスプレッドを計算する。"""
    spread = price_a - hedge_ratio * price_b
    return spread - spread.mean()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def generate_pair_signals(
    spread: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 20,
) -> pd.DataFrame:
    """
    スプレッド z スコアに基づくペアトレードシグナルを生成する。

    スプレッド上昇時（z > entry_z）: A ショート(-1), B ロング(+1)
    スプレッド下降時（z < -entry_z）: A ロング(+1), B ショート(-1)
    |z| < exit_z でイグジット。

    Returns:
        position_a, position_b 列を持つ DataFrame（+1/-1/0）
    """
    z = _rolling_zscore(spread, window)
    pos_a = np.zeros(len(spread))
    pos_b = np.zeros(len(spread))

    in_position = False
    direction = 0  # +1: A ロング/B ショート, -1: A ショート/B ロング

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            continue

        if not in_position:
            if zi > entry_z:
                direction = -1
                in_position = True
            elif zi < -entry_z:
                direction = 1
                in_position = True
        else:
            if abs(zi) < exit_z:
                in_position = False
                direction = 0

        if in_position:
            pos_a[i] = float(direction)
            pos_b[i] = float(-direction)

    return pd.DataFrame(
        {"position_a": pos_a, "position_b": pos_b},
        index=spread.index,
    )


def backtest_pair(
    prices: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 20,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
) -> Optional[PairBacktestResult]:
    """
    ペアトレード戦略のバックテストを実行する（等金額ロング/ショート）。

    Args:
        prices: 列が銘柄コード、行が日付の終値 DataFrame
        symbol_a, symbol_b: ペアの銘柄コード
        entry_z: エントリーの z スコア閾値
        exit_z: イグジットの z スコア閾値
        window: ローリング統計ウィンドウ日数
        initial_cash: 初期資金
        fee_rate: 片道手数料率

    Returns:
        PairBacktestResult（コインテグレーションなし・データ不足の場合は None）
    """
    if symbol_a not in prices.columns or symbol_b not in prices.columns:
        logger.error("[pair] 銘柄がデータに存在しない: %s, %s", symbol_a, symbol_b)
        return None

    price_a = prices[symbol_a].dropna()
    price_b = prices[symbol_b].dropna()
    common_idx = price_a.index.intersection(price_b.index)

    if len(common_idx) < 50:
        logger.warning("[pair] 共通データ不足: %d 行", len(common_idx))
        return None

    price_a = price_a.loc[common_idx]
    price_b = price_b.loc[common_idx]

    hedge_ratio, adf_stat, is_cointegrated = check_cointegration(price_a, price_b)
    if not is_cointegrated:
        logger.info(
            "[pair] %s/%s はコインテグレーションなし (ADF=%.4f)", symbol_a, symbol_b, adf_stat
        )
        return None

    corr = float(price_a.corr(price_b))
    spread = compute_spread(price_a, price_b, hedge_ratio)
    signals = generate_pair_signals(spread, entry_z=entry_z, exit_z=exit_z, window=window)

    ret_a = price_a.pct_change().fillna(0.0).values
    ret_b = price_b.pct_change().fillna(0.0).values
    pos_a = signals["position_a"].values
    pos_b = signals["position_b"].values

    half_cash = initial_cash / 2.0
    equity = initial_cash
    daily_returns: list[float] = []
    trades: list[float] = []
    prev_pos_a = 0.0
    prev_pos_b = 0.0

    for i in range(len(signals)):
        cur_pa, cur_pb = pos_a[i], pos_b[i]

        if cur_pa != prev_pos_a or cur_pb != prev_pos_b:
            equity -= equity * fee_rate * 2

        pnl = cur_pa * half_cash * ret_a[i] + cur_pb * half_cash * ret_b[i]
        new_equity = equity + pnl
        daily_returns.append((new_equity - equity) / max(equity, 1.0))

        if (prev_pos_a != 0.0 or prev_pos_b != 0.0) and cur_pa == 0.0:
            trades.append(pnl)

        prev_pos_a, prev_pos_b = cur_pa, cur_pb
        equity = new_equity

    if not daily_returns:
        return None

    rets = np.array(daily_returns)
    total_return = (equity - initial_cash) / initial_cash
    num_trades = len(trades)
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades > 0 else 0.0
    std = float(np.std(rets))
    sharpe = float(np.mean(rets) / std * np.sqrt(252)) if std > 1e-10 else 0.0
    cumrets = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cumrets)
    max_drawdown = float(np.min((cumrets - peak) / np.maximum(peak, 1e-10)))

    logger.info(
        "[pair] %s/%s バックテスト完了: return=%.2f%%, trades=%d, win=%.1f%%, sharpe=%.2f",
        symbol_a,
        symbol_b,
        total_return * 100,
        num_trades,
        win_rate * 100,
        sharpe,
    )

    return PairBacktestResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        correlation=corr,
        hedge_ratio=hedge_ratio,
        adf_stat=adf_stat,
        total_return=float(total_return),
        num_trades=num_trades,
        win_rate=float(win_rate),
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
    )


def run_pair_trading_scan(
    prices: pd.DataFrame,
    corr_threshold: float = 0.8,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 20,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
) -> list[PairBacktestResult]:
    """
    全銘柄ペアをスキャンしてペアトレード候補を返す。

    高相関ペアを自動選定し、コインテグレーション検定を通過したペアの
    バックテスト結果を Sharpe 比降順で返す。

    Args:
        prices: 列が銘柄コード、行が日付の終値 DataFrame
        corr_threshold: 高相関とみなす相関係数の絶対値下限
        entry_z, exit_z: エントリー/イグジット z スコア
        window: ローリング統計ウィンドウ日数
        initial_cash: バックテスト初期資金
        fee_rate: 片道手数料率

    Returns:
        コインテグレーション検定を通過したペアのバックテスト結果リスト（sharpe 降順）
    """
    pairs = select_correlated_pairs(prices, corr_threshold=corr_threshold)
    logger.info("[pair] 高相関ペア候補: %d ペア (threshold=%.2f)", len(pairs), corr_threshold)

    results: list[PairBacktestResult] = []
    for sym_a, sym_b, _ in pairs:
        result = backtest_pair(
            prices=prices,
            symbol_a=sym_a,
            symbol_b=sym_b,
            entry_z=entry_z,
            exit_z=exit_z,
            window=window,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
        )
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    logger.info("[pair] コインテグレーション通過ペア: %d ペア", len(results))
    return results
