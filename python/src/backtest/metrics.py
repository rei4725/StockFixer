"""
バックテスト評価指標

equity_curve (日次・取引ごとの資産曲線 pd.Series) を入力として
各種パフォーマンス指標を計算する関数群。
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from src.backtest.data_port import get_backtest_data_port
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    trade_log: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
    cash_column: str = "cash",
    n_trials: int = 0,
    equity_series: Optional[pd.Series] = None,
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
    }
    if n_trials > 0:
        # DSR には非年率の取引単位 Sharpe を渡す（年率化済み Sharpe だと z 値が
        # 巨大化して DSR が 0/1 に飽和し、過学習検知が機能しなくなるため）。
        result["dsr"] = deflated_sharpe_ratio(sharpe_per_trade, n_trials, num_trades)
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
) -> dict[str, Any]:
    """控除後（net）と控除前（gross）のKPI比較を返す。

    equity_net / equity_gross に各バーの mark-to-market 日次 equity を渡すと、
    max_drawdown を保有中の含み損益込みで算出する（#492）。
    """
    net = compute_metrics(
        trade_log=trade_log,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        cash_column="cash",
        equity_series=equity_net,
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


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (López de Prado, 2018).

    Args:
        sharpe:    算出済みの Sharpe Ratio
        n_trials:  グリッドサーチの試行数（例: パラメータ組み合わせ数）
        n_obs:     バックテストの取引回数（observations）
        skewness:  取引リターンの歪度（デフォルト 0 = 正規分布近似）
        kurtosis:  取引リターンの尖度（デフォルト 3 = 正規分布近似）

    Returns:
        0〜1 の確率値。1 に近いほど「偶然ではないスキルベースの成績」。
    """
    EULER_GAMMA = 0.5772156649

    var = (1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2) / max(n_obs - 1, 1)
    sr_std = math.sqrt(max(var, 0.0))
    if sr_std <= 0 or n_trials <= 0:
        return 0.0

    # n_trials=1 は多重比較なし: 期待最大値=0 (1試行のN(0,1)の期待値)
    if n_trials == 1:
        expected_max = 0.0
    else:
        expected_max = sr_std * (
            (1 - EULER_GAMMA) * _norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * _norm.ppf(1 - 1 / (n_trials * math.e))
        )
    z = (sharpe - expected_max) / sr_std
    return float(_norm.cdf(z))


def _sharpe_by_column(block: np.ndarray) -> np.ndarray:
    """各列（=各戦略候補）の Sharpe（mean/std）を返す。std=0 の列は 0。"""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1) if block.shape[0] > 1 else np.ones(block.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, mean / std, 0.0)
    return np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 10,
) -> float:
    """CSCV (Combinatorially Symmetric Cross-Validation) による PBO を計算する。

    López de Prado (2014) の手法。N 個の候補戦略（パラメータ組合せ等）の期間
    リターン行列から「インサンプル最良の戦略がアウトオブサンプルで中央値を下回る確率」
    を推定する。PBO が高い（>0.5 目安）ほど、最適化結果が過学習である可能性が高い。

    Args:
        returns_matrix: shape (T, N) の配列。T=期間数、N=候補戦略数。各列が1戦略の
            期間リターン系列（例: トレードごと or 日次リターン）。
        n_splits: 期間 T を分割するブロック数 S（偶数）。C(S, S/2) 通りの IS/OOS 分割を作る。

    Returns:
        PBO（0〜1）。候補が2未満・データ不足のときは NaN。
    """
    from itertools import combinations

    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    if n_splits % 2 != 0:
        n_splits -= 1
    if n_splits < 2:
        return float("nan")

    T, N = M.shape
    block_size = T // n_splits
    if block_size < 1:
        return float("nan")

    # 末尾の端数は捨てて等分割
    blocks = [M[i * block_size : (i + 1) * block_size] for i in range(n_splits)]
    indices = list(range(n_splits))

    lambdas: list[float] = []
    for is_sel in combinations(indices, n_splits // 2):
        is_set = set(is_sel)
        oos_sel = [i for i in indices if i not in is_set]
        is_mat = np.vstack([blocks[i] for i in is_sel])
        oos_mat = np.vstack([blocks[i] for i in oos_sel])

        is_perf = _sharpe_by_column(is_mat)
        oos_perf = _sharpe_by_column(oos_mat)

        # IS 最良戦略
        n_star = int(np.argmax(is_perf))
        # その戦略の OOS 相対順位 ω = rank / (N+1)（rank: 1=最下位 … N=最上位）
        order = np.argsort(oos_perf, kind="stable")  # 昇順
        rank = int(np.where(order == n_star)[0][0]) + 1
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        lambdas.append(math.log(omega / (1.0 - omega)))

    if not lambdas:
        return float("nan")
    arr = np.asarray(lambdas)
    # λ <= 0 = IS最良が OOS で中央値以下 = 過学習の兆候
    return float(np.mean(arr <= 0.0))


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


def monte_carlo_equity(
    trade_pnl_list: list[float],
    initial_cash: float,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    取引損益系列をランダムシャッフルして equity curve を n_simulations 回シミュレートし、
    最大ドローダウンと最終資産の分布統計を返す。

    Args:
        trade_pnl_list: 取引ごとの損益リスト（正=利益, 負=損失）
        initial_cash:   初期資金
        n_simulations:  シミュレーション回数
        confidence:     信頼水準（デフォルト 0.95 = 95%）
        seed:           乱数シード（再現性確保用）

    Returns:
        {
            "max_drawdown_mean":  最大ドローダウンの平均（負の小数）,
            "max_drawdown_p95":   最大ドローダウンの 95 パーセンタイル（負の小数）,
            "final_cash_p05":     最終資産の 5 パーセンタイル,
            "final_cash_p50":     最終資産の中央値,
            "final_cash_p95":     最終資産の 95 パーセンタイル,
        }
    """
    if not trade_pnl_list:
        return {
            "max_drawdown_mean": 0.0,
            "max_drawdown_p95": 0.0,
            "final_cash_p05": 0.0,
            "final_cash_p50": 0.0,
            "final_cash_p95": 0.0,
        }

    rng = np.random.default_rng(seed)
    arr = np.array(trade_pnl_list, dtype=float)

    max_dds: list[float] = []
    final_cashes: list[float] = []
    for _ in range(n_simulations):
        # 復元抽出（ブートストラップ）で equity curve を生成
        shuffled = rng.choice(arr, size=len(arr), replace=True)
        equity = np.empty(len(shuffled) + 1)
        equity[0] = initial_cash
        equity[1:] = initial_cash + np.cumsum(shuffled)

        roll_max = np.maximum.accumulate(equity)
        dd = (equity - roll_max) / np.where(roll_max > 0, roll_max, 1.0)
        max_dds.append(float(dd.min()))
        final_cashes.append(float(equity[-1]))

    pct = int(confidence * 100)
    return {
        "max_drawdown_mean": float(np.mean(max_dds)),
        "max_drawdown_p95": float(np.percentile(max_dds, pct)),
        "final_cash_p05": float(np.percentile(final_cashes, 5)),
        "final_cash_p50": float(np.percentile(final_cashes, 50)),
        "final_cash_p95": float(np.percentile(final_cashes, pct)),
    }


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
