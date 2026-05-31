"""
ルールベースバックテスター

TradingRule のリストを受け取り、銘柄ごと・ルールごとにバックテストを実行して
勝率・総利益でランキングした結果を返す。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

import src.market_data.yf_client as yf_client
from src.backtest.backtester import Backtester
from src.backtest.rules.base import TradingRule
from src.market_data.technical import add_technical_indicators
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_backtester(
    initial_cash: float,
    fee_rate: float,
    slippage: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> Backtester:
    return Backtester(
        model_manager=None,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market="",
        symbol="",
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing="full",
    )


def run_rule_backtest(
    market: str,
    symbol: str,
    rules: list[TradingRule],
    start: str,
    end: str,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: float | None = 0.07,
    take_profit_pct: float | None = None,
) -> list[dict[str, Any]]:
    """
    指定銘柄に対して複数ルールのバックテストを実行する。

    Args:
        market: マーケット識別子（"jp" or "us"）
        symbol: 銘柄コード
        rules: 検証するルールのリスト
        start: バックテスト開始日（"YYYY-MM-DD"）
        end: バックテスト終了日（"YYYY-MM-DD"）
        initial_cash: 初期資金
        fee_rate: 片道手数料率
        slippage: スリッページ率
        stop_loss_pct: ストップロス率（Noneで無効）
        take_profit_pct: テイクプロフィット率（Noneで無効）

    Returns:
        各ルールのバックテスト結果リスト（win_rate 降順）
    """
    ticker = get_ticker(market, symbol)
    logger.info(f"データ取得: {ticker} ({start} → {end})")

    df = yf_client.download(ticker, start=start, end=end)
    if df is None or df.empty:
        logger.error(f"データ取得失敗: {ticker}")
        return []

    df = add_technical_indicators(df)
    df = df.dropna(subset=["Close"])

    logger.info(f"データ取得完了: {len(df)} 行")

    backtester = _make_backtester(initial_cash, fee_rate, slippage, stop_loss_pct, take_profit_pct)
    results = []

    for rule in rules:
        try:
            signal = rule.generate_signal(df)
            buy_count = int((signal == 1).sum())
            sell_count = int((signal == -1).sum())
            logger.info(f"  [{rule.name}] Buy={buy_count}, Sell={sell_count}")

            if buy_count == 0:
                logger.warning(f"  [{rule.name}] Buyシグナルなし → スキップ")
                results.append(_empty_result(market, symbol, rule))
                continue

            _, metrics = backtester.simulate_trading(df, signal)

            net_profit = metrics.get("final_cash", initial_cash) - initial_cash
            results.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "rule": rule.name,
                    "description": rule.description,
                    "win_rate": metrics.get("win_rate", 0.0),
                    "net_profit": round(net_profit, 0),
                    "total_return": metrics.get("total_return", 0.0),
                    "num_trades": metrics.get("num_trades", 0),
                    "profit_factor": metrics.get("profit_factor") or 0.0,
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "avg_win": metrics.get("avg_win", 0.0),
                    "avg_loss": metrics.get("avg_loss", 0.0),
                    "buy_signals": buy_count,
                }
            )
        except Exception as exc:
            logger.error(f"  [{rule.name}] バックテスト失敗: {exc}", exc_info=True)
            results.append(_empty_result(market, symbol, rule))

    results.sort(key=lambda r: (r["win_rate"], r["net_profit"]), reverse=True)
    return results


def run_multi_symbol_rule_backtest(
    market: str,
    symbols: list[str],
    rules: list[TradingRule],
    start: str,
    end: str,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: float | None = 0.07,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """
    複数銘柄 × 複数ルールのバックテストを実行し、結果を DataFrame で返す。

    Returns:
        各 (symbol, rule) の結果を行とする DataFrame（win_rate × net_profit 降順）
    """
    all_rows = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {market}/{symbol}")
        rows = run_rule_backtest(
            market=market,
            symbol=symbol,
            rules=rules,
            start=start,
            end=end,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            slippage=slippage,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["win_rate", "net_profit"], ascending=[False, False])
    return df


def print_rule_ranking(
    results: list[dict[str, Any]] | pd.DataFrame,
    market: str,
    symbol: str,
    initial_cash: float = 1_000_000,
) -> None:
    """バックテスト結果を勝率・総利益でランキング表示する。"""
    rows: list[dict[str, Any]]
    if isinstance(results, pd.DataFrame):
        rows = results.to_dict("records")  # type: ignore[assignment]
    else:
        rows = results

    header = f" 法則ランキング {market}/{symbol} "
    print("=" * 70)
    print(header.center(70))
    print("=" * 70)
    print(
        f"{'順位':<4} {'法則名':<24} {'勝率':>6} {'総利益(円)':>12} "
        f"{'取引数':>6} {'PF':>6} {'MDD':>8}"
    )
    print("-" * 70)

    for rank, row in enumerate(rows, 1):
        profit_str = f"JPY {row['net_profit']:>+,.0f}"
        mdd_str = f"{row['max_drawdown']:.1%}"
        pf = row["profit_factor"]
        pf_str = f"{pf:.2f}" if pf and not math.isinf(pf) else "∞" if pf else "-"
        print(
            f"{rank:<4} {row['rule']:<24} {row['win_rate']:>5.1%} "
            f"{profit_str:>12} {row['num_trades']:>6} {pf_str:>6} {mdd_str:>8}"
        )

    print("=" * 70)


def _empty_result(market: str, symbol: str, rule: TradingRule) -> dict[str, Any]:
    return {
        "market": market,
        "symbol": symbol,
        "rule": rule.name,
        "description": rule.description,
        "win_rate": 0.0,
        "net_profit": 0.0,
        "total_return": 0.0,
        "num_trades": 0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "buy_signals": 0,
    }
