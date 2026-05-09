"""
銘柄×ルール評価エンジン（週次ジョブ用）

全ウォッチリスト銘柄に対してルールをバックテストし、
判定基準を満たす「最優秀ルール」を DuckDB に保存する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.backtest.rule_backtester import run_rule_backtest
from src.backtest.rules import ALL_RULES
from src.utils.db.rule_results import load_rule_best_all, upsert_rule_best
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WATCHLIST_PATH = Path(__file__).parent.parent.parent / "config" / "watchlist.json"

# 有効ルールの判定閾値
_MIN_WIN_RATE = 0.50
_MIN_NET_PROFIT = 0.0


def _load_symbols(market: str) -> list[str]:
    with open(_WATCHLIST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(market, [])


def _select_best_rule(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    勝率50%以上 AND 純利益プラス の条件を満たすルールを選ぶ。
    複数該当する場合は win_rate → net_profit の優先順で選択。
    """
    valid = [
        r for r in results if r["win_rate"] >= _MIN_WIN_RATE and r["net_profit"] > _MIN_NET_PROFIT
    ]
    if not valid:
        return None
    return max(valid, key=lambda r: (r["win_rate"], r["net_profit"]))


def evaluate_all_symbols(
    market: str,
    backtest_start: str,
    backtest_end: str,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: float | None = 0.07,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """
    ウォッチリスト全銘柄をバックテストし、最優秀ルールを DB に保存する。

    Args:
        market: マーケット識別子
        backtest_start: バックテスト開始日
        backtest_end: バックテスト終了日
        initial_cash: 初期資金
        fee_rate: 片道手数料率
        slippage: スリッページ率
        stop_loss_pct: ストップロス率
        symbols: 評価対象銘柄リスト（Noneでウォッチリスト全銘柄）

    Returns:
        {"evaluated": int, "effective": int, "skipped": int}
    """
    target_symbols = symbols or _load_symbols(market)
    logger.info(f"ルール評価開始: {len(target_symbols)} 銘柄 ({market})")

    evaluated = 0
    effective = 0
    skipped = 0

    for i, symbol in enumerate(target_symbols, 1):
        logger.info(f"[{i}/{len(target_symbols)}] {market}/{symbol} 評価中...")
        try:
            results = run_rule_backtest(
                market=market,
                symbol=symbol,
                rules=ALL_RULES,
                start=backtest_start,
                end=backtest_end,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                stop_loss_pct=stop_loss_pct,
            )
            evaluated += 1

            best = _select_best_rule(results)
            if best is None:
                logger.info(f"  [{symbol}] 有効ルールなし → スキップ")
                skipped += 1
                continue

            upsert_rule_best(
                market=market,
                symbol=symbol,
                best_rule=best["rule"],
                win_rate=best["win_rate"],
                net_profit=best["net_profit"],
                num_trades=best["num_trades"],
                profit_factor=best.get("profit_factor"),
                max_drawdown=best.get("max_drawdown", 0.0),
                backtest_start=backtest_start,
                backtest_end=backtest_end,
            )
            effective += 1
            logger.info(
                f"  [{symbol}] 最優秀: {best['rule']}"
                f"  勝率={best['win_rate']:.1%}  純利益=JPY{best['net_profit']:+,.0f}"
            )

        except Exception as exc:
            logger.error(f"  [{symbol}] 評価失敗: {exc}", exc_info=True)
            skipped += 1

    summary = {"evaluated": evaluated, "effective": effective, "skipped": skipped}
    logger.info(f"ルール評価完了: 評価={evaluated}  有効={effective}  スキップ={skipped}")
    return summary


def print_rule_summary(market: str) -> None:
    """DB に保存された最優秀ルール一覧を表示する。"""
    df = load_rule_best_all(market)
    if df.empty:
        print("データなし（まだ評価が実行されていません）")
        return

    print(f"\n{'=' * 70}")
    print(f"  最優秀ルール一覧: {market}  ({len(df)} 銘柄)")
    print(f"{'=' * 70}")
    print(f"{'銘柄':>6}  {'ルール':<22}  {'勝率':>6}  {'純利益(円)':>12}  {'取引数':>6}")
    print("-" * 60)
    for _, row in df.iterrows():
        profit_str = f"JPY{row['net_profit']:>+,.0f}"
        print(
            f"{row['symbol']:>6}  {row['best_rule']:<22}  {row['win_rate']:>5.1%}"
            f"  {profit_str:>12}  {row['num_trades']:>6}"
        )
    print(f"{'=' * 70}")
