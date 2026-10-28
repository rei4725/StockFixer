"""ポートフォリオシミュレーション（リバランス・ウェイト計算・セクター制約）。

Issue #511: 肥大化した portfolio.py を責務分割。
日次シミュレーション本体と、それが利用するウェイト/セクター/リバランス
ヘルパーを集約する。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from config.settings import MAX_SECTOR_POSITIONS
from src.backtest.data_port import get_backtest_data_port
from src.backtest.execution import ExecutionModel
from src.domain.regime_weights import get_regime_sector_weight
from src.domain.sector_cap import filter_by_sector_cap
from src.utils.logger import get_logger
from src.utils.sector_constraints import get_symbol_sector

logger = get_logger(__name__)

# 動的スリッページの平均出来高の窓（backtester.py の _AVG_VOL_WINDOW と同じ）
_AVG_VOL_WINDOW = 20


def _build_exec_avg_volume(volume_matrix: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """約定の日に使う平均出来高の表を返す（#1299）。

    約定の日より前の足までの 20 本の平均にし、約定の日の出来高を含めない
    （backtester.py と同じ rolling(20).mean().shift(1)、PR 783）。先頭の日は 0（フラットのみ）。
    """
    if volume_matrix is None or volume_matrix.empty:
        return None
    avg_vol = volume_matrix.sort_index().rolling(_AVG_VOL_WINDOW, min_periods=1).mean()
    return avg_vol.shift(1).fillna(0.0)


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[Any]:
    """リバランス日のリストを返す。"""
    if freq == "daily":
        return list(index)
    elif freq == "weekly":
        # 各週の最初の取引日（月曜相当）
        week_groups = pd.Series(index, index=index).groupby(pd.Grouper(freq="W-MON"))
        return [g.iloc[0] for _, g in week_groups if len(g) > 0]
    elif freq == "monthly":
        month_groups = pd.Series(index, index=index).groupby(pd.Grouper(freq="MS"))
        return [g.iloc[0] for _, g in month_groups if len(g) > 0]
    else:
        raise ValueError(f"未対応のリバランス頻度: {freq}")


def _softmax_weights(scores: pd.Series) -> pd.Series:
    """正のスコアのみを対象に softmax でウェイトを計算する。"""
    valid = scores.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return pd.Series(dtype=float)
    exp_s = np.exp(valid - valid.max())  # 数値安定化
    return exp_s / exp_s.sum()


def _apply_sector_rotation(scores: pd.Series, regime: str) -> pd.Series:
    """レジームに応じたセクターウェイト乗数をスコア Series に適用して返す。"""
    result = scores.copy()
    for sym in result.index:
        sector = _get_portfolio_symbol_sector(str(sym))
        result[sym] *= get_regime_sector_weight(regime, sector)
    return result


def _simulate_portfolio(
    score_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    rebalance_dates: list[Any],
    top_n: int,
    initial_cash: float,
    execution: ExecutionModel,
    max_sector_positions: int,
    use_sector_rotation: bool = False,
    execution_by_market: Optional[dict[str, ExecutionModel]] = None,
    open_matrix: Optional[pd.DataFrame] = None,
    volume_matrix: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    ポートフォリオシミュレーションを実行する。

    毎リバランス日 t のスコアで Top-N 銘柄とスコア比例のウェイトを決め、
    翌営業日 t+1 の始値（open_matrix）で売買する（#1099）。最後の日のスコアは約定させない。
    日次の portfolio_value と equal_weight_value を終値で記録する。

    execution_by_market を与えると、銘柄の鍵の市場（jp_・us_）ごとにその
    ExecutionModel で売買のコストを計算する（#847）。無い市場は execution を使う。

    volume_matrix を与えると、売りと買い（戦略と等分の基準の両方）に約定の日より前の
    足までの平均出来高を渡し、ExecutionModel の動的スリッページを効かせる（#1299）。
    """

    def _exec(sym: str) -> ExecutionModel:
        if not execution_by_market:
            return execution
        return execution_by_market.get(_split_symbol_key(sym)[0], execution)

    # セクターローテーション用に市場レジームを事前計算
    regime_series: Optional[pd.Series] = None
    if use_sector_rotation:
        proxy_df = _build_market_proxy_frame(close_matrix)
        if not proxy_df.empty:
            regime_series = get_backtest_data_port().get_market_regime(proxy_df)

    rebalance_set = set(str(d)[:10] for d in rebalance_dates)

    cash = initial_cash
    # {symbol: {"qty": int, "price": float}}
    holdings: dict[str, dict[str, Any]] = {}
    prev_symbols: set[str] = set()
    holdings_records: list[dict[str, Any]] = []

    equity_rows: list[dict[str, Any]] = []

    # 等分戦略の基準として全銘柄を均等保有したシリーズ（簡易近似）
    eq_cash = initial_cash
    eq_holdings: dict[str, dict[str, Any]] = {}

    # Issue #846: 評価と売却は、その日までの最後の値段（過去の方向にだけ埋めた値）で行う。
    # 購入はその日に値段のある銘柄だけにする。
    last_close_matrix = close_matrix.sort_index().ffill()
    # Issue #1099: 日 t のスコアは t の引けの後にしか出せないため、売買は翌営業日 t+1 の
    # 始値で約定させる（backtester の execution_lag=1 と同じ）。Open の表が無いときは終値の表で代える。
    fill_matrix = open_matrix if open_matrix is not None else close_matrix
    exec_avg_vol = _build_exec_avg_volume(volume_matrix)

    # 前のリバランスの日に決めて、次の日の寄りで約定させる注文
    pending: Optional[dict[str, Any]] = None
    prev_last_prices = pd.Series(dtype=float)
    dates = list(score_matrix.index)

    for i, date in enumerate(dates):
        date_str = str(date)[:10]
        last_prices = last_close_matrix.loc[date].dropna()

        # ─ 前の日に決めた売買を、この日の始値で約定 ─
        if pending is not None:
            fill_prices = (
                fill_matrix.loc[date].dropna()
                if date in fill_matrix.index
                else pd.Series(dtype=float)
            )
            # 始値の無い保有は、前の日までの最後の値段で売る（#846）
            sell_prices = prev_last_prices.copy()
            sell_prices.update(fill_prices)
            volumes = (
                exec_avg_vol.loc[date]
                if exec_avg_vol is not None and date in exec_avg_vol.index
                else pd.Series(dtype=float)
            )
            cash, holdings, records = _fill_rebalance(
                cash, holdings, pending, fill_prices, sell_prices, date_str, _exec, volumes
            )
            holdings_records.extend(records)
            eq_syms = [s for s in pending["eq_symbols"] if s in fill_prices.index]
            if eq_syms:
                eq_cash, eq_holdings = _fill_equal_weight(
                    eq_cash, eq_holdings, eq_syms, fill_prices, sell_prices, _exec, volumes
                )
            pending = None

        # ─ リバランス（最後の日のスコアは約定させない） ─
        if date_str in rebalance_set and i + 1 < len(dates):
            scores_today = score_matrix.loc[date].dropna()

            # セクターローテーション: レジームに応じてスコアを調整
            if use_sector_rotation and regime_series is not None:
                date_ts = pd.Timestamp(date_str)
                current_regime = "range"
                if date_ts >= regime_series.index.min():
                    raw = regime_series.asof(date_ts)
                    if not pd.isna(raw):
                        current_regime = str(raw)
                scores_today = _apply_sector_rotation(scores_today, current_regime)

            # 上位 top_n を選択
            top_candidates = _limit_portfolio_candidates_by_sector(
                scores_today.nlargest(top_n * 3),
                max_sector_positions=max_sector_positions,
            ).head(top_n)
            weights = _softmax_weights(top_candidates)

            if not weights.empty:
                new_symbols = set(weights.index)

                # ターンオーバー計算
                if prev_symbols:
                    stayed = prev_symbols & new_symbols
                    turnover = 1.0 - len(stayed) / max(len(prev_symbols), len(new_symbols))
                else:
                    turnover = 1.0

                pending = {
                    "rebalance_date": date_str,
                    "weights": weights,
                    "scores": top_candidates,
                    "turnover": turnover,
                    "eq_symbols": [str(s) for s in scores_today.index],
                }
                prev_symbols = new_symbols

        prev_last_prices = last_prices

        # ─ 日次ポートフォリオ価値 ─
        pf_value = cash + sum(
            pos["qty"] * last_prices.get(sym, pos.get("price", 0)) for sym, pos in holdings.items()
        )
        eq_value = eq_cash + sum(
            pos["qty"] * last_prices.get(sym, 0) for sym, pos in eq_holdings.items()
        )

        equity_rows.append(
            {
                "date": date_str,
                "portfolio_value": round(pf_value, 2),
                "equal_weight_value": round(eq_value, 2),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, holdings_records


def _fill_rebalance(
    cash: float,
    holdings: dict[str, dict[str, Any]],
    pending: dict[str, Any],
    fill_prices: pd.Series,
    sell_prices: pd.Series,
    fill_date: str,
    exec_for: Callable[[str], ExecutionModel],
    volumes: Optional[pd.Series] = None,
) -> tuple[float, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """既存保有を全売却し、決めたウェイトで約定の日の始値で買う（#1099）。

    volumes は約定の日より前の足までの平均出来高（#1299）。無い銘柄はフラットの率だけ。
    """
    for sym, pos in holdings.items():
        if sym in sell_prices.index:
            cash += exec_for(sym).sell_proceeds(
                pos["qty"], sell_prices[sym], _volume_of(volumes, sym)
            )

    new_holdings: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    total_budget = cash
    for sym_key, w in pending["weights"].items():
        sym = str(sym_key)
        if sym not in fill_prices.index:
            continue
        budget = total_budget * float(w)
        price = fill_prices[sym]
        if price <= 0:
            continue
        volume = _volume_of(volumes, sym)
        qty = exec_for(sym).max_affordable_qty(budget, price, volume)
        if qty > 0:
            cash -= exec_for(sym).buy_cost(qty, price, volume)
            new_holdings[sym] = {"qty": qty, "price": price}
            records.append(
                {
                    "rebalance_date": pending["rebalance_date"],
                    "fill_date": fill_date,
                    "symbol": sym,
                    "sector": _get_portfolio_symbol_sector(sym),
                    "weight": float(w),
                    "score": float(pending["scores"].get(sym, 0)),
                    "price": price,
                    "qty": qty,
                    "turnover": pending["turnover"],
                }
            )
    return cash, new_holdings, records


def _fill_equal_weight(
    eq_cash: float,
    eq_holdings: dict[str, dict[str, Any]],
    valid_syms: list[str],
    fill_prices: pd.Series,
    sell_prices: pd.Series,
    exec_for: Callable[[str], ExecutionModel],
    volumes: Optional[pd.Series] = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """等分戦略のリバランス（約定の日に始値のある銘柄に均等配分、#1099）。

    戦略と同じ平均出来高（volumes）で動的スリッページを当てる（#1299）。
    """
    for sym, pos in eq_holdings.items():
        if sym in sell_prices.index:
            eq_cash += exec_for(sym).sell_proceeds(
                pos["qty"], sell_prices[sym], _volume_of(volumes, sym)
            )
    new_holdings: dict[str, dict[str, Any]] = {}
    eq_per = eq_cash / len(valid_syms)
    for sym in valid_syms:
        price = fill_prices[sym]
        if price <= 0:
            continue
        volume = _volume_of(volumes, sym)
        qty = exec_for(sym).max_affordable_qty(eq_per, price, volume)
        if qty > 0:
            eq_cash -= exec_for(sym).buy_cost(qty, price, volume)
            new_holdings[sym] = {"qty": qty}
    return eq_cash, new_holdings


def _volume_of(volumes: Optional[pd.Series], sym: str) -> float:
    """銘柄の平均出来高を返す。無い・NaN のときは 0（フラットの率だけ）。"""
    if volumes is None or sym not in volumes.index:
        return 0.0
    value = float(volumes[sym])
    return value if value > 0 else 0.0


def _get_portfolio_symbol_sector(sym_key: str) -> str:
    market, symbol = _split_symbol_key(sym_key)
    return get_symbol_sector(market, symbol)


def _split_symbol_key(sym_key: str) -> tuple[str, str]:
    if "_" not in sym_key:
        return "jp", sym_key
    return tuple(sym_key.split("_", 1))  # type: ignore[return-value]


def _limit_portfolio_candidates_by_sector(
    top_candidates: pd.Series,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
) -> pd.Series:
    if top_candidates.empty or max_sector_positions <= 0:
        return top_candidates.copy()

    ordered_items = list(top_candidates.items())
    selected_items = filter_by_sector_cap(
        ordered_items,
        max_sector_positions=max_sector_positions,
        sector_getter=lambda item: _get_portfolio_symbol_sector(str(item[0])),
    )
    if not selected_items:
        return top_candidates.iloc[0:0].copy()
    return pd.Series(
        data=[float(score) for _, score in selected_items],
        index=[str(symbol) for symbol, _ in selected_items],
        dtype=float,
    )


def _build_market_proxy_frame(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """複数銘柄の終値行列から市場全体の簡易プロキシ価格を構築する。"""
    if close_matrix is None or close_matrix.empty:
        return pd.DataFrame()

    aligned = close_matrix.sort_index().ffill()
    proxy_df = pd.DataFrame(index=aligned.index)
    proxy_df["Close"] = aligned.mean(axis=1, skipna=True)
    proxy_df["High"] = aligned.max(axis=1, skipna=True)
    proxy_df["Low"] = aligned.min(axis=1, skipna=True)
    return proxy_df.dropna(subset=["Close"])
