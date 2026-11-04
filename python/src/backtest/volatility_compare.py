"""値動きに合わせた量と損切りの線の比べ（#1526 の手順4・#1547）。

今の固定の率と量（fixed）、量の補正だけ（size）、損切りの線だけ（stop）、両方（both）の4つを、
手数料・動的スリッページ込みで、同じ銘柄・同じ期間・同じシグナルでシミュレーションする。
stop と both は k を変えて比べる。fixed と size は k に関わらないので1つずつにする。

この比べは材料を集めるだけで、発注は変えない。数字で発注を変えるかは結果を読んだ別の Issue で決める。
成績で k や目標の値動きを選んでいないので、ここの定数を成績に合わせて動かさない。

銘柄ごとの値は等金額で合成し、運用者が経験する曲線で比べる（factory_portfolio.py と同じ考え）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional, Sequence

import pandas as pd

from config.settings import DEFAULT_STOP_LOSS_PCT
from src.backtest.backtester import Backtester
from src.backtest.execution import default_slippage_for
from src.backtest.factory_portfolio import build_portfolio_equity
from src.backtest.metrics import _max_drawdown
from src.backtest.pipeline.runner import PreparedBacktest, prepare_backtest_signals
from src.backtest.slippage import make_slippage_fn
from src.backtest.volatility_overlay import lagged_forecast
from src.domain.types import SymbolTask
from src.domain.volatility_sizing import DEFAULT_VOLATILITY_SIZING, VolatilitySizingParams
from src.utils.logger import get_logger

logger = get_logger(__name__)

COMPARE_KS: tuple[float, ...] = (2.0, 3.0, 4.0)
# 1市場の銘柄の上限。週のジョブが長くなりすぎないよう、watchlist の先頭から取る
MAX_SYMBOLS_PER_MARKET = 100
COMPARE_LOOKBACK_YEARS = 5
COMPARE_FEE_RATE = 0.001
# 量を増やす余地を残すため全額でなく半分を基準にする。元手は、半分の量が単元株に潰れない大きさにする
COMPARE_POSITION_FRACTION = 0.5
COMPARE_INITIAL_CASH = 10_000_000.0
_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CompareForm:
    """比べる形の1つ。k は stop と both だけが持つ。"""

    name: str
    k: Optional[float] = None

    def _params(self) -> VolatilitySizingParams:
        return VolatilitySizingParams(
            enabled=True,
            stop_volatility_multiple=(
                self.k if self.k is not None else DEFAULT_VOLATILITY_SIZING.stop_volatility_multiple
            ),
        )

    @property
    def size_params(self) -> Optional[VolatilitySizingParams]:
        return self._params() if self.name in ("size", "both") else None

    @property
    def stop_params(self) -> Optional[VolatilitySizingParams]:
        return self._params() if self.name in ("stop", "both") else None


def compare_forms(ks: Sequence[float] = COMPARE_KS) -> list[CompareForm]:
    """比べる形の一覧を返す。"""
    forms = [CompareForm("fixed"), CompareForm("size")]
    forms += [CompareForm("stop", k) for k in ks]
    forms += [CompareForm("both", k) for k in ks]
    return forms


@dataclass(frozen=True)
class _SymbolRun:
    equity: pd.Series
    num_trades: int
    stop_outs: int


def _simulate(
    market: str, symbol: str, prepared: PreparedBacktest, form: CompareForm
) -> _SymbolRun:
    backtester = Backtester(
        model_manager=prepared.model_manager,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market=market,
        symbol=symbol,
        initial_cash=COMPARE_INITIAL_CASH,
        fee_rate=COMPARE_FEE_RATE,
        slippage=default_slippage_for(market),
        slippage_fn=make_slippage_fn(),
        stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
        position_sizing="fixed",
        position_fraction=COMPARE_POSITION_FRACTION,
        volatility_size_params=form.size_params,
        volatility_stop_params=form.stop_params,
    )
    trades, metrics = backtester.simulate_trading(
        prepared.sim_df,
        prepared.signal,
        pred=prepared.pred,
        collect_equity=True,
        volatility_closes=prepared.closes,
    )
    stop_outs = int((trades["action"] == "stop_loss").sum()) if not trades.empty else 0
    return _SymbolRun(metrics["equity_curve"], int(metrics.get("num_trades", 0) or 0), stop_outs)


def _lacks_forecast(prepared: PreparedBacktest) -> bool:
    """既定の扱いに倒れる足がある銘柄かを返す。"""
    forecast = lagged_forecast(prepared.closes, prepared.sim_df.index, DEFAULT_VOLATILITY_SIZING)
    return bool(forecast.isna().any())


def _return_over_volatility(equity: pd.Series) -> Optional[float]:
    """日次リターンの年率の平均 ÷ 年率の値動き。計算できなければ None。"""
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return None
    volatility = float(returns.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0.0:
        return None
    ratio = float(returns.mean()) / volatility * math.sqrt(_TRADING_DAYS_PER_YEAR)
    return round(ratio, 6)


def _form_summary(form: CompareForm, runs: list[_SymbolRun]) -> dict[str, Any]:
    equity = build_portfolio_equity([r.equity for r in runs], COMPARE_INITIAL_CASH)
    return {
        "form": form.name,
        "k": form.k,
        "symbols": len(runs),
        "num_trades": sum(r.num_trades for r in runs),
        "stop_outs": sum(r.stop_outs for r in runs),
        "max_drawdown": None if equity.empty else round(float(_max_drawdown(equity)), 6),
        "total_return": None if equity.empty else round(float(equity.iloc[-1]) - 1.0, 6),
        "return_over_volatility": None if equity.empty else _return_over_volatility(equity),
    }


def _compare_market(
    market: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    forms: list[CompareForm],
) -> dict[str, Any]:
    targets = symbols[:MAX_SYMBOLS_PER_MARKET]
    runs: dict[CompareForm, list[_SymbolRun]] = {form: [] for form in forms}
    failed = 0
    lacking_forecast = 0
    starts: list[Any] = []
    ends: list[Any] = []
    for symbol in targets:
        try:
            prepared = prepare_backtest_signals(
                market, symbol, start_date=start_date, end_date=end_date
            )
            symbol_runs = {form: _simulate(market, symbol, prepared, form) for form in forms}
        except Exception:
            # 1銘柄の失敗で週の比べを止めない。数は結果に残す
            logger.warning("比べの失敗: %s/%s", market, symbol, exc_info=True)
            failed += 1
            continue
        for form, run in symbol_runs.items():
            runs[form].append(run)
        lacking_forecast += int(_lacks_forecast(prepared))
        starts.append(prepared.sim_df.index[0])
        ends.append(prepared.sim_df.index[-1])
    return {
        "symbols_target": len(targets),
        "symbols_over_limit": len(symbols) - len(targets),
        "symbols": len(targets) - failed,
        "symbols_failed": failed,
        "symbols_lacking_forecast": lacking_forecast,
        "period": {
            "start": str(min(starts))[:10] if starts else None,
            "end": str(max(ends))[:10] if ends else None,
        },
        "forms": [_form_summary(form, runs[form]) for form in forms],
    }


def run_volatility_sizing_compare(
    tasks: Sequence[SymbolTask], start_date: str, end_date: str
) -> dict[str, Any]:
    """市場ごとに4つの形と3つの k を比べた結果を返す。

    Raises:
        RuntimeError: どの市場でも1銘柄も比べられなかったとき。
    """
    by_market: dict[str, list[str]] = {}
    for task in tasks:
        by_market.setdefault(task.market, []).append(task.symbol)

    forms = compare_forms()
    markets = {
        market: _compare_market(market, symbols, start_date, end_date, forms)
        for market, symbols in by_market.items()
    }
    if not any(result["symbols"] for result in markets.values()):
        raise RuntimeError(f"比べられた銘柄が0件: 対象 {len(tasks)} 銘柄")
    return {
        "start_date": start_date,
        "end_date": end_date,
        "settings": {
            "fixed_stop_loss_pct": DEFAULT_STOP_LOSS_PCT,
            "position_fraction": COMPARE_POSITION_FRACTION,
            "initial_cash": COMPARE_INITIAL_CASH,
            "fee_rate": COMPARE_FEE_RATE,
            "dynamic_slippage": True,
            "max_symbols_per_market": MAX_SYMBOLS_PER_MARKET,
            "ks": list(COMPARE_KS),
        },
        "markets": markets,
    }


def write_compare_result(result: dict[str, Any], as_of: date, output_dir: str) -> str:
    """結果を output_dir/YYYY-Www.json に書き、そのパスを返す。週は as_of の ISO 週。"""
    iso = as_of.isocalendar()
    week = f"{iso.year}-W{iso.week:02d}"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{week}.json")
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(
            {"week": week, "as_of": as_of.isoformat(), **result}, f, ensure_ascii=False, indent=2
        )
    os.replace(temporary, path)
    logger.info("値動きの量と損切りの線の比べを書いた: %s", path)
    return path
