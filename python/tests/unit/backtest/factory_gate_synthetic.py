"""戦略ファクトリーのゲート分類テスト用: 正解ラベル付き合成戦略の生成器。

各銘柄について「保有している日だけ日次リターンが N(mu, sigma) で発生する」取引列を
乱数 seed から決定的に作り、バックテスタが返すのと同じ形（取引リターン列と日次
mark-to-market equity）で返す。

保有日のリターンは「市場ファクター × beta + 銘柄固有ノイズ」で、市場ファクターは
全銘柄で共通（候補とチャンピオンも同じ相場を経験する）。実際の株と同じく、保有銘柄を
増やしても市場ファクター分のリスクは分散されない。

正解（ground truth）は、生成パラメータから理論的に決まる「等金額ポートフォリオの
日次リターンの真の年率 Sharpe」である。運用者が実際に経験する損益曲線の質であり、
取引頻度の水増し（期待値ゼロの取引の追加）や相関を無視した分散効果では上がらない。
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import FACTORY_GATE_MIN_TRADES_PER_SYMBOL
from src.backtest.factory_aggregation import SymbolMetrics
from src.backtest.factory_evaluation import build_evaluation
from src.backtest.factory_gate import portfolio_dsr
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

TRADING_DAYS = 252
INITIAL_CASH = 1_000_000.0


@dataclass(frozen=True)
class Leg:
    """保有区間の生成規則。1 つの戦略は 1 本以上の Leg の和（OR 合成に相当）。

    trades_per_symbol: 銘柄あたり取引数の期待値（ポアソン）
    hold_days: 1 取引の保有日数
    mu: 保有日の日次リターン平均
    sigma: 保有日の日次リターン標準偏差
    """

    trades_per_symbol: float
    hold_days: int
    mu: float
    sigma: float


@dataclass(frozen=True)
class StrategySpec:
    legs: tuple[Leg, ...]
    n_symbols: int = 60
    n_days: int = 2 * TRADING_DAYS
    # 保有日リターンの市場ファクター感応度と、市場ファクターの日次標準偏差
    beta: float = 0.0
    market_sigma: float = 0.01


@dataclass
class SyntheticResult:
    """1 戦略を全銘柄で「バックテストした」結果。"""

    rows: list[SymbolMetrics] = field(default_factory=list)
    equity_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    span_years: float = 0.0


def _exposure(spec: StrategySpec, leg: Leg) -> float:
    """その Leg が保有している日の割合（期待値）。"""
    return min(leg.trades_per_symbol * leg.hold_days / spec.n_days, 1.0)


def true_portfolio_sharpe(spec: StrategySpec) -> float:
    """生成パラメータから決まる等金額ポートフォリオの真の年率 Sharpe。

    銘柄 s の日次リターンは x = pos * (mu + beta * m + e)。pos は保有中なら 1。
    Leg 同士は保有日が重ならないため、保有確率は各 Leg の保有率の和 p になる。
    ポートフォリオ日次リターン avg = (1/S) Σ_s x_s について、保有中の銘柄割合を q とすると
        E[avg]   = Σ p_i mu_i
        Var[avg] = Var(Σ q_i mu_i) + beta^2 σ_m^2 E[q^2] + (1/S) Σ p_i σ_i^2
    q_i は銘柄ごとに独立な保有の平均なので Var(q_i) ≈ p_i(1-p_i)/S。
    第 2 項は S を増やしても消えない（市場ファクターは分散できない）。
    """
    s_count = spec.n_symbols
    p_total = sum(_exposure(spec, leg) for leg in spec.legs)
    mean = sum(_exposure(spec, leg) * leg.mu for leg in spec.legs)
    # q_i（Leg ごとの保有割合）の共分散は、同じ銘柄で排他的なため -p_i p_j / S
    var_mean_part = 0.0
    for i, leg_i in enumerate(spec.legs):
        for j, leg_j in enumerate(spec.legs):
            p_i, p_j = _exposure(spec, leg_i), _exposure(spec, leg_j)
            cov = p_i * (1 - p_i) / s_count if i == j else -p_i * p_j / s_count
            var_mean_part += leg_i.mu * leg_j.mu * cov
    e_q2 = p_total**2 + p_total * (1 - p_total) / s_count
    var_market = spec.beta**2 * spec.market_sigma**2 * e_q2
    var_idio = sum(_exposure(spec, leg) * leg.sigma**2 for leg in spec.legs) / s_count
    var = var_mean_part + var_market + var_idio
    if var <= 0:
        return 0.0
    return math.sqrt(TRADING_DAYS) * mean / math.sqrt(var)


@functools.lru_cache(maxsize=None)
def _business_days(n_days: int) -> pd.DatetimeIndex:
    """評価期間の営業日。生成のたびに作ると 1 件あたり数ミリ秒かかるため使い回す。

    freq を持たない DatetimeIndex にする。バックテスタの equity も freq を持たず、
    freq 付きだと Series を作るたびに pandas が日付列を再検証して遅くなる。
    """
    return pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=n_days).to_numpy())


def market_path(n_days: int, market_sigma: float, seed: int) -> np.ndarray:
    """全銘柄・全戦略が共有する市場ファクターの日次リターン。"""
    return np.random.default_rng(seed).normal(0.0, market_sigma, size=n_days)


def simulate(spec: StrategySpec, seed: int, market: np.ndarray | None = None) -> SyntheticResult:
    """spec を seed で決定的に生成する。

    market を渡すとその市場ファクター系列を使う（候補とチャンピオンに同じ相場を
    経験させるため）。省略時は seed から生成する。

    Leg ごとに hold_days 刻みの区画から取引開始日を抽選し、既に他の Leg が保有して
    いる日とは重ならないようにする（OR 合成で同じ資金を二重に使わない）。
    """
    hold = spec.legs[0].hold_days
    if any(leg.hold_days != hold for leg in spec.legs):
        raise ValueError("全 Leg の hold_days は同一であること（区画を共有して重なりを防ぐ）")
    rng = np.random.default_rng(seed)
    dates = _business_days(spec.n_days)
    result = SyntheticResult(span_years=(dates[-1] - dates[0]).days / 365.25)
    n_slots = spec.n_days // hold
    n_sym = spec.n_symbols
    if market is None:
        market = market_path(spec.n_days, spec.market_sigma, seed + 10_000_019)

    # 全銘柄を一度に生成する。区画（hold 日の塊）を銘柄ごとにランダムな順位に並べ、
    # Leg ごとに順位の連続した範囲を割り当てる＝同じ銘柄で Leg 同士は重ならない。
    rank = np.argsort(rng.random((n_sym, n_slots)), axis=1).argsort(axis=1)
    slot_leg = np.full((n_sym, n_slots), -1)
    cursor = np.zeros(n_sym, dtype=int)
    for k, leg in enumerate(spec.legs):
        n_trades = np.minimum(rng.poisson(leg.trades_per_symbol, size=n_sym), n_slots - cursor)
        upper = cursor + n_trades
        slot_leg[(rank >= cursor[:, None]) & (rank < upper[:, None])] = k
        cursor = upper

    # 保有日リターン = Leg の mu + 銘柄固有ノイズ + beta × 市場ファクター（区画単位で展開）
    mus = np.array([leg.mu for leg in spec.legs])
    sigmas = np.array([leg.sigma for leg in spec.legs])
    held = slot_leg >= 0
    leg_of = np.where(held, slot_leg, 0)
    noise = rng.normal(size=(n_sym, n_slots, hold))
    slot_market = market[: n_slots * hold].reshape(n_slots, hold)[None, :, :]
    r = mus[leg_of][:, :, None] + sigmas[leg_of][:, :, None] * noise + spec.beta * slot_market
    r = np.where(held[:, :, None], r, 0.0)

    daily = np.zeros((n_sym, spec.n_days))
    daily[:, : n_slots * hold] = r.reshape(n_sym, n_slots * hold)
    equity = INITIAL_CASH * np.cumprod(1.0 + daily, axis=1)
    trade_ret = np.prod(1.0 + r, axis=2) - 1.0

    # 銘柄ごとの集計は先に配列でまとめて計算する（銘柄ループ内で小さな numpy 呼び出しを
    # 繰り返すと、それだけで 1 件あたり数ミリ秒かかる）
    n_trades_by_sym = held.sum(axis=1)
    wins_by_sym = ((trade_ret > 0) & held).sum(axis=1)
    total_return_by_sym = equity[:, -1] / INITIAL_CASH - 1.0
    symbols = [f"S{s:03d}" for s in range(n_sym)]
    # Series を銘柄ごとに作るより、DataFrame を 1 回作って列を取り出すほうが速い
    equity_frame = pd.DataFrame(equity.T, index=dates, columns=symbols)

    for s in range(n_sym):
        n = int(n_trades_by_sym[s])
        if n == 0:
            continue
        # 取引は時系列順に並べる（バックテスタの出力と同じ順序）
        trade_returns = trade_ret[s, held[s]].tolist()
        symbol = symbols[s]
        result.equity_by_symbol[symbol] = equity_frame[symbol]
        result.rows.append(
            SymbolMetrics(
                symbol=symbol,
                num_trades=n,
                sharpe_ratio=0.0,
                sharpe_per_trade=0.0,
                win_rate=float(wins_by_sym[s]) / n,
                total_return=float(total_return_by_sym[s]),
                max_drawdown=0.0,
                trade_returns=trade_returns,
            )
        )
    return result


def to_evaluation(result: SyntheticResult, n_trials: int, is_control: bool) -> FactoryEvaluation:
    """本番と同じ関数で FactoryEvaluation を組み立て、DSR まで設定して返す。

    組み立て（build_evaluation）と DSR（portfolio_dsr）は本番の関数そのものを呼ぶ。
    テスト側に写しを持たないことで、本番のゲートの「量」を検証していることを保証する。
    """
    evaluation = build_evaluation(
        FactoryHypothesis(
            rule_spec={"type": "atomic", "rule": "synthetic", "params": {}},
            market="jp",
            is_control=is_control,
        ),
        result.rows,
        result.equity_by_symbol,
        initial_cash=INITIAL_CASH,
        min_trades_per_symbol=FACTORY_GATE_MIN_TRADES_PER_SYMBOL,
        n_symbols=len(result.rows),
    )
    evaluation.dsr = portfolio_dsr(evaluation, n_trials)
    return evaluation


def daily_return_series(daily_mean: float, seed: int, n_days: int = 2 * TRADING_DAYS) -> pd.Series:
    """ゲートの単体テスト用: 平均 daily_mean・標準偏差 1% の日次リターン系列。

    チャンピオン比較を確実に通したい／落としたいテストで、候補とチャンピオンの
    portfolio_returns に使う（例: 候補 0.003・チャンピオン 0.0 なら z は十分大きい）。
    """
    values = np.random.default_rng(seed).normal(daily_mean, 0.01, size=n_days)
    return pd.Series(values, index=_business_days(n_days))


def champion_evaluation(daily_mean: float = 0.0, seed: int = 1) -> FactoryEvaluation:
    """ゲートの単体テスト用: 日次リターンだけを持つチャンピオン評価。"""
    returns = daily_return_series(daily_mean, seed)
    return FactoryEvaluation(
        hypothesis=FactoryHypothesis(
            rule_spec={"type": "atomic", "rule": "synthetic", "params": {}},
            market="jp",
            is_control=True,
        ),
        portfolio_sharpe_ratio=float(returns.mean() / returns.std() * math.sqrt(TRADING_DAYS)),
        portfolio_returns=returns,
    )
