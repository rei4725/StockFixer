"""
戦略ファクトリー Phase 1（#369）

夜間にルール組合せ仮説をサンプリングし、窓分割バックテストで評価して
過学習ゲート（DSR / PBO / 取引数 / DD / 対チャンピオン改善）を通った仮説のみ
results/factory/reports/ へ不変 JSON レポートを出力する。

GitHub への Issue 起票は IssueAgent 側（--factory-intake）の責務であり、
本モジュールは GitHub トークンを一切扱わない。
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import (
    FACTORY_GATE_CHAMPION_MARGIN,
    FACTORY_GATE_MAX_DRAWDOWN,
    FACTORY_GATE_MAX_PBO,
    FACTORY_GATE_MIN_DSR,
    FACTORY_GATE_MIN_TRADES,
)
from src.backtest.backtester import Backtester
from src.backtest.data_port import get_backtest_data_port
from src.backtest.factory_report import write_report
from src.backtest.hypothesis_review import review_hypothesis
from src.backtest.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from src.backtest.rules import (
    AndRule,
    BollingerBandRule,
    EMAMomentumRule,
    MACDRSIRule,
    OrRule,
    RSIContrarianRule,
    TradingRule,
    VolatilityBreakoutRule,
    VolumeBreakoutRule,
)
from src.backtest.types import FactoryBatchResult, FactoryEvaluation, FactoryHypothesis
from src.utils.data_path_utils import get_ticker
from src.utils.db import count_factory_runs, load_factory_hashes, save_factory_run
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 探索空間
# ---------------------------------------------------------------------------

_RULE_CLASSES: dict[str, type] = {
    "volume_breakout": VolumeBreakoutRule,
    "ema_momentum": EMAMomentumRule,
    "rsi_contrarian": RSIContrarianRule,
    "bollinger_band": BollingerBandRule,
    "macd_rsi": MACDRSIRule,
    "volatility_breakout": VolatilityBreakoutRule,
}

# 各ルールのパラメータ候補（先頭がデフォルト相当 = 対照に使用）
_PARAM_GRID: dict[str, list[dict[str, Any]]] = {
    "volume_breakout": [
        {"volume_ratio": 2.0, "breakout_window": 20},
        {"volume_ratio": 1.5, "breakout_window": 20},
        {"volume_ratio": 3.0, "breakout_window": 20},
        {"volume_ratio": 2.0, "breakout_window": 10},
        {"volume_ratio": 2.0, "breakout_window": 40},
    ],
    "ema_momentum": [
        {"fast_window": 12, "slow_window": 26},
        {"fast_window": 8, "slow_window": 21},
        {"fast_window": 20, "slow_window": 50},
    ],
    "rsi_contrarian": [
        {"oversold": 30.0, "overbought": 70.0},
        {"oversold": 25.0, "overbought": 75.0},
        {"oversold": 20.0, "overbought": 80.0},
    ],
    "bollinger_band": [
        {"sell_at_upper": False},
        {"sell_at_upper": True},
    ],
    "macd_rsi": [
        {"rsi_filter": 60.0},
        {"rsi_filter": 50.0},
        {"rsi_filter": 70.0},
    ],
    "volatility_breakout": [
        {"buy_k": 1.0, "sell_k": 1.5},
        {"buy_k": 0.8, "sell_k": 1.2},
        {"buy_k": 1.5, "sell_k": 2.0},
    ],
}

# ゲート閾値は config/settings.py（FACTORY_GATE_*）で定義し env 上書き可能

_MIN_SYMBOL_ROWS = 100


# ---------------------------------------------------------------------------
# ルール構築
# ---------------------------------------------------------------------------


def build_rule(spec: dict) -> TradingRule:
    """rule_spec（再帰構造）から TradingRule インスタンスを構築する。"""
    spec_type = spec.get("type")
    if spec_type == "atomic":
        rule_name = spec["rule"]
        if rule_name not in _RULE_CLASSES:
            raise ValueError(f"未知のルール名: {rule_name}")
        return _RULE_CLASSES[rule_name](**(spec.get("params") or {}))
    if spec_type in ("and", "or"):
        children = [build_rule(s) for s in spec.get("rules", [])]
        if len(children) < 2:
            raise ValueError(f"合成ルールには2つ以上の子が必要: {spec}")
        return AndRule(children) if spec_type == "and" else OrRule(children)
    raise ValueError(f"未知の spec type: {spec_type}")


# ---------------------------------------------------------------------------
# サンプラー
# ---------------------------------------------------------------------------


def _sample_atomic(rng: random.Random) -> dict:
    rule_name = rng.choice(sorted(_RULE_CLASSES))
    params = rng.choice(_PARAM_GRID[rule_name])
    return {"type": "atomic", "rule": rule_name, "params": dict(params)}


def sample_hypotheses(
    market: str,
    budget: int,
    existing_hashes: set[str],
    seed: Optional[int] = None,
    lookback_years: int = 2,
) -> list[FactoryHypothesis]:
    """評価済みハッシュと重複しない仮説を budget 本サンプリングする。

    構成比: 原子 40% / AND合成 40% / OR合成 20%（子は2〜3個、ネストなし）。
    探索空間が枯渇した場合は budget 未満で打ち切る。
    """
    rng = random.Random(seed)
    sampled: list[FactoryHypothesis] = []
    seen = set(existing_hashes)
    max_attempts = budget * 50

    for _ in range(max_attempts):
        if len(sampled) >= budget:
            break
        roll = rng.random()
        if roll < 0.4:
            spec: dict = _sample_atomic(rng)
        else:
            n_children = rng.choice([2, 2, 3])
            children = []
            used_rules: set[str] = set()
            for _ in range(n_children):
                child = _sample_atomic(rng)
                if child["rule"] in used_rules:
                    continue  # 同一ルールの重複合成は意味が薄いため避ける
                used_rules.add(child["rule"])
                children.append(child)
            if len(children) < 2:
                continue
            spec = {"type": "and" if roll < 0.8 else "or", "rules": children}

        hypothesis = FactoryHypothesis(rule_spec=spec, market=market, lookback_years=lookback_years)
        if hypothesis.hypothesis_hash in seen:
            continue
        seen.add(hypothesis.hypothesis_hash)
        sampled.append(hypothesis)

    if len(sampled) < budget:
        logger.warning(
            "サンプリング打ち切り: %d/%d 本（探索空間の枯渇または重複多数）",
            len(sampled),
            budget,
        )
    return sampled


def control_hypotheses(market: str, lookback_years: int = 2) -> list[FactoryHypothesis]:
    """対照群: デフォルトパラメータの原子ルール6種（現行ベースライン）。"""
    controls = []
    for rule_name in sorted(_RULE_CLASSES):
        spec = {
            "type": "atomic",
            "rule": rule_name,
            "params": dict(_PARAM_GRID[rule_name][0]),
        }
        controls.append(
            FactoryHypothesis(
                rule_spec=spec,
                market=market,
                lookback_years=lookback_years,
                is_control=True,
            )
        )
    return controls


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------


def _load_symbol_data(
    market: str, symbols: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """銘柄ごとに OHLCV + テクニカル指標を1回だけ取得する。"""
    port = get_backtest_data_port()
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = port.download(get_ticker(market, symbol), start=start, end=end)
            if df is None or len(df) < _MIN_SYMBOL_ROWS:
                logger.warning("データ不足のためスキップ: %s/%s", market, symbol)
                continue
            df = port.add_technical_indicators(df)
            data[symbol] = df.dropna(subset=["Close"])
        except Exception:
            logger.warning("データ取得失敗のためスキップ: %s/%s", market, symbol, exc_info=True)
    return data


def _window_bounds(start: str, end: str, n_windows: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end] を n_windows 個の連続期間に等分割する。"""
    ts = pd.date_range(start=start, end=end, periods=n_windows + 1)
    return [(ts[i], ts[i + 1]) for i in range(n_windows)]


def _make_backtester(
    initial_cash: float, fee_rate: float, slippage: float, stop_loss_pct: Optional[float]
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
        take_profit_pct=None,
        position_sizing="full",
    )


def evaluate_hypothesis(
    hypothesis: FactoryHypothesis,
    data_by_symbol: dict[str, pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: Optional[float] = 0.07,
) -> FactoryEvaluation:
    """1仮説を全銘柄 × 全期間 + 窓別に評価する。

    - 全期間メトリクス: 銘柄ごとの全期間シミュレーションを平均（取引数は合計、DD は最悪値）
    - 窓別リターン: 各窓を独立にシミュレーションし銘柄平均（PBO の行列要素になる）
    """
    rule = build_rule(hypothesis.rule_spec)
    backtester = _make_backtester(initial_cash, fee_rate, slippage, stop_loss_pct)

    sharpes: list[float] = []
    sharpes_per_trade: list[float] = []
    win_rates: list[float] = []
    total_returns: list[float] = []
    drawdowns: list[float] = []
    num_trades = 0
    window_returns_by_symbol: list[list[float]] = []

    for symbol, df in data_by_symbol.items():
        try:
            signal = rule.generate_signal(df)
            if int((signal == 1).sum()) == 0:
                window_returns_by_symbol.append([0.0] * len(windows))
                continue

            _, metrics = backtester.simulate_trading(df, signal)
            sharpes.append(float(metrics.get("sharpe_ratio", 0.0) or 0.0))
            sharpes_per_trade.append(float(metrics.get("sharpe_per_trade", 0.0) or 0.0))
            win_rates.append(float(metrics.get("win_rate", 0.0) or 0.0))
            total_returns.append(float(metrics.get("total_return", 0.0) or 0.0))
            drawdowns.append(float(metrics.get("max_drawdown", 0.0) or 0.0))
            num_trades += int(metrics.get("num_trades", 0) or 0)

            sym_window_returns = []
            for w_start, w_end in windows:
                mask = (df.index >= w_start) & (df.index < w_end)
                w_df, w_sig = df.loc[mask], signal.loc[mask]
                if len(w_df) < 5 or int((w_sig == 1).sum()) == 0:
                    sym_window_returns.append(0.0)
                    continue
                _, w_metrics = backtester.simulate_trading(w_df, w_sig)
                sym_window_returns.append(float(w_metrics.get("total_return", 0.0) or 0.0))
            window_returns_by_symbol.append(sym_window_returns)
        except Exception:
            logger.warning(
                "仮説評価失敗（銘柄スキップ）: %s [%s]",
                hypothesis.label,
                symbol,
                exc_info=True,
            )

    window_returns = (
        np.mean(np.asarray(window_returns_by_symbol, dtype=float), axis=0).tolist()
        if window_returns_by_symbol
        else [0.0] * len(windows)
    )
    return FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=float(np.mean(sharpes)) if sharpes else 0.0,
        sharpe_per_trade=float(np.mean(sharpes_per_trade)) if sharpes_per_trade else 0.0,
        win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
        num_trades=num_trades,
        max_drawdown=float(min(drawdowns)) if drawdowns else 0.0,
        total_return=float(np.mean(total_returns)) if total_returns else 0.0,
        window_returns=window_returns,
        n_symbols=len(data_by_symbol),
    )


# ---------------------------------------------------------------------------
# ゲート
# ---------------------------------------------------------------------------


def apply_gate(evaluation: FactoryEvaluation, champion_sharpe: float) -> None:
    """ゲート条件を判定し evaluation.gate_passed / gate_reasons を更新する。

    PBO はバッチ全体で1値となる性質上、per-hypothesis ゲートに使うと「一晩全滅」に
    なるため、ここでは判定しない（バッチ診断としてレポート/通知に警告表示する）。
    DSR はトレード単位 Sharpe（年率化を打ち消した値）で算出済みのため飽和しない。
    """
    reasons: list[str] = []
    if evaluation.num_trades < FACTORY_GATE_MIN_TRADES:
        reasons.append(f"num_trades {evaluation.num_trades} < {FACTORY_GATE_MIN_TRADES}")
    if math.isnan(evaluation.dsr) or evaluation.dsr < FACTORY_GATE_MIN_DSR:
        reasons.append(f"dsr {evaluation.dsr:.3f} < {FACTORY_GATE_MIN_DSR}")
    if evaluation.max_drawdown < FACTORY_GATE_MAX_DRAWDOWN:
        reasons.append(f"max_drawdown {evaluation.max_drawdown:.3f} < {FACTORY_GATE_MAX_DRAWDOWN}")
    if not math.isnan(champion_sharpe):
        required = champion_sharpe * FACTORY_GATE_CHAMPION_MARGIN
        if evaluation.sharpe_ratio <= required:
            reasons.append(
                f"sharpe {evaluation.sharpe_ratio:.3f} <= champion×{FACTORY_GATE_CHAMPION_MARGIN}"
                f" ({required:.3f})"
            )
    evaluation.gate_reasons = reasons
    evaluation.gate_passed = not reasons


# ---------------------------------------------------------------------------
# バッチ実行（orchestration から呼ばれるエントリ）
# ---------------------------------------------------------------------------


def run_factory_batch(
    market: str,
    symbols: list[str],
    budget: int = 10,
    lookback_years: int = 2,
    n_windows: int = 8,
    seed: Optional[int] = None,
) -> FactoryBatchResult:
    """夜間バッチ1回分: サンプリング → 評価 → ゲート → 記録 → レポート出力。

    symbols は呼び出し元（orchestration）が load_target_symbols() 等で注入する。
    """
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(lookback_years * 365))).strftime("%Y-%m-%d")
    if seed is None:
        seed = int(datetime.now().strftime("%Y%m%d"))

    existing = load_factory_hashes()
    candidates = sample_hypotheses(
        market, budget, existing, seed=seed, lookback_years=lookback_years
    )
    controls = control_hypotheses(market, lookback_years=lookback_years)
    batch = controls + candidates
    logger.info(
        "[factory] バッチ開始: market=%s 候補=%d 対照=%d 期間=%s〜%s",
        market,
        len(candidates),
        len(controls),
        start,
        end,
    )

    data = _load_symbol_data(market, symbols, start, end)
    result = FactoryBatchResult(market=market)
    if not data:
        logger.error("[factory] 有効な銘柄データがないため中止: market=%s", market)
        return result

    windows = _window_bounds(start, end, n_windows)
    evaluations = [evaluate_hypothesis(h, data, windows) for h in batch]

    # PBO: バッチ全体（対照込み）の窓別リターン行列 (W, N)
    matrix = np.asarray([e.window_returns for e in evaluations], dtype=float).T
    batch_pbo = probability_of_backtest_overfitting(
        matrix, n_splits=min(10, n_windows - (n_windows % 2))
    )

    # DSR: n_trials は累計評価数（過去全試行 + 今夜の候補数）
    n_trials = count_factory_runs() + len(candidates)
    control_sharpes = [
        e.sharpe_ratio for e in evaluations if e.hypothesis.is_control and e.num_trades > 0
    ]
    champion_sharpe = max(control_sharpes) if control_sharpes else float("nan")

    for evaluation in evaluations:
        evaluation.pbo = float(batch_pbo)
        # DSR には非年率の取引単位 Sharpe を渡す（compute_metrics が直接出力する。
        # metrics 側で年率化が実取引頻度ベースに是正されたため、旧 √252 de-scale は不要）。
        evaluation.dsr = deflated_sharpe_ratio(
            evaluation.sharpe_per_trade, max(n_trials, 1), max(evaluation.num_trades, 1)
        )
        if not evaluation.hypothesis.is_control:
            apply_gate(evaluation, champion_sharpe)

    # PBO はバッチ単位の過学習診断。高ければログ警告（レポートにも注記される）。
    if not math.isnan(batch_pbo) and batch_pbo > FACTORY_GATE_MAX_PBO:
        logger.warning(
            "[factory] バッチPBO=%.3f > %.2f（選択過程の過学習リスク高）。"
            "合格レポートには警告を付与する。",
            batch_pbo,
            FACTORY_GATE_MAX_PBO,
        )

    # 記録 + レポート出力（候補のみ。逐次実行なので DuckDB 書き込み規約に適合）
    for evaluation in result_candidates(evaluations):
        if evaluation.gate_passed:
            review = review_hypothesis(evaluation, champion_sharpe)
            evaluation.report_path = write_report(
                evaluation, champion_sharpe, (start, end), review=review
            )
        save_factory_run(
            hypothesis_hash=evaluation.hypothesis.hypothesis_hash,
            market=market,
            spec_json=json.dumps(evaluation.hypothesis.rule_spec, ensure_ascii=False),
            sharpe_ratio=evaluation.sharpe_ratio,
            win_rate=evaluation.win_rate,
            num_trades=evaluation.num_trades,
            max_drawdown=evaluation.max_drawdown,
            total_return=evaluation.total_return,
            dsr=evaluation.dsr,
            pbo=evaluation.pbo,
            gate_passed=evaluation.gate_passed,
            gate_reasons="; ".join(evaluation.gate_reasons) or None,
            report_path=evaluation.report_path,
        )

    result.evaluated = evaluations
    result.champion_sharpe = champion_sharpe
    result.pbo = float(batch_pbo)
    logger.info(
        "[factory] バッチ完了: market=%s 評価=%d 合格=%d champion_sharpe=%.3f pbo=%.3f",
        market,
        len(result.candidates),
        len(result.passed),
        champion_sharpe if not math.isnan(champion_sharpe) else float("nan"),
        batch_pbo if not math.isnan(batch_pbo) else float("nan"),
    )
    return result


def result_candidates(evaluations: list[FactoryEvaluation]) -> list[FactoryEvaluation]:
    """候補（非対照）の評価のみを返す。"""
    return [e for e in evaluations if not e.hypothesis.is_control]
