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
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    FACTORY_CLAUDE_RULEGEN_ENABLED,
    FACTORY_GATE_MAX_PBO,
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,
    FACTORY_GATE_MIN_TRADES_PER_SYMBOL,
)
from src.backtest.backtester import Backtester
from src.backtest.claude_rule_generator import generate_claude_hypotheses
from src.backtest.data_port import get_backtest_data_port
from src.backtest.factory_aggregation import SymbolMetrics, aggregate_symbol_metrics

# apply_gate は factory_gate.py へ切り出したが、既存の
# `from src.backtest.factory import apply_gate` を維持するため再エクスポートする。
from src.backtest.factory_gate import apply_gate  # noqa: F401
from src.backtest.factory_portfolio import portfolio_max_drawdown
from src.backtest.factory_report import write_report

# 探索空間とサンプラーは factory_sampling.py へ切り出したが、既存の
# `from src.backtest.factory import sample_hypotheses` を維持するため再エクスポートする。
from src.backtest.factory_sampling import (  # noqa: F401
    _PARAM_GRID,
    _RULE_CLASSES,
    control_hypotheses,
    sample_hypotheses,
)
from src.backtest.hypothesis_review import review_hypothesis
from src.backtest.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from src.backtest.rules import AndRule, OrRule, TradingRule
from src.backtest.sandbox_executor import prepare_sandbox_data
from src.backtest.types import FactoryBatchResult, FactoryEvaluation, FactoryHypothesis
from src.utils.data_path_utils import get_ticker
from src.utils.db import count_factory_runs, load_factory_hashes, save_factory_run
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ゲート閾値は config/settings.py（FACTORY_GATE_*）で定義し env 上書き可能

_MIN_SYMBOL_ROWS = 100

# 指標の助走（ウォームアップ）区間が評価期間に混入しないための前倒しダウンロード日数（#629）。
# 探索空間の最長ウィンドウ（volume_breakout の breakout_window=40 / ema_momentum の
# slow_window=50 等、トレーディング日数）に、週末・祝日を考慮した安全マージンを載せた
# カレンダー日数。
_WARMUP_CALENDAR_DAYS = 120


# ---------------------------------------------------------------------------
# ルール構築
# ---------------------------------------------------------------------------


_SANDBOX_ENV_FLAG = "STOCKFIXER_SANDBOX"


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
    if spec_type == "generated_code":
        if os.environ.get(_SANDBOX_ENV_FLAG) != "1":
            raise RuntimeError(
                "generated_code スペックはサンドボックスコンテナ内でのみ構築できます"
                f"（環境変数 {_SANDBOX_ENV_FLAG}=1 が必要）。信頼された本体プロセスから"
                "未検証の生成コードをexecすることを防ぐガードです。"
            )
        return _build_generated_rule(spec)
    raise ValueError(f"未知の spec type: {spec_type}")


def _build_generated_rule(spec: dict) -> TradingRule:
    """generated_code spec からクラスをexecして TradingRule インスタンスを構築する。

    呼び出し元（build_rule）がサンドボックス環境変数を検証済みであることが前提。
    """
    source_code = spec["source_code"]
    class_name = spec["class_name"]
    namespace: dict = {}
    exec(compile(source_code, "<generated_rule>", "exec"), namespace)  # nosec B102
    if class_name not in namespace:
        raise ValueError(f"生成コードにクラス '{class_name}' が見つかりません")
    rule_cls = namespace[class_name]
    return rule_cls()


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------


def _load_symbol_data(
    market: str, symbols: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """銘柄ごとに OHLCV + テクニカル指標を1回だけ取得する。

    指標（EMA/MACD/出来高ブレイクアウト等）の助走区間が評価期間 [start, end) に
    混入すると評価値がデータ取得開始日に依存してしまうため、start より
    _WARMUP_CALENDAR_DAYS 日前からダウンロードして指標を計算し、その後
    評価対象期間へスライスしてから返す（#629）。
    """
    port = get_backtest_data_port()
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=_WARMUP_CALENDAR_DAYS)).strftime(
        "%Y-%m-%d"
    )
    eval_start = pd.Timestamp(start)

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = port.download(get_ticker(market, symbol), start=warmup_start, end=end)
            if df is None or df.empty:
                logger.warning("データ不足のためスキップ: %s/%s", market, symbol)
                continue
            df = port.add_technical_indicators(df)
            df = df.dropna(subset=["Close"])
            df = df[df.index >= eval_start]
            if len(df) < _MIN_SYMBOL_ROWS:
                logger.warning("データ不足のためスキップ: %s/%s", market, symbol)
                continue
            data[symbol] = df
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


def _gate_sharpe(evaluation: FactoryEvaluation) -> float:
    """ゲート判定に使う Sharpe。候補と対照で同じ指標を使うための共通ヘルパ。

    プール済み per-trade ベースの値を優先し、算出不能なら従来の銘柄別平均に落とす
    （factory_gate._append_champion_reason のフォールバックと同じ規則）。
    """
    value = evaluation.portfolio_sharpe_ratio
    return evaluation.sharpe_ratio if math.isnan(value) else value


def _nullable(value: float) -> Optional[float]:
    """NaN を None に落とす（DB に NaN を書かないため）。"""
    return None if math.isnan(value) else value


def _span_years(data_by_symbol: dict[str, pd.DataFrame]) -> float:
    """評価データが実際にカバーする期間の年数を返す（算出できなければ 0.0）。

    宣言値 lookback_years ではなく実データの範囲を使う。データ取得が短く終わった
    夜に取引頻度を過小評価して Sharpe を不当に低く見積もらないため。
    """
    starts, ends = [], []
    for df in data_by_symbol.values():
        if df is None or df.empty:
            continue
        starts.append(df.index.min())
        ends.append(df.index.max())
    if not starts:
        return 0.0
    span_days = (max(ends) - min(starts)).days
    return span_days / 365.25 if span_days > 0 else 0.0


def evaluate_hypothesis(
    hypothesis: FactoryHypothesis,
    data_by_symbol: dict[str, pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: Optional[float] = 0.07,
    min_trades_per_symbol: Optional[int] = None,
) -> FactoryEvaluation:
    """1仮説を全銘柄 × 全期間 + 窓別に評価する。

    - 全期間メトリクス: 銘柄あたり最低取引数を満たす銘柄のみで集計する（#625）。
      取引が 2 回だけの銘柄は Sharpe が発散するため、混ぜると平均が壊れる。
    - 窓別リターン: 各窓を独立にシミュレーションし全銘柄で平均する。
      こちらは意図的にフィルタしない。PBO はバッチ単位の診断指標であり
      apply_gate の判定に使われないため、母数を変えると PBO の意味が変わる。
    """
    if min_trades_per_symbol is None:
        min_trades_per_symbol = FACTORY_GATE_MIN_TRADES_PER_SYMBOL
    rule = build_rule(hypothesis.rule_spec)
    backtester = _make_backtester(initial_cash, fee_rate, slippage, stop_loss_pct)

    symbol_rows: list[SymbolMetrics] = []
    window_returns_by_symbol: list[list[float]] = []
    equity_by_symbol: dict[str, pd.Series] = {}

    for symbol, df in data_by_symbol.items():
        try:
            signal = rule.generate_signal(df)
            if int((signal == 1).sum()) == 0:
                window_returns_by_symbol.append([0.0] * len(windows))
                continue

            _, metrics = backtester.simulate_trading(df, signal, collect_equity=True)
            equity = metrics.get("equity_curve")
            if equity is not None and not equity.empty:
                equity_by_symbol[symbol] = equity
            symbol_rows.append(
                SymbolMetrics(
                    symbol=symbol,
                    num_trades=int(metrics.get("num_trades", 0) or 0),
                    sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0) or 0.0),
                    sharpe_per_trade=float(metrics.get("sharpe_per_trade", 0.0) or 0.0),
                    win_rate=float(metrics.get("win_rate", 0.0) or 0.0),
                    total_return=float(metrics.get("total_return", 0.0) or 0.0),
                    max_drawdown=float(metrics.get("max_drawdown", 0.0) or 0.0),
                    trade_returns=list(metrics.get("trade_returns", []) or []),
                )
            )

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

    aggregated = aggregate_symbol_metrics(
        symbol_rows, min_trades_per_symbol, span_years=_span_years(data_by_symbol)
    )
    # ゲートの DD 指標は、集計と同じ有効銘柄だけを等金額で保有したポートフォリオDD
    portfolio_dd = portfolio_max_drawdown(
        [equity_by_symbol[s] for s in aggregated.effective_symbols if s in equity_by_symbol],
        initial_cash,
    )
    window_returns = (
        np.mean(np.asarray(window_returns_by_symbol, dtype=float), axis=0).tolist()
        if window_returns_by_symbol
        else [0.0] * len(windows)
    )
    return FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=aggregated.sharpe_ratio,
        sharpe_per_trade=aggregated.sharpe_per_trade,
        portfolio_sharpe_ratio=aggregated.portfolio_sharpe_ratio,
        win_rate=aggregated.win_rate,
        num_trades=aggregated.num_trades,
        max_drawdown=aggregated.max_drawdown,
        portfolio_max_drawdown=portfolio_dd,
        total_return=aggregated.total_return,
        window_returns=window_returns,
        n_symbols=len(data_by_symbol),
        n_symbols_with_signal=aggregated.n_symbols_with_signal,
        n_effective_symbols=aggregated.n_effective_symbols,
        avg_trades_per_symbol=aggregated.avg_trades_per_symbol,
    )


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

    本関数だけが batch_run_id を払い出し、レポートの産地情報として write_report に渡す（#703）。
    write_report を直接呼ぶ経路には batch_run_id が渡らないため provenance.source は
    "unknown" となり、IssueAgent の intake が起票を拒否する。
    """
    batch_run_id = str(uuid.uuid4())
    logger.info("[factory] バッチ開始: batch_run_id=%s 銘柄数=%d", batch_run_id, len(symbols))
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

    claude_evaluations: list[FactoryEvaluation] = []
    if FACTORY_CLAUDE_RULEGEN_ENABLED:
        control_sharpes_pre = [
            _gate_sharpe(e)
            for e in evaluations
            if e.hypothesis.is_control
            and e.n_effective_symbols >= FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS
        ]
        pre_champion_sharpe = max(control_sharpes_pre) if control_sharpes_pre else float("nan")
        shared_data_dir, windows_file = prepare_sandbox_data(data, windows)
        try:
            claude_evaluations = generate_claude_hypotheses(
                market, pre_champion_sharpe, shared_data_dir, windows_file
            )
            evaluations.extend(claude_evaluations)
        finally:
            shutil.rmtree(shared_data_dir, ignore_errors=True)
            Path(windows_file).unlink(missing_ok=True)

    # PBO: バッチ全体（対照込み）の窓別リターン行列 (W, N)
    matrix = np.asarray([e.window_returns for e in evaluations], dtype=float).T
    batch_pbo = probability_of_backtest_overfitting(
        matrix, n_splits=min(10, n_windows - (n_windows % 2))
    )

    # DSR: n_trials は累計評価数（過去全試行 + 今夜の候補数、Claude生成候補を含む）
    n_trials = count_factory_runs() + len(candidates) + len(claude_evaluations)
    # チャンピオンプールの選抜条件は候補ゲートの有効銘柄数下限と揃える（#627）。
    # 以前は num_trades > 0 だったため銘柄あたり最低取引数フィルタ後も
    # 1取引あれば通ってしまい、実質チェックとして機能していなかった。
    control_sharpes = [
        _gate_sharpe(e)
        for e in evaluations
        if e.hypothesis.is_control and e.n_effective_symbols >= FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS
    ]
    champion_sharpe = max(control_sharpes) if control_sharpes else float("nan")
    if not control_sharpes and controls:
        # 対照群が1件以上評価されたにもかかわらず全滅した場合、apply_gate は
        # champion_sharpe=NaN を fail-closed（不合格）として扱う（#627）。
        # 一晩探索が止まるが、最強のゲートが無言で外れて質の悪い仮説が
        # 通過するより安全側に倒す判断とする。
        logger.warning(
            "[factory] 対照群が全て有効銘柄数フィルタで除外されたため、"
            "今回のバッチは全候補を不合格（fail-closed）とします: 評価済み対照数=%d",
            len(controls),
        )

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
                evaluation,
                champion_sharpe,
                (start, end),
                review=review,
                batch_run_id=batch_run_id,
                symbol_universe_size=len(symbols),
            )
        save_factory_run(
            hypothesis_hash=evaluation.hypothesis.hypothesis_hash,
            market=market,
            spec_json=json.dumps(evaluation.hypothesis.rule_spec, ensure_ascii=False),
            sharpe_ratio=evaluation.sharpe_ratio,
            portfolio_sharpe_ratio=_nullable(evaluation.portfolio_sharpe_ratio),
            win_rate=evaluation.win_rate,
            num_trades=evaluation.num_trades,
            max_drawdown=evaluation.max_drawdown,
            portfolio_max_drawdown=_nullable(evaluation.portfolio_max_drawdown),
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
