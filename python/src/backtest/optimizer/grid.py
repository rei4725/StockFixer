"""グリッドサーチ最適化。

閾値・ストップロス・テイクプロフィット・ATR パラメータの総当たりを
Walk-Forward 検証で実行し、全組み合わせの結果 DataFrame を返す。
"""

import itertools
import math
from typing import Optional

import pandas as pd

from src.backtest.optimizer._pbo import _compute_pbo_from_fold_returns
from src.backtest.pipeline import run_backtest_walk_forward
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _frange(start: float, stop: float, step: float) -> list[float]:
    """浮動小数点レンジを生成する"""
    result = []
    val = start
    while val <= stop + step * 0.01:
        result.append(round(val, 6))
        val += step
    return result


def _parse_grid_values(values: Optional[list[float]], default: float) -> list[float]:
    if not values:
        return [default]
    return [float(v) for v in values]


def run_optimization(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: Optional[float] = None,
    dynamic_slippage: bool = True,
    position_sizing: str = "full",
    position_fraction: float = 0.5,
    atr_risk_pcts: Optional[list[float]] = None,
    atr_multipliers: Optional[list[float]] = None,
    atr_min_fraction: float = 0.1,
    atr_max_fraction: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 0.015,
    threshold_step: float = 0.001,
    optimize_risk: bool = False,
) -> pd.DataFrame:
    """
    グリッドサーチを実行し、全パラメータ組み合わせの結果を返す。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_type: モデルタイプ
        ensemble: アンサンブル予測を使用するか
        source: データソース
        n_splits: Walk-Forward分割数
        initial_cash: 初期資金
        fee_rate: 取引手数料率
        slippage: スリッページ
        position_sizing: ポジションサイジング種別
        position_fraction: fixed モード用比率
        atr_risk_pcts: ATR リスク割合の候補一覧
        atr_multipliers: ATR 倍率の候補一覧
        atr_min_fraction: ATR 建玉下限比率
        atr_max_fraction: ATR 建玉上限比率
        threshold_min: 閾値の最小値
        threshold_max: 閾値の最大値
        threshold_step: 閾値のステップ
        optimize_risk: ストップロス・テイクプロフィットもグリッドサーチするか

    Returns:
        全パラメータ組み合わせの結果 DataFrame
    """
    thresholds = _frange(threshold_min, threshold_max, threshold_step)

    if optimize_risk:
        stop_losses = [None, 0.02, 0.03, 0.05, 0.07, 0.10]
        take_profits = [None, 0.03, 0.05, 0.07, 0.10, 0.15]
    else:
        stop_losses = [None]
        take_profits = [None]

    atr_risk_grid = _parse_grid_values(atr_risk_pcts, 0.02)
    atr_multiplier_grid = _parse_grid_values(atr_multipliers, 1.0)
    if position_sizing != "atr":
        atr_risk_grid = [0.02]
        atr_multiplier_grid = [1.0]

    param_grid = list(
        itertools.product(thresholds, stop_losses, take_profits, atr_risk_grid, atr_multiplier_grid)
    )
    total = len(param_grid)

    print(f"\n最適化開始: {market}/{symbol}")
    print(f"パラメータ組み合わせ数: {total}")
    print(f"  閾値: {thresholds}")
    if optimize_risk:
        print(f"  ストップロス: {stop_losses}")
        print(f"  テイクプロフィット: {take_profits}")
    if position_sizing == "atr":
        print(f"  ATR risk_pct: {atr_risk_grid}")
        print(f"  ATR multiplier: {atr_multiplier_grid}")
        print(f"  ATR fraction range: {atr_min_fraction} - {atr_max_fraction}")
    print()

    all_results = []
    # PBO 算出用: 成功した候補ごとの fold 別リターン系列（時系列順）
    fold_returns_by_candidate: list[list[float]] = []

    for i, (threshold, stop_loss, take_profit, atr_risk_pct, atr_multiplier) in enumerate(
        param_grid, 1
    ):
        label = f"[{i}/{total}] threshold={threshold}"
        if stop_loss is not None:
            label += f", SL={stop_loss}"
        if take_profit is not None:
            label += f", TP={take_profit}"
        if position_sizing == "atr":
            label += f", ATRrisk={atr_risk_pct}, ATRx={atr_multiplier}"
        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}")

        try:
            _, _, wf_df = run_backtest_walk_forward(
                market=market,
                symbol=symbol,
                model_type=model_type,
                threshold=threshold,
                source=source,
                n_splits=n_splits,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                dynamic_slippage=dynamic_slippage,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                position_sizing=position_sizing,
                position_fraction=position_fraction,
                atr_risk_pct=atr_risk_pct,
                atr_multiplier=atr_multiplier,
                atr_min_fraction=atr_min_fraction,
                atr_max_fraction=atr_max_fraction,
                ensemble=ensemble,
            )

            if wf_df is not None and not wf_df.empty:
                # 各fold平均をサマリーとして保存
                numeric_cols = [
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "gross_total_return",
                    "gross_sharpe_ratio",
                    "gross_max_drawdown",
                    "cost_impact_return",
                    "cost_impact_cash",
                    "win_rate",
                    "profit_factor",
                    "num_trades",
                    "avg_position_fraction",
                    "max_position_fraction",
                    "avg_position_value",
                    "atr_fallback_trades",
                    "avg_win",
                    "avg_loss",
                ]
                available = [c for c in numeric_cols if c in wf_df.columns]
                # None値をnp.nanに変換して平均計算可能にする
                wf_df_numeric = wf_df[available].copy()
                for col in wf_df_numeric.columns:
                    wf_df_numeric[col] = pd.to_numeric(wf_df_numeric[col], errors="coerce")
                summary = wf_df_numeric.mean().to_dict()
                summary["threshold"] = threshold
                summary["stop_loss_pct"] = stop_loss
                summary["take_profit_pct"] = take_profit
                summary["position_sizing"] = position_sizing
                summary["position_fraction"] = position_fraction
                summary["atr_risk_pct"] = atr_risk_pct
                summary["atr_multiplier"] = atr_multiplier
                summary["atr_min_fraction"] = atr_min_fraction
                summary["atr_max_fraction"] = atr_max_fraction
                summary["ensemble"] = ensemble
                all_results.append(summary)
                if "total_return" in wf_df.columns:
                    fold_returns_by_candidate.append(
                        pd.to_numeric(wf_df["total_return"], errors="coerce")
                        .fillna(0.0)
                        .astype(float)
                        .tolist()
                    )
        except Exception as e:
            logger.warning(f"最適化エラー: {symbol}", exc_info=True)
            all_results.append(
                {
                    "threshold": threshold,
                    "stop_loss_pct": stop_loss,
                    "take_profit_pct": take_profit,
                    "position_sizing": position_sizing,
                    "atr_risk_pct": atr_risk_pct,
                    "atr_multiplier": atr_multiplier,
                    "error": str(e),
                }
            )

    result_df = pd.DataFrame(all_results)

    # 過学習ガード: 全候補の fold リターン行列から PBO を算出する。
    # 探索全体に対する1つの指標のため、全行に同じ値を付与する
    # （save_optimal_params_json が best 行から読めるようにする）。
    pbo = _compute_pbo_from_fold_returns(fold_returns_by_candidate, market, symbol)
    if not math.isnan(pbo) and not result_df.empty:
        result_df["pbo"] = pbo

    return result_df
