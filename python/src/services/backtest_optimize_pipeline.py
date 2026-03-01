"""
バックテスト最適化パイプラインサービス

閾値・ストップロス・テイクプロフィットのグリッドサーチを
Walk-Forward検証で実行し、最適パラメータを特定する。

run_backtest_optimize.py はこのモジュールの関数を呼び出すラッパーとして機能する。
"""
import itertools
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from src.services.backtest_pipeline import run_backtest_walk_forward
from src.utils.data_path_utils import get_results_dir, ensure_dir


def _frange(start: float, stop: float, step: float) -> list[float]:
    """浮動小数点レンジを生成する"""
    result = []
    val = start
    while val <= stop + step * 0.01:
        result.append(round(val, 6))
        val += step
    return result


def run_optimization(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
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

    param_grid = list(itertools.product(thresholds, stop_losses, take_profits))
    total = len(param_grid)

    print(f"\n最適化開始: {market}/{symbol}")
    print(f"パラメータ組み合わせ数: {total}")
    print(f"  閾値: {thresholds}")
    if optimize_risk:
        print(f"  ストップロス: {stop_losses}")
        print(f"  テイクプロフィット: {take_profits}")
    print()

    all_results = []

    for i, (threshold, stop_loss, take_profit) in enumerate(param_grid, 1):
        label = f"[{i}/{total}] threshold={threshold}"
        if stop_loss is not None:
            label += f", SL={stop_loss}"
        if take_profit is not None:
            label += f", TP={take_profit}"
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
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                ensemble=ensemble,
            )

            if wf_df is not None and not wf_df.empty:
                # 各fold平均をサマリーとして保存
                numeric_cols = ["total_return", "sharpe_ratio", "max_drawdown",
                                "win_rate", "profit_factor", "num_trades"]
                available = [c for c in numeric_cols if c in wf_df.columns]
                # None値をnp.nanに変換して平均計算可能にする
                wf_df_numeric = wf_df[available].copy()
                for col in wf_df_numeric.columns:
                    wf_df_numeric[col] = pd.to_numeric(wf_df_numeric[col], errors='coerce')
                summary = wf_df_numeric.mean().to_dict()
                summary["threshold"] = threshold
                summary["stop_loss_pct"] = stop_loss
                summary["take_profit_pct"] = take_profit
                summary["ensemble"] = ensemble
                all_results.append(summary)
        except Exception as e:
            print(f"  エラー: {e}")
            all_results.append({
                "threshold": threshold,
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
                "error": str(e),
            })

    return pd.DataFrame(all_results)


def print_optimization_results(result_df: pd.DataFrame, sort_by: str) -> None:
    """
    最適化結果を表示する。

    Args:
        result_df: run_optimization が返す DataFrame
        sort_by: ソート基準列名
    """
    print("\n" + "=" * 70)
    print("最適化結果サマリー")
    print("=" * 70)

    if result_df.empty:
        print("結果なし")
        return

    # エラー行を除外
    valid = result_df[~result_df.get("error", pd.Series(dtype=str)).notna()].copy()
    if "error" in valid.columns:
        valid = valid.drop(columns=["error"])

    if valid.empty:
        print("有効な結果なし（全てエラー）")
        return

    # ソート
    ascending = sort_by == "max_drawdown"
    if sort_by in valid.columns:
        valid = valid.sort_values(sort_by, ascending=ascending)

    display_cols = ["threshold", "stop_loss_pct", "take_profit_pct",
                    "total_return", "sharpe_ratio", "max_drawdown",
                    "win_rate", "profit_factor", "num_trades"]
    display_cols = [c for c in display_cols if c in valid.columns]

    print(valid[display_cols].to_string(index=False))

    # ベスト結果
    best = valid.iloc[-1] if not ascending else valid.iloc[0]
    print(f"\n{'='*70}")
    print(f"ベスト（{sort_by}基準）:")
    print(f"  閾値: {best.get('threshold', 'N/A')}")
    if "stop_loss_pct" in best and best["stop_loss_pct"] is not None:
        print(f"  ストップロス: {best['stop_loss_pct']}")
    if "take_profit_pct" in best and best["take_profit_pct"] is not None:
        print(f"  テイクプロフィット: {best['take_profit_pct']}")
    for col in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]:
        if col in best:
            print(f"  {col}: {best[col]}")
    print(f"{'='*70}")


def save_optimization_results(
    result_df: pd.DataFrame,
    market: str,
    symbol: str,
) -> str:
    """
    最適化結果を CSV に保存する。

    Args:
        result_df: run_optimization が返す DataFrame
        market: マーケット識別子
        symbol: 銘柄シンボル

    Returns:
        保存先ファイルパス
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(get_results_dir(), "optimize", f"{market}_{symbol}")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"optimize_{ts}.csv")
    result_df.to_csv(path, index=False)
    return path
