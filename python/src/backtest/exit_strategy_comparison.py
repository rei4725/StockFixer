"""
Walk-Forward によるエグジット戦略比較ユーティリティ (R-402)

固定 TP/SL 戦略と ML ベースエグジット戦略の成績（Net Return, MDD）を
Walk-Forward で比較し、結果 DataFrame を返す。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 固定 TP/SL のデフォルト値
_DEFAULT_TAKE_PROFIT = 0.10  # 10% 利確
_DEFAULT_STOP_LOSS = 0.05  # 5% 損切


def compare_exit_strategies(
    market: str,
    symbol: str,
    model_manager: Any,
    signal_generator: Any,
    model_name: str,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
    n_splits: int = 5,
    take_profit_pct: float = _DEFAULT_TAKE_PROFIT,
    stop_loss_pct: float = _DEFAULT_STOP_LOSS,
) -> pd.DataFrame:
    """
    固定 TP/SL 戦略と ML エグジット戦略を Walk-Forward で比較する。

    固定 TP/SL 戦略: 指定した take_profit_pct / stop_loss_pct で機械的に決済。
    ML エグジット戦略: TP/SL なし（モデルシグナルによる決済のみ）。

    Args:
        market: マーケットコード（例: "jp"）
        symbol: 銘柄コード（例: "7203"）
        model_manager: ModelManager インスタンス
        signal_generator: シグナル生成オブジェクト
        model_name: 使用するモデル名
        initial_cash: 初期資金
        fee_rate: 手数料率
        slippage: スリッページ率
        n_splits: Walk-Forward の分割数
        take_profit_pct: 固定利確幅
        stop_loss_pct: 固定損切幅

    Returns:
        各戦略・フォールドの比較結果 DataFrame。列:
        - strategy: "fixed_tp_sl" | "ml_exit"
        - fold: フォールド番号
        - val_start / val_end: 検証期間
        - total_return: Net Return
        - max_drawdown: MDD
        - sharpe_ratio: シャープレシオ
        - win_rate: 勝率
    """
    from src.backtest.walk_forward import WalkForwardValidator

    common_kwargs: dict = {
        "market": market,
        "symbol": symbol,
        "model_manager": model_manager,
        "signal_generator": signal_generator,
        "initial_cash": initial_cash,
        "fee_rate": fee_rate,
        "slippage": slippage,
        "n_splits": n_splits,
    }

    logger.info(
        "[exit_cmp] %s/%s: Walk-Forward 比較開始 (n_splits=%d, TP=%.0f%%, SL=%.0f%%)",
        market,
        symbol,
        n_splits,
        take_profit_pct * 100,
        stop_loss_pct * 100,
    )

    # 固定 TP/SL 戦略
    fixed_wfv = WalkForwardValidator(
        **common_kwargs,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    fixed_result = fixed_wfv.run(model_name=model_name)
    fixed_result["strategy"] = "fixed_tp_sl"

    # ML エグジット戦略（TP/SL なし）
    ml_wfv = WalkForwardValidator(
        **common_kwargs,
        stop_loss_pct=None,
        take_profit_pct=None,
    )
    ml_result = ml_wfv.run(model_name=model_name)
    ml_result["strategy"] = "ml_exit"

    result = pd.concat([fixed_result, ml_result], ignore_index=True)

    _print_comparison(result, take_profit_pct, stop_loss_pct)
    return result


def _print_comparison(
    result: pd.DataFrame,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> None:
    """比較結果のサマリーを標準出力に表示する。"""
    summary_cols = [
        c
        for c in [
            "strategy",
            "fold",
            "val_start",
            "val_end",
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "win_rate",
        ]
        if c in result.columns
    ]
    numeric_cols = [
        c
        for c in ["total_return", "max_drawdown", "sharpe_ratio", "win_rate"]
        if c in result.columns
    ]

    print("\n" + "=" * 70)
    print(f"エグジット戦略比較: 固定 TP/SL ({take_profit_pct:.0%}/{stop_loss_pct:.0%}) vs. ML エグジット")
    print("=" * 70)
    print(result[summary_cols].to_string(index=False))

    if numeric_cols:
        print("\n--- 戦略別平均 ---")
        for strategy, grp in result.groupby("strategy"):
            means = grp[numeric_cols].apply(pd.to_numeric, errors="coerce").mean().round(4)
            print(f"\n[{strategy}]")
            print(means.to_string())
    print("=" * 70)
