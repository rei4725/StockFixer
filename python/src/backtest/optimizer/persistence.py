"""最適化結果の表示・永続化。

結果サマリーのコンソール表示、CSV 保存、最適パラメータの JSON 統合保存、
保存済み最適パラメータの読み込みを担う。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    if "error" in result_df.columns:
        valid = result_df[result_df["error"].isna()].copy()
        valid = valid.drop(columns=["error"])
    else:
        valid = result_df.copy()

    if valid.empty:
        print("有効な結果なし（全てエラー）")
        return

    # ソート
    ascending = sort_by in {"max_drawdown", "cost_impact_return", "cost_impact_cash"}
    if sort_by in valid.columns:
        valid = valid.sort_values(sort_by, ascending=ascending)

    display_cols = [
        "threshold",
        "stop_loss_pct",
        "take_profit_pct",
        "position_sizing",
        "atr_risk_pct",
        "atr_multiplier",
        "total_return",
        "gross_total_return",
        "cost_impact_return",
        "sharpe_ratio",
        "gross_sharpe_ratio",
        "max_drawdown",
        "gross_max_drawdown",
        "win_rate",
        "profit_factor",
        "num_trades",
        "avg_position_fraction",
        "max_position_fraction",
        "avg_position_value",
        "atr_fallback_trades",
    ]
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
    for col in [
        "total_return",
        "gross_total_return",
        "cost_impact_return",
        "sharpe_ratio",
        "gross_sharpe_ratio",
        "max_drawdown",
        "gross_max_drawdown",
        "win_rate",
        "profit_factor",
    ]:
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


def save_optimal_params_json(
    result_df: pd.DataFrame,
    market: str,
    symbol: str,
    sort_by: str = "sharpe_ratio",
) -> str:
    """
    最適パラメータを JSON に保存する。
    シャープレシオ（またはソート基準）が最高のパラメータを抽出し、
    python/config/optimal_params.json に統合保存する。

    Args:
        result_df: run_optimization が返す DataFrame
        market: マーケット識別子
        symbol: 銘柄シンボル
        sort_by: ソート基準列名（デフォルト: sharpe_ratio）

    Returns:
        保存先ファイルパス
    """
    if result_df.empty:
        print("警告: 最適化結果が空です")
        return ""

    # エラー行を除外
    if "error" in result_df.columns:
        valid = result_df[result_df["error"].isna()].copy()
    else:
        valid = result_df.copy()

    if valid.empty:
        print("警告: 有効な最適化結果がありません")
        return ""

    # 最適パラメータを取得
    ascending = sort_by in {"max_drawdown", "cost_impact_return", "cost_impact_cash"}
    best_row = valid.sort_values(sort_by, ascending=ascending).iloc[-1 if not ascending else 0]

    # JSON形式に変換
    optimal_param = {
        "market": market,
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "sort_by": sort_by,
        "threshold": float(best_row.get("threshold", 0.0)),
        "stop_loss_pct": (
            float(best_row["stop_loss_pct"]) if best_row.get("stop_loss_pct") is not None else None
        ),
        "take_profit_pct": (
            float(best_row["take_profit_pct"])
            if best_row.get("take_profit_pct") is not None
            else None
        ),
        "position_sizing": str(best_row.get("position_sizing", "full")),
        "position_fraction": float(best_row.get("position_fraction", 0.5)),
        "atr_risk_pct": float(best_row.get("atr_risk_pct", 0.02)),
        "atr_multiplier": float(best_row.get("atr_multiplier", 1.0)),
        "atr_min_fraction": float(best_row.get("atr_min_fraction", 0.1)),
        "atr_max_fraction": float(best_row.get("atr_max_fraction", 1.0)),
        "metrics": {
            "total_return": float(best_row.get("total_return", 0.0)),
            "gross_total_return": float(best_row.get("gross_total_return", 0.0)),
            "cost_impact_return": float(best_row.get("cost_impact_return", 0.0)),
            "cost_impact_cash": float(best_row.get("cost_impact_cash", 0.0)),
            "sharpe_ratio": float(best_row.get("sharpe_ratio", 0.0)),
            "gross_sharpe_ratio": float(best_row.get("gross_sharpe_ratio", 0.0)),
            "max_drawdown": float(best_row.get("max_drawdown", 0.0)),
            "gross_max_drawdown": float(best_row.get("gross_max_drawdown", 0.0)),
            "win_rate": float(best_row.get("win_rate", 0.0)),
            "profit_factor": float(best_row.get("profit_factor", 1.0)),
            "avg_position_fraction": float(best_row.get("avg_position_fraction", 0.0)),
            "max_position_fraction": float(best_row.get("max_position_fraction", 0.0)),
            "avg_position_value": float(best_row.get("avg_position_value", 0.0)),
            "atr_fallback_trades": int(best_row.get("atr_fallback_trades", 0)),
            "num_trades": (
                int(best_row.get("num_trades", 0)) if pd.notna(best_row.get("num_trades")) else 0
            ),
            "deflated_sharpe_ratio": (
                float(best_row["dsr"]) if pd.notna(best_row.get("dsr")) else None
            ),
            "pbo": (float(best_row["pbo"]) if pd.notna(best_row.get("pbo")) else None),
            "avg_win": (
                float(best_row.get("avg_win", 0.0))
                if pd.notna(best_row.get("avg_win", 0.0))
                else 0.0
            ),
            "avg_loss": (
                float(best_row.get("avg_loss", 0.0))
                if pd.notna(best_row.get("avg_loss", 0.0))
                else 0.0
            ),
        },
    }

    # 既存の JSON ファイルを読み込み、統合
    # python/src/backtest/optimizer/persistence.py -> python/config
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "config",
    )
    ensure_dir(config_dir)
    json_path = os.path.join(config_dir, "optimal_params.json")

    all_params = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        except Exception as e:
            logger.warning(f"既存JSONの読み込みエラー（空として初期化）: {e}", exc_info=True)

    # マーケット・シンボルをキーに保存
    key = f"{market}_{symbol}"
    all_params[key] = optimal_param

    # JSON を保存
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_params, f, ensure_ascii=False, indent=2)

    return json_path


def get_optimal_params(
    market: str,
    symbol: str,
    json_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    保存された最適パラメータを JSON から読み込む。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        json_path: JSONファイルパス（Noneの場合は默认位置）

    Returns:
        最適パラメータ辞書、または見つからない場合は None
    """
    if json_path is None:
        # python/src/backtest/optimizer/persistence.py -> python/config
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "config",
        )
        json_path = os.path.join(config_dir, "optimal_params.json")

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
        key = f"{market}_{symbol}"
        return dict(all_params[key]) if key in all_params else None
    except Exception as e:
        logger.error(f"JSONの読み込みエラー: {e}", exc_info=True)
        return None
