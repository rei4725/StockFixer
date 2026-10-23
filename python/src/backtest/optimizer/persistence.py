"""最適化結果の表示・永続化。

結果サマリーのコンソール表示、CSV 保存、最適パラメータの JSON 統合保存、
保存済み最適パラメータの読み込みを担う。
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, NoReturn, Optional

import pandas as pd
from filelock import FileLock

from src.utils.data_path_utils import ensure_dir
from src.utils.logger import get_logger
from src.utils.results_io import save_result_csvs

logger = get_logger(__name__)


class OptimalParamsCorruptedError(RuntimeError):
    """既存の optimal_params.json が壊れていることを表す。

    保存では退避して止めたとき、読み出しでは在るのに読めない・最上位が dict でないときに投げる。
    """


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
    return save_result_csvs({"optimize": result_df}, f"optimize/{market}_{symbol}")["optimize"]


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

    # 別のプロセス（手での実行とスケジューラ）が足した銘柄を消さないよう、ロックの内で読み直してから足す
    with FileLock(f"{json_path}.lock"):
        all_params = _load_params_for_update(json_path)

        # マーケット・シンボルをキーに保存
        key = f"{market}_{symbol}"
        all_params[key] = optimal_param

        _write_params_atomically(json_path, all_params)

    return json_path


def _load_params_for_update(json_path: str) -> Dict[str, Any]:
    """既存の JSON を読む（ロックの内で呼ぶ）。壊れていれば退避して例外を投げる。

    空として続けると、1銘柄だけのファイルで全銘柄の閾値と metrics を消すため。
    """
    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        _quarantine_params_file(json_path, str(e))
    except OSError:
        logger.error(
            f"最適パラメータJSONの読み込みに失敗したため保存を止めます: {json_path}", exc_info=True
        )
        raise

    if not isinstance(all_params, dict):
        _quarantine_params_file(json_path, f"最上位が dict ではない: {type(all_params).__name__}")
    return all_params


def _quarantine_params_file(json_path: str, detail: str) -> NoReturn:
    """壊れた JSON を隣へ日時つきで退避し、保存を止める。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine_path = f"{json_path}.corrupt-{stamp}"
    try:
        os.replace(json_path, quarantine_path)
    except OSError:
        logger.error(f"壊れた最適パラメータJSONの退避に失敗: {json_path}", exc_info=True)
        raise OptimalParamsCorruptedError(f"壊れた最適パラメータJSON（退避失敗）: {json_path}")
    logger.error(
        "最適パラメータJSONが読めないため退避し、保存を止めます"
        f": {detail} 退避先={quarantine_path}"
    )
    raise OptimalParamsCorruptedError(f"壊れた最適パラメータJSONを退避しました: {quarantine_path}")


def _write_params_atomically(json_path: str, all_params: Dict[str, Any]) -> None:
    """同じフォルダの一時ファイルに書いてから置き換える（ロックの内で呼ぶ）。"""
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(json_path)),
        prefix=f"{os.path.basename(json_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(all_params, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, json_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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
        最適パラメータ辞書、またはファイル未存在・キー未登録の場合は None

    Raises:
        OptimalParamsCorruptedError: ファイルが在るのに読めない・JSON が壊れている・
            最上位が dict でない場合（未登録と区別し、良い結果を上書きしないため）
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
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        logger.error(f"最適パラメータJSONが在るのに読めません: {json_path}", exc_info=True)
        raise OptimalParamsCorruptedError(
            f"最適パラメータJSONを読めません: {json_path}: {e}"
        ) from e
    if not isinstance(all_params, dict):
        detail = f"最上位が dict ではない: {type(all_params).__name__}"
        logger.error(f"最適パラメータJSONが壊れています: {json_path} {detail}")
        raise OptimalParamsCorruptedError(f"最適パラメータJSONが壊れています: {json_path} {detail}")
    key = f"{market}_{symbol}"
    return dict(all_params[key]) if key in all_params else None
