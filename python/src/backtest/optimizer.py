"""
バックテスト最適化パイプラインサービス

閾値・ストップロス・テイクプロフィットのグリッドサーチを
Walk-Forward検証で実行し、最適パラメータを特定する。

run_backtest_optimize.py はこのモジュールの関数を呼び出すラッパーとして機能する。
"""

import itertools
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from src.backtest.pipeline import run_backtest_walk_forward
from src.utils.data_path_utils import ensure_dir, get_results_dir
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
    slippage: float = 0.0,
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
    # python/src/services/backtest_optimize_pipeline.py -> python/config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
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
        # python/src/services/backtest_optimize_pipeline.py -> python/config
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config"
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


def run_optimize_batch(
    tasks: list,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
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
    max_workers: int = 3,
    sort_by: str = "sharpe_ratio",
    skip_days: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    ウォッチリストの全銘柄に対してバックテスト最適化をバッチ実行する。

    tasks は呼び出し元（orchestration 等）で load_target_symbols() して渡す。

    フェーズ1: 最適化計算（並列） — DuckDB読み込みのみ・書き込みなし
    フェーズ2: 結果保存（逐次） — optimal_params.json の同時書き込み破損を防止

    Args:
        tasks: 対象銘柄タスクリスト（SymbolTask または dict）
        model_type: モデルタイプ
        ensemble: XGBoost+LightGBMアンサンブル予測を使用するか
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
        max_workers: 並列数
        sort_by: 最適パラメータ選定基準
        skip_days: 最終最適化からこの日数以内ならスキップ（None=常に実行）

    Returns:
        各銘柄の結果サマリー list[dict]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = tasks
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return []

    logger.info(f"全銘柄最適化バッチ開始: {len(symbols)}銘柄 / 並列数={max_workers}")

    def _optimize_task(task: Any) -> dict[str, Any]:
        market = getattr(task, "market", None) or task["market"]
        symbol = getattr(task, "symbol", None) or task["symbol"]

        # スキップ判定: 直近 skip_days 日以内に最適化済みならスキップ
        if skip_days is not None:
            existing = get_optimal_params(market, symbol)
            if existing and existing.get("timestamp"):
                try:
                    last_optimized = datetime.fromisoformat(existing["timestamp"])
                    if datetime.now() - last_optimized < timedelta(days=skip_days):
                        logger.info(
                            f"[{market}/{symbol}] スキップ"
                            f"（最終最適化: {last_optimized.strftime('%Y-%m-%d %H:%M')}"
                            f", {skip_days}日以内）"
                        )
                        return {"market": market, "symbol": symbol, "status": "skipped"}
                except ValueError:
                    pass  # timestamp パース失敗時は通常実行

        try:
            result_df = run_optimization(
                market=market,
                symbol=symbol,
                model_type=model_type,
                ensemble=ensemble,
                source=source,
                n_splits=n_splits,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                position_sizing=position_sizing,
                position_fraction=position_fraction,
                atr_risk_pcts=atr_risk_pcts,
                atr_multipliers=atr_multipliers,
                atr_min_fraction=atr_min_fraction,
                atr_max_fraction=atr_max_fraction,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                optimize_risk=optimize_risk,
            )
            return {"market": market, "symbol": symbol, "status": "success", "result_df": result_df}
        except Exception as e:
            logger.error(f"[{market}/{symbol}] 最適化エラー: {e}", exc_info=True)
            return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}

    # フェーズ1: 最適化計算（並列）
    optimize_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_optimize_task, task): task for task in symbols}
        for future in as_completed(futures):
            optimize_results.append(future.result())

    # フェーズ2: 結果保存（逐次 — JSON書き込みの競合を防止）
    summary = []
    success_count = 0
    error_count = 0
    skip_count = 0
    for res in optimize_results:
        market = res["market"]
        symbol = res["symbol"]
        if res["status"] == "skipped":
            summary.append({"market": market, "symbol": symbol, "status": "skipped"})
            skip_count += 1
            continue
        if res["status"] != "success":
            logger.error(f"[{market}/{symbol}] スキップ（最適化失敗）: {res.get('error')}")
            summary.append(
                {"market": market, "symbol": symbol, "status": "error", "error": res.get("error")}
            )
            error_count += 1
            continue

        result_df = res["result_df"]
        try:
            csv_path = save_optimization_results(result_df, market, symbol)
            json_path = save_optimal_params_json(result_df, market, symbol, sort_by=sort_by)
            logger.info(f"[{market}/{symbol}] 保存完了 CSV={csv_path} JSON={json_path}")
            summary.append({"market": market, "symbol": symbol, "status": "success"})
            success_count += 1
        except Exception as e:
            logger.error(f"[{market}/{symbol}] 保存エラー: {e}", exc_info=True)
            summary.append({"market": market, "symbol": symbol, "status": "error", "error": str(e)})
            error_count += 1

    logger.info(
        f"全銘柄最適化バッチ完了: 成功={success_count} / スキップ={skip_count} / エラー={error_count}"
    )
    return summary


# ---------------------------------------------------------------------------
# R-206: Optuna 自動ハイパーパラメータ探索
# ---------------------------------------------------------------------------


def run_optuna_optimization(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
    n_trials: int = 50,
    sort_by: str = "sharpe_ratio",
) -> pd.DataFrame:
    """
    Optuna TPE サンプラーによるハイパーパラメータ探索。

    グリッドサーチより少ない試行回数で高品質な最適値を見つける。
    結果は ``run_optimization`` と同形式の DataFrame で返すため、
    ``save_optimal_params_json`` にそのまま渡せる。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        n_trials: Optuna 試行回数（デフォルト 50）
        sort_by: 最適化指標（デフォルト "sharpe_ratio"）
        その他: ``run_optimization`` と同様

    Returns:
        各試行の結果 DataFrame（threshold / metrics 列を含む）
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna が未インストールです。pip install optuna を実行してください。")
        return pd.DataFrame()

    _minimize = sort_by in {"max_drawdown", "cost_impact_return", "cost_impact_cash"}

    def _objective(trial: "optuna.Trial") -> float:
        threshold = trial.suggest_float("threshold", 0.0, 0.02, step=0.001)
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
                ensemble=ensemble,
            )
            if wf_df is None or wf_df.empty or sort_by not in wf_df.columns:
                return float("-inf") if not _minimize else float("inf")
            val = float(wf_df[sort_by].mean())
            return -val if _minimize else val
        except Exception:
            return float("-inf") if not _minimize else float("inf")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    logger.info(f"[{market}/{symbol}] Optuna最適化開始: n_trials={n_trials}")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"[{market}/{symbol}] Optuna最適化完了: best_value={study.best_value:.4f}")

    # 結果を run_optimization 互換の DataFrame に変換
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        raw_metric = -trial.value if _minimize else trial.value
        row: Dict[str, Any] = dict(trial.params)
        row[sort_by] = raw_metric
        rows.append(row)

    return pd.DataFrame(rows)


def run_optuna_batch(
    tasks: list,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    n_trials: int = 50,
    max_workers: int = 3,
    sort_by: str = "sharpe_ratio",
) -> list[dict[str, Any]]:
    """
    ウォッチリスト全銘柄に Optuna 最適化をバッチ実行し、
    optimal_params.json を更新する（週次スケジューラから呼び出す）。

    tasks は呼び出し元（orchestration 等）で load_target_symbols() して渡す。

    Args:
        tasks: 対象銘柄タスクリスト（SymbolTask または dict）
        n_trials: 銘柄ごとの Optuna 試行回数
        max_workers: 並列数
        その他: ``run_optuna_optimization`` と同様

    Returns:
        各銘柄の結果サマリー list[dict[str, Any]]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = tasks
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return []

    logger.info(
        f"Optunaバッチ最適化開始: {len(symbols)}銘柄 / n_trials={n_trials} / 並列={max_workers}"
    )

    def _task(task: Any) -> dict[str, Any]:
        m = getattr(task, "market", None) or task["market"]
        s = getattr(task, "symbol", None) or task["symbol"]
        try:
            result_df = run_optuna_optimization(
                market=m,
                symbol=s,
                model_type=model_type,
                ensemble=ensemble,
                source=source,
                n_splits=n_splits,
                n_trials=n_trials,
                sort_by=sort_by,
            )
            return {"market": m, "symbol": s, "status": "success", "result_df": result_df}
        except Exception as e:
            logger.error(f"[{m}/{s}] Optuna最適化エラー: {e}", exc_info=True)
            return {"market": m, "symbol": s, "status": "error", "error": str(e)}

    # フェーズ1: 並列最適化
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, t): t for t in symbols}
        for future in as_completed(futures):
            results.append(future.result())

    # フェーズ2: 逐次保存
    summary, success, error = [], 0, 0
    for res in results:
        m, s = res["market"], res["symbol"]
        if res["status"] != "success":
            summary.append({"market": m, "symbol": s, "status": "error", "error": res.get("error")})
            error += 1
            continue
        try:
            save_optimal_params_json(res["result_df"], m, s, sort_by=sort_by)
            summary.append({"market": m, "symbol": s, "status": "success"})
            success += 1
        except Exception as e:
            logger.error(f"[{m}/{s}] 保存エラー: {e}", exc_info=True)
            summary.append({"market": m, "symbol": s, "status": "error", "error": str(e)})
            error += 1

    logger.info(f"Optunaバッチ最適化完了: 成功={success} / エラー={error}")
    return summary
