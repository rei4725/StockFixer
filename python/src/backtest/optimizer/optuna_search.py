"""R-206: Optuna 自動ハイパーパラメータ探索。

TPE サンプラーで閾値を探索し、選択バイアス補正済みの DSR と PBO を
付与した run_optimization 互換の DataFrame を返す。
"""

import math
from typing import Any, Dict

import pandas as pd

from src.backtest.metrics import deflated_sharpe_ratio
from src.backtest.optimizer._pbo import _DSR_WARN_THRESHOLD, _compute_pbo_from_fold_returns
from src.backtest.pipeline import run_backtest_walk_forward
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            # DSR 算出用に最良試行の Sharpe と総取引回数（observations）を保持する。
            # DSR には非年率の取引単位 Sharpe を渡す（年率化済みだと飽和するため）。
            if "sharpe_per_trade" in wf_df.columns:
                trial.set_user_attr("sharpe", float(wf_df["sharpe_per_trade"].mean()))
            elif "sharpe_ratio" in wf_df.columns:
                trial.set_user_attr("sharpe", float(wf_df["sharpe_ratio"].mean()))
            if "num_trades" in wf_df.columns:
                trial.set_user_attr("num_trades", int(wf_df["num_trades"].sum()))
            # PBO 算出用に fold 別リターン系列（時系列順）を保持する
            if "total_return" in wf_df.columns:
                trial.set_user_attr(
                    "fold_returns",
                    pd.to_numeric(wf_df["total_return"], errors="coerce")
                    .fillna(0.0)
                    .astype(float)
                    .tolist(),
                )
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

    # 選択バイアス補正: 最良戦略を「完了試行数」で多重比較補正した DSR を算出する。
    # N 試行から最良 Sharpe を選ぶこと自体に上振れバイアスがあるため、DSR で割り引く。
    completed = [t for t in study.trials if t.value is not None]
    n_done = len(completed)
    best_trial = study.best_trial
    best_sharpe = float(best_trial.user_attrs.get("sharpe", study.best_value))
    best_nobs = int(best_trial.user_attrs.get("num_trades", 0))
    best_dsr = deflated_sharpe_ratio(best_sharpe, n_done, best_nobs)
    logger.info(
        "[%s/%s] DSR=%.3f (best_sharpe=%.3f, n_trials=%d, n_obs=%d)",
        market,
        symbol,
        best_dsr,
        best_sharpe,
        n_done,
        best_nobs,
    )
    if best_nobs > 0 and best_dsr < _DSR_WARN_THRESHOLD:
        logger.warning(
            "[%s/%s] DSR=%.3f < %.2f: 多重比較を除くと有意でない可能性（過学習の疑い）",
            market,
            symbol,
            best_dsr,
            _DSR_WARN_THRESHOLD,
        )

    # 過学習ガード: 全試行の fold リターン行列から PBO を算出する
    pbo = _compute_pbo_from_fold_returns(
        [t.user_attrs["fold_returns"] for t in completed if "fold_returns" in t.user_attrs],
        market,
        symbol,
    )

    # 結果を run_optimization 互換の DataFrame に変換
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        raw_metric = -trial.value if _minimize else trial.value
        row: Dict[str, Any] = dict(trial.params)
        row[sort_by] = raw_metric
        row["num_trades"] = int(trial.user_attrs.get("num_trades", 0))
        # DSR は最良戦略に対してのみ意味を持つため最良行にだけ付与する
        if trial.number == best_trial.number:
            row["dsr"] = best_dsr
        # PBO は探索全体に対する指標のため全行に付与する
        if not math.isnan(pbo):
            row["pbo"] = pbo
        rows.append(row)

    return pd.DataFrame(rows)
