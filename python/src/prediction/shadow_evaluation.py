"""
シャドーモード A/B テスト評価パイプライン（R-207）

新旧モデルの予測精度（Sharpe / Hit Rate）を定量比較し、
本番切り替えの判断材料を提供する。

評価フロー:
1. prediction_accuracy から production / challenger 両バージョンの
   直近 evaluation_days 営業日分の予測精度データを取得する。
2. 銘柄ごとに Sharpe Ratio（予測変化率ベース）と Hit Rate を計算する。
3. 比較結果を experiment_runs テーブルに記録する（R-211 実験トラッキング連携）。
4. 判定: 全体 Hit Rate / Sharpe が challenger > production かどうかを返す。
"""

from __future__ import annotations

import math
import os
import shutil
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.data_path_utils import get_models_subdir
from src.utils.db.experiment import generate_run_id, save_experiment_run
from src.utils.db.prediction import load_prediction_accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# デフォルト評価期間（営業日数）
DEFAULT_EVALUATION_DAYS = 20


# ---------------------------------------------------------------------------
# 指標計算ヘルパー
# ---------------------------------------------------------------------------


def _compute_hit_rate(df: pd.DataFrame) -> Optional[float]:
    """direction_match 列から Hit Rate を計算する。"""
    if df.empty or "direction_match" not in df.columns:
        return None
    valid = df["direction_match"].dropna()
    if valid.empty:
        return None
    return float(valid.astype(float).mean())


def _compute_sharpe(df: pd.DataFrame) -> Optional[float]:
    """predicted_ratio 列から Sharpe Ratio を計算する（年率換算なし・簡易版）。

    シグナルとして predicted_ratio の符号で方向判断し、
    actual_ratio × sign(predicted_ratio) をリターン系列として使う。
    """
    if df.empty:
        return None
    required = {"predicted_ratio", "actual_ratio"}
    if not required.issubset(df.columns):
        return None

    valid = df[["predicted_ratio", "actual_ratio"]].dropna()
    if len(valid) < 2:
        return None

    # 予測方向に従った実現リターン（ゼロの場合は正方向として扱う）
    signs = np.where(valid["predicted_ratio"] >= 0, 1.0, -1.0)
    returns = valid["actual_ratio"] * signs
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    if std_ret == 0 or math.isnan(std_ret):
        return None
    return float(mean_ret / std_ret)


# ---------------------------------------------------------------------------
# 評価メイン関数
# ---------------------------------------------------------------------------


def evaluate_shadow_models(
    production_version: str = "production",
    challenger_version: str = "challenger",
    evaluation_days: int = DEFAULT_EVALUATION_DAYS,
    horizon: int = 1,
    market: Optional[str] = None,
    symbol: Optional[str] = None,
) -> dict:
    """
    production / challenger 両バージョンの予測精度を比較評価する。

    prediction_accuracy テーブルの直近 evaluation_days 件を使って
    Sharpe Ratio と Hit Rate を算出し、challenger が優れているかどうかを判定する。
    結果は experiment_runs テーブルに記録される。

    Args:
        production_version: 本番バージョンのラベル（prediction_accuracy の model_name に対応）
        challenger_version: チャレンジャーバージョンのラベル
        evaluation_days: 評価期間（直近 N 件の accuracy レコードを使用）
        horizon: 対象予測ホライズン（デフォルト 1）
        market: マーケットフィルタ（None なら全マーケット）
        symbol: 銘柄フィルタ（None なら全銘柄）

    Returns:
        dict:
            production_hit_rate: float | None
            production_sharpe: float | None
            challenger_hit_rate: float | None
            challenger_sharpe: float | None
            challenger_wins: bool  # True = challenger 切り替え推奨
            evaluated_at: str
            n_production: int
            n_challenger: int
    """
    logger.info(
        f"=== シャドーモード評価開始 "
        f"({production_version} vs {challenger_version}, "
        f"horizon={horizon}, evaluation_days={evaluation_days}) ==="
    )

    df_prod = load_prediction_accuracy(
        market=market,
        symbol=symbol,
        horizon=horizon,
        limit=evaluation_days * 100,  # 銘柄数 × evaluation_days を上限目安に取得
    )
    df_prod = df_prod[df_prod["model_name"] == production_version] if not df_prod.empty else df_prod

    df_chal = load_prediction_accuracy(
        market=market,
        symbol=symbol,
        horizon=horizon,
        limit=evaluation_days * 100,
    )
    df_chal = df_chal[df_chal["model_name"] == challenger_version] if not df_chal.empty else df_chal

    prod_hit = _compute_hit_rate(df_prod)
    prod_sharpe = _compute_sharpe(df_prod)
    chal_hit = _compute_hit_rate(df_chal)
    chal_sharpe = _compute_sharpe(df_chal)

    # 判定: Hit Rate と Sharpe の両方で challenger が上回れば切り替え推奨
    def _better(chal_val, prod_val) -> bool:
        if chal_val is None or prod_val is None:
            return False
        return chal_val > prod_val

    hit_better = _better(chal_hit, prod_hit)
    sharpe_better = _better(chal_sharpe, prod_sharpe)
    challenger_wins = hit_better and sharpe_better

    evaluated_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "production_hit_rate": prod_hit,
        "production_sharpe": prod_sharpe,
        "challenger_hit_rate": chal_hit,
        "challenger_sharpe": chal_sharpe,
        "challenger_wins": challenger_wins,
        "evaluated_at": evaluated_at,
        "n_production": len(df_prod),
        "n_challenger": len(df_chal),
    }

    logger.info(
        f"評価結果: production=hit_rate={prod_hit}, sharpe={prod_sharpe} / "
        f"challenger=hit_rate={chal_hit}, sharpe={chal_sharpe} / "
        f"challenger_wins={challenger_wins}"
    )

    # experiment_runs に記録（production / challenger それぞれ）
    _record_ab_result(
        version=production_version,
        hit_rate=prod_hit,
        sharpe=prod_sharpe,
        evaluated_at=evaluated_at,
        market=market or "all",
        symbol=symbol or "all",
        horizon=horizon,
        n_samples=len(df_prod),
        challenger_wins=challenger_wins,
        role="production",
    )
    _record_ab_result(
        version=challenger_version,
        hit_rate=chal_hit,
        sharpe=chal_sharpe,
        evaluated_at=evaluated_at,
        market=market or "all",
        symbol=symbol or "all",
        horizon=horizon,
        n_samples=len(df_chal),
        challenger_wins=challenger_wins,
        role="challenger",
    )

    return result


# ---------------------------------------------------------------------------
# 実験記録ヘルパー
# ---------------------------------------------------------------------------


def _record_ab_result(
    version: str,
    hit_rate: Optional[float],
    sharpe: Optional[float],
    evaluated_at: str,
    market: str,
    symbol: str,
    horizon: int,
    n_samples: int,
    challenger_wins: bool,
    role: str,
) -> None:
    """A/B テスト評価結果を experiment_runs テーブルに保存する。"""
    run_id = generate_run_id()
    params = {
        "role": role,
        "challenger_wins": challenger_wins,
        "sharpe_ratio": sharpe,
    }
    save_experiment_run(
        run_id=run_id,
        market=market,
        symbol=symbol,
        model_name=version,
        trained_at=evaluated_at,
        horizon=horizon,
        directional_accuracy=hit_rate,
        n_samples=n_samples,
        params=params,
    )
    logger.debug(
        f"experiment_run 記録: run_id={run_id} version={version} role={role} "
        f"hit_rate={hit_rate} sharpe={sharpe}"
    )


# ---------------------------------------------------------------------------
# 便利関数: shadow モード予測を実行して両バージョン保存
# ---------------------------------------------------------------------------


def run_shadow_prediction(
    predict_fn,
    production_version: str = "production",
    challenger_version: str = "challenger",
    challenger_predict_fn=None,
    mode: str = "unified",
) -> dict:
    """
    production / challenger 両モデルを並行実行し、それぞれ DB に保存する。

    prediction_pipeline の predict_all_* 関数を受け取って実行し、
    output_top_worst_results をシャドーモードで呼び出す。

    Args:
        predict_fn: 本番モデルの予測関数（引数なし callable → list[PredictionResult]）
        production_version: 本番バージョンラベル
        challenger_version: チャレンジャーバージョンラベル
        challenger_predict_fn: チャレンジャーモデルの予測関数
                               （None なら predict_fn を流用し版を変えて保存）
        mode: output_top_worst_results の mode 引数

    Returns:
        dict: production_count, challenger_count
    """
    from src.prediction.prediction_pipeline import output_top_worst_results

    logger.info(f"シャドー予測開始: {production_version} / {challenger_version}")

    # 本番予測
    prod_results = predict_fn()
    output_top_worst_results(
        prod_results,
        mode=mode,
        shadow_mode=True,
        model_version=production_version,
    )

    # チャレンジャー予測
    chal_fn = challenger_predict_fn if challenger_predict_fn is not None else predict_fn
    chal_results = chal_fn()
    output_top_worst_results(
        chal_results,
        mode=mode,
        shadow_mode=True,
        model_version=challenger_version,
    )

    return {
        "production_count": len(prod_results),
        "challenger_count": len(chal_results),
    }


# ---------------------------------------------------------------------------
# モデル昇格フロー
# ---------------------------------------------------------------------------

_PRODUCTION_MODEL_NAMES = ["StockXGBoostModel", "StockLightGBMModel"]
_CHALLENGER_MODEL_NAMES = ["ChallengerXGBoostModel", "ChallengerLightGBMModel"]


def promote_challenger_to_production(
    market: str,
    symbol: str,
    dry_run: bool = False,
) -> dict:
    """
    チャレンジャーモデルを本番モデルとして昇格させる（手動承認フロー）。

    ChallengerXGBoostModel / ChallengerLightGBMModel の joblib ファイルを
    StockXGBoostModel / StockLightGBMModel として上書きコピーする。
    evaluate_shadow_models() で challenger_wins=True を確認した後に呼び出すこと。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        dry_run: True の場合はファイル操作を行わずログのみ出力する

    Returns:
        dict:
            promoted: list[str] — 昇格に成功したモデル名のリスト
            skipped: list[str] — チャレンジャーファイルが存在しなかったモデル名
            dry_run: bool
    """
    model_dir = get_models_subdir(market, symbol)
    promoted = []
    skipped = []

    for challenger_name, production_name in zip(_CHALLENGER_MODEL_NAMES, _PRODUCTION_MODEL_NAMES):
        src = os.path.join(model_dir, f"{challenger_name}.joblib")
        dst = os.path.join(model_dir, f"{production_name}.joblib")

        if not os.path.exists(src):
            logger.warning(f"チャレンジャーモデルが見つかりません: {src}")
            skipped.append(challenger_name)
            continue

        if dry_run:
            logger.info(f"[dry_run] 昇格スキップ: {challenger_name} -> {production_name}")
        else:
            shutil.copy2(src, dst)
            logger.info(f"モデル昇格完了: {challenger_name} -> {production_name} ({market}/{symbol})")

        promoted.append(production_name)

    return {"promoted": promoted, "skipped": skipped, "dry_run": dry_run}
