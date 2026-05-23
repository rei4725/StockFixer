"""
モデル昇格ゲート（R-302）

shadow モデルの Walk-Forward 成績が current モデルを上回った場合に
昇格可否を判定し、根拠をログ・JSON に記録する。

昇格判定基準（全条件を満たすこと）:
    1. Net Return: shadow > current（Walk-Forward fold 平均）
    2. MDD:        |shadow_mdd| < |current_mdd|（ドローダウン改善）
    3. Sharpe:     shadow >= current * 0.95（Sharpe は 5% 劣化まで許容）
    4. Hit Rate:   shadow >= 0.50（最低ライン）
    5. slippage 乖離: shadow の avg_abs_diff_ratio < 0.02（2% 以内）

手動承認フラグ: require_manual_approval=True（デフォルト）の場合、
判定結果を JSON に保存するが実際のファイル操作は行わない。

A/B テスト連携:
    evaluate_promotion() は run_id を引数に取ることができ、
    experiment_runs テーブルから対象ランの情報を取得して判断根拠に含める。
    run_id が省略された場合は最新ランを使用する。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.data_path_utils import get_results_dir
from src.utils.db import load_prediction_accuracy
from src.utils.db.experiment import load_experiment_runs
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 昇格基準の定数
# ---------------------------------------------------------------------------

# Net Return: shadow が current を上回ること（絶対比較）
_NET_RETURN_STRICT = True  # shadow > current（等号は不可）

# MDD: shadow の絶対値が current の絶対値を下回ること
_MDD_STRICT = True  # |shadow| < |current|（等号は不可）

# Sharpe: shadow >= current * (1 - _SHARPE_TOLERANCE)
_SHARPE_TOLERANCE = 0.05  # 5% 劣化まで許容

# Hit Rate: shadow の方向正解率が _HIT_RATE_FLOOR 以上
_HIT_RATE_FLOOR = 0.50

# Slippage 乖離: avg_abs_diff_ratio が _SLIPPAGE_THRESHOLD 未満
_SLIPPAGE_THRESHOLD = 0.02  # 2%


# ---------------------------------------------------------------------------
# 戻り値型
# ---------------------------------------------------------------------------


@dataclass
class PromotionResult:
    """昇格判定結果。

    evaluate_promotion() の戻り値として使用する。
    criteria フィールドには各基準の詳細（actual / threshold / passed）を格納する。

    Attributes:
        shadow_model: shadow モデル名
        current_model: current モデル名
        evaluated_at: 評価実行日時（ISO 8601）
        eligible: 全昇格基準を満たしているか
        require_manual_approval: True の場合は手動承認が別途必要
        criteria: 各基準の詳細 dict
            {
              "net_return":  {"shadow": float, "current": float, "passed": bool},
              "mdd":         {"shadow": float, "current": float, "passed": bool},
              "sharpe":      {"shadow": float, "current": float,
                               "threshold": float, "passed": bool},
              "hit_rate":    {"shadow": float, "floor": float, "passed": bool},
              "slippage":    {"shadow": float, "threshold": float, "passed": bool},
            }
        reason: 判定理由のサマリー文（1 行）
        run_ids: 判断に使用した run_id のリスト（A/B テスト連携）
        wf_source_file: 参照した Walk-Forward CSV ファイル名（デバッグ用）
    """

    shadow_model: str
    current_model: str
    evaluated_at: str
    eligible: bool
    require_manual_approval: bool
    criteria: dict
    reason: str
    run_ids: list[str] = field(default_factory=list)
    wf_source_file: Optional[str] = None


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _load_latest_wf_summary() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """最新の Walk-Forward サマリー CSV を (DataFrame, ファイル名) で返す。"""
    report_dir = Path(get_results_dir()) / "backtest" / "walk_forward_reports"
    candidates = sorted(report_dir.glob("wf_summary_*.csv"), reverse=True)
    if not candidates:
        logger.warning("Walk-Forward レポート CSV が見つかりません")
        return None, None
    path = candidates[0]
    try:
        df = pd.read_csv(path)
        logger.info(f"WF サマリー読み込み: {path.name} ({len(df)} 銘柄)")
        return df, path.name
    except Exception as e:
        logger.error(f"WF サマリー読み込み失敗: {e}", exc_info=True)
        return None, None


def _wf_mean(df: pd.DataFrame, model_name: str, col: str) -> Optional[float]:
    """Walk-Forward サマリーから指定モデルの指定列の平均値を返す。

    model_name 列が存在しない場合は全行を対象とする（単一モデルのレポート）。
    """
    work = df.copy()
    if "model_name" in work.columns:
        mask = work["model_name"] == model_name
        work = work[mask]
    if work.empty or col not in work.columns:
        return None
    val = pd.to_numeric(work[col], errors="coerce").mean()
    return float(val) if not pd.isna(val) else None


def _compute_hit_rate(model_name: str, horizon: int) -> Optional[float]:
    """prediction_accuracy テーブルから指定モデルの方向正解率を返す。"""
    df = load_prediction_accuracy(horizon=horizon, limit=5000)
    if df.empty or "direction_match" not in df.columns:
        return None
    if "model_name" in df.columns:
        df = df[df["model_name"] == model_name]
    if df.empty:
        return None
    return float(df["direction_match"].astype(float).mean())


def _compute_slippage(model_name: str, horizon: int) -> Optional[float]:
    """prediction_accuracy テーブルから avg_abs_diff_ratio を近似計算する。

    prediction_accuracy には price 差分の絶対値を集計した列はないため、
    predicted_ratio と actual_ratio の絶対乖離の平均を代用する。
    """
    df = load_prediction_accuracy(horizon=horizon, limit=5000)
    if df.empty:
        return None
    if "model_name" in df.columns:
        df = df[df["model_name"] == model_name]
    if df.empty or "predicted_ratio" not in df.columns or "actual_ratio" not in df.columns:
        return None
    pred = pd.to_numeric(df["predicted_ratio"], errors="coerce")
    actual = pd.to_numeric(df["actual_ratio"], errors="coerce")
    diff = (pred - actual).abs()
    val = diff.mean()
    return float(val) if not pd.isna(val) else None


def _collect_run_ids(shadow_model: str, current_model: str) -> list[str]:
    """experiment_runs テーブルから両モデルの最新 run_id を収集する。"""
    run_ids: list[str] = []
    for model_name in [shadow_model, current_model]:
        df = load_experiment_runs(model_name=model_name, limit=5)
        if not df.empty and "run_id" in df.columns:
            run_ids.extend(df["run_id"].tolist()[:3])
    return run_ids


# ---------------------------------------------------------------------------
# パブリック API
# ---------------------------------------------------------------------------


def _log_promotion_to_mlflow(result: PromotionResult) -> None:
    """昇格判定結果を MLflow に記録する。失敗しても例外を伝播させない。"""
    try:
        import mlflow

        mlflow.set_experiment("StockFixer/promotion")
        with mlflow.start_run(run_name=f"promotion_{result.evaluated_at}"):
            mlflow.set_tags(
                {
                    "shadow_model": result.shadow_model,
                    "current_model": result.current_model,
                    "eligible": str(result.eligible),
                    "reason": result.reason,
                    "require_manual_approval": str(result.require_manual_approval),
                }
            )
            for criterion_name, criterion_data in result.criteria.items():
                mlflow.log_metric(
                    f"{criterion_name}_passed",
                    float(criterion_data.get("passed", False)),
                )
            if result.run_ids:
                mlflow.set_tag("db_run_ids", ",".join(result.run_ids[:5]))
    except Exception as e:
        logger.debug("MLflow昇格記録スキップ: %s", e)


def evaluate_promotion(
    shadow_model_name: str,
    current_model_name: str,
    horizon: int = 1,
    require_manual_approval: bool = True,
    shadow_run_id: Optional[str] = None,
    current_run_id: Optional[str] = None,
) -> PromotionResult:
    """shadow モデルの昇格可否を判定する。

    Walk-Forward 成績（Net Return / MDD / Sharpe）は最新の
    wf_summary_*.csv から取得する。model_name 列が CSV に存在する場合は
    それでフィルタし、ない場合は全銘柄の平均を両モデルで共有する。

    Hit Rate と Slippage は prediction_accuracy テーブルから取得する。

    Args:
        shadow_model_name: 評価対象の shadow モデル名
        current_model_name: 比較基準の current モデル名
        horizon: 予測ホライズン（日）。prediction_accuracy のフィルタに使用
        require_manual_approval: True の場合、eligible=True でも手動承認が必要
        shadow_run_id: shadow モデルの run_id（省略時は最新を使用）
        current_run_id: current モデルの run_id（省略時は最新を使用）

    Returns:
        PromotionResult: 判定結果（eligible + criteria + reason）
    """
    evaluated_at = datetime.now().isoformat()
    logger.info(
        f"昇格ゲート評価開始: shadow={shadow_model_name} current={current_model_name} "
        f"horizon={horizon}"
    )

    # --- Walk-Forward 成績 ---
    wf_df, wf_filename = _load_latest_wf_summary()

    if wf_df is not None:
        shadow_net_return = _wf_mean(wf_df, shadow_model_name, "total_return")
        current_net_return = _wf_mean(wf_df, current_model_name, "total_return")
        shadow_mdd = _wf_mean(wf_df, shadow_model_name, "max_drawdown")
        current_mdd = _wf_mean(wf_df, current_model_name, "max_drawdown")
        shadow_sharpe = _wf_mean(wf_df, shadow_model_name, "sharpe_ratio")
        current_sharpe = _wf_mean(wf_df, current_model_name, "sharpe_ratio")
    else:
        shadow_net_return = None
        current_net_return = None
        shadow_mdd = None
        current_mdd = None
        shadow_sharpe = None
        current_sharpe = None

    # --- prediction_accuracy 成績 ---
    shadow_hit_rate = _compute_hit_rate(shadow_model_name, horizon)
    shadow_slippage = _compute_slippage(shadow_model_name, horizon)

    # --- 各基準の判定 ---
    # 1. Net Return
    if shadow_net_return is not None and current_net_return is not None:
        net_return_passed = shadow_net_return > current_net_return
    else:
        net_return_passed = False
    criteria_net_return: dict = {
        "shadow": shadow_net_return,
        "current": current_net_return,
        "passed": net_return_passed,
    }

    # 2. MDD（絶対値が小さいほど良い）
    if shadow_mdd is not None and current_mdd is not None:
        mdd_passed = abs(shadow_mdd) < abs(current_mdd)
    else:
        mdd_passed = False
    criteria_mdd: dict = {
        "shadow_abs": abs(shadow_mdd) if shadow_mdd is not None else None,
        "current_abs": abs(current_mdd) if current_mdd is not None else None,
        "passed": mdd_passed,
    }

    # 3. Sharpe（5% 劣化まで許容）
    if shadow_sharpe is not None and current_sharpe is not None:
        sharpe_threshold = current_sharpe * (1.0 - _SHARPE_TOLERANCE)
        sharpe_passed = shadow_sharpe >= sharpe_threshold
    else:
        sharpe_passed = False
        sharpe_threshold = None
    criteria_sharpe: dict = {
        "shadow": shadow_sharpe,
        "current": current_sharpe,
        "threshold": sharpe_threshold,
        "passed": sharpe_passed,
    }

    # 4. Hit Rate（方向正解率 >= 0.50）
    if shadow_hit_rate is not None:
        hit_rate_passed = shadow_hit_rate >= _HIT_RATE_FLOOR
    else:
        hit_rate_passed = False
    criteria_hit_rate: dict = {
        "shadow": shadow_hit_rate,
        "floor": _HIT_RATE_FLOOR,
        "passed": hit_rate_passed,
    }

    # 5. Slippage 乖離（avg_abs_diff_ratio < 0.02）
    if shadow_slippage is not None:
        slippage_passed = shadow_slippage < _SLIPPAGE_THRESHOLD
    else:
        slippage_passed = False
    criteria_slippage: dict = {
        "shadow": shadow_slippage,
        "threshold": _SLIPPAGE_THRESHOLD,
        "passed": slippage_passed,
    }

    criteria = {
        "net_return": criteria_net_return,
        "mdd": criteria_mdd,
        "sharpe": criteria_sharpe,
        "hit_rate": criteria_hit_rate,
        "slippage": criteria_slippage,
    }

    eligible = all(
        [
            net_return_passed,
            mdd_passed,
            sharpe_passed,
            hit_rate_passed,
            slippage_passed,
        ]
    )

    # --- 判定理由サマリー ---
    failed_names = [name for name, c in criteria.items() if not c["passed"]]
    if eligible:
        if require_manual_approval:
            reason = (
                f"全基準クリア（{shadow_model_name} は {current_model_name} を上回る）。"
                "手動承認待ち。"
            )
        else:
            reason = (
                f"全基準クリア（{shadow_model_name} は {current_model_name} を上回る）。"
                "自動昇格可能。"
            )
    else:
        reason = (
            f"昇格基準未達: {', '.join(failed_names)} が条件を満たしていない。"
            f"shadow={shadow_model_name} current={current_model_name}"
        )

    # --- A/B テスト run_id 収集 ---
    run_ids = _collect_run_ids(shadow_model_name, current_model_name)
    if shadow_run_id and shadow_run_id not in run_ids:
        run_ids.insert(0, shadow_run_id)
    if current_run_id and current_run_id not in run_ids:
        run_ids.insert(0, current_run_id)

    result = PromotionResult(
        shadow_model=shadow_model_name,
        current_model=current_model_name,
        evaluated_at=evaluated_at,
        eligible=eligible,
        require_manual_approval=require_manual_approval,
        criteria=criteria,
        reason=reason,
        run_ids=run_ids,
        wf_source_file=wf_filename,
    )

    logger.info(
        f"昇格ゲート判定: eligible={eligible} reason={reason} "
        f"net_return={net_return_passed} mdd={mdd_passed} "
        f"sharpe={sharpe_passed} hit_rate={hit_rate_passed} slippage={slippage_passed}"
    )
    _log_promotion_to_mlflow(result)
    return result


def save_promotion_result(result: PromotionResult) -> str:
    """判定結果を JSON ファイルに保存し、ファイルパスを返す。

    保存先: results/promotion/YYYYMMDD_HHMMSS_promotion_result.json

    Args:
        result: evaluate_promotion() の戻り値

    Returns:
        保存したファイルの絶対パス文字列
    """
    promotion_dir = Path(get_results_dir()) / "promotion"
    promotion_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_promotion_result.json"
    dest = promotion_dir / filename

    payload = asdict(result)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"昇格判定結果を保存: {dest}")
    return str(dest)
