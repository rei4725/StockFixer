"""
シャドーモード A/B テスト評価パイプライン（R-207）

新旧モデルの予測精度（Sharpe / Hit Rate）を定量比較し、
本番切り替えの判断材料を提供する。

評価フロー:
1. prediction_accuracy から production / challenger 両バージョンの
   直近 evaluation_days 営業日分の予測精度データを取得する。
2. 銘柄ごとに Sharpe Ratio（予測変化率ベース）と Hit Rate を計算する。
3. 比較結果を experiment_runs テーブルに記録する（R-211 実験トラッキング連携）。
4. 判定: 同じ窓・同じ銘柄の日ごとの順位相関（daily_rank_corr）が challenger > production で、
   全体 Hit Rate / Sharpe が challenger >= production かどうかを返す（#1985）。
   challenger の順位相関が None（定数の予想・行なし）なら勝ちにしない。production の None も
   勝ちにしないが、production が日ごとに定数のときだけは並べの腕が無いとして 0 と比べる。
   銘柄が1つの比べは順位相関を測れないので、前の条件（Hit Rate と Sharpe の両方で上回る）で決める。
   使った条件の名前を win_rule に返す。
   challenger の予想が全銘柄でほぼ同じ値なら勝ちにせず、負けと分けて印を付ける（#1602）。
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.prediction.baseline_metrics import baseline_metrics
from src.prediction.challenger_window import (
    CHALLENGER_LABEL,
    _day_series,
    filter_comparable_challenger,
    filter_to_days,
)
from src.prediction.constant_prediction import is_constant_prediction
from src.prediction.db import load_prediction_accuracy
from src.utils.db.experiment import generate_run_id, save_experiment_run
from src.utils.logger import get_logger

logger = get_logger(__name__)

# デフォルト評価期間（営業日数）
DEFAULT_EVALUATION_DAYS = 20

# 勝ちの条件の版。週の記録に書き、物差しを変えた前後の週を見分ける（#1985）。
# 前の版 "hit_rate_and_sharpe_better" は、Hit Rate と Sharpe の両方で上回れば勝ちだった。
# 次の版 "daily_rank_corr_better_hit_sharpe_not_worse" は、順位相関を市場と取引日でなく
# 予想の回（predicted_at）ごとに束ねていた（#2072）
SHADOW_WIN_RULE = "market_day_rank_corr_better_hit_sharpe_not_worse"
# 銘柄が1つの比べ（symbol を絞った評価など）は順位相関を測れないので、前の条件で決める
SINGLE_SYMBOL_WIN_RULE = "hit_rate_and_sharpe_better"


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


def _better(chal_val: Optional[float], prod_val: Optional[float]) -> bool:
    if chal_val is None or prod_val is None:
        return False
    return chal_val > prod_val


def _not_worse(chal_val: Optional[float], prod_val: Optional[float]) -> bool:
    if chal_val is None or prod_val is None:
        return False
    return chal_val >= prod_val


def _single_symbol(*frames: pd.DataFrame) -> bool:
    """どの表も (market, symbol) が1つ以下なら True（銘柄の間の順位相関を測れない比べ）。"""
    for df in frames:
        keys = [c for c in ("market", "symbol") if c in df.columns]
        if df.empty:
            continue
        if not keys or len(df[keys].drop_duplicates()) > 1:
            return False
    return True


def _with_market_day(df: pd.DataFrame) -> pd.DataFrame:
    """日ごとの物差しの束ねの鍵（市場と取引日）を date の列に付けた写しを返す（#2072）。

    1回の予想は jp と us を同じ predicted_at で書くので、回の時刻で束ねると市場の間の水準の差を
    銘柄の並べの腕と取り違える。取引日の決め方は影の比べの窓（challenger_window）と同じ。
    """
    if df.empty or "predicted_at" not in df.columns:
        return df
    days = _day_series(df)
    if "market" in df.columns:
        days = df["market"].astype(str) + ":" + days
    return df.assign(date=days)


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
    raise_on_error: bool = False,
) -> dict:
    """
    production / challenger 両バージョンの予測精度を比較評価する。

    prediction_accuracy テーブルのモデルごとの直近 evaluation_days 日分を使って
    Sharpe Ratio と Hit Rate を算出し、challenger が優れているかどうかを判定する。
    結果は experiment_runs テーブルに記録される。

    Args:
        production_version: 本番バージョンのラベル（prediction_accuracy の model_name に対応）
        challenger_version: チャレンジャーバージョンのラベル
        evaluation_days: 評価期間（モデルごとに predicted_at の日付の直近 N 日分を全部使う）
        horizon: 対象予測ホライズン（デフォルト 1）
        market: マーケットフィルタ（None なら全マーケット）
        symbol: 銘柄フィルタ（None なら全銘柄）
        raise_on_error: True なら prediction_accuracy の読み込みの例外を投げ、
            DB が読めない週を「challenger は勝たなかった」にしない（週次学習の道。#1295）

    Returns:
        dict:
            production_hit_rate: float | None
            production_sharpe: float | None
            challenger_hit_rate: float | None
            challenger_sharpe: float | None
            production_daily_rank_corr: float | None  # 同じ市場・取引日の銘柄の間の順位相関の平均
            challenger_daily_rank_corr: float | None
            win_rule: str  # 勝ちの条件の版（SHADOW_WIN_RULE）
            challenger_wins: bool  # True = challenger 切り替え推奨
            challenger_prediction_std: float | None  # 同じ市場・取引日の予想の標準偏差（#1602）
            challenger_constant: bool  # True = 全銘柄がほぼ同じ値。指標によらず勝ちにしない
            evaluated_at: str
            n_production: int
            n_challenger: int
            以下は challenger_version が challenger のラベルのときだけ（#1327）:
            challenger_trained_at: str | None  # 今の challenger のファイルの更新時刻
            production_changed_at: str | None  # production が最後に変わった時刻（#1559）
            window_days: int  # 今の challenger と今の production の予想がそろう日数

    challenger のラベルの行は、今の challenger のファイルの更新時刻と production が最後に変わった
    時刻の遅い方より後の predicted_at に絞り、production もその後の同じ日だけで比べる。
    ファイルが無いか0件なら challenger は「データなし」になる。
    """
    logger.info(
        f"=== シャドーモード評価開始 "
        f"({production_version} vs {challenger_version}, "
        f"horizon={horizon}, evaluation_days={evaluation_days}) ==="
    )

    # モデルごとに直近 evaluation_days 日分の行を全部読む（全モデルを行数で切らない。#958）
    df_prod = load_prediction_accuracy(
        market=market,
        symbol=symbol,
        horizon=horizon,
        limit=0,
        model_name=production_version,
        days=evaluation_days,
        raise_on_error=raise_on_error,
    )
    df_prod = df_prod[df_prod["model_name"] == production_version] if not df_prod.empty else df_prod

    df_chal = load_prediction_accuracy(
        market=market,
        symbol=symbol,
        horizon=horizon,
        limit=0,
        model_name=challenger_version,
        days=evaluation_days,
        raise_on_error=raise_on_error,
    )
    df_chal = df_chal[df_chal["model_name"] == challenger_version] if not df_chal.empty else df_chal

    # challenger のラベルは毎週の版が混ざるので、今の版の予想の日だけで両方を比べる（#1327）。
    # production のラベルも入れ替えで版が混ざるので、変わり目より後の行だけにする（#1559）
    window_info: dict = {}
    if challenger_version == CHALLENGER_LABEL:
        df_chal, window = filter_comparable_challenger(df_chal)
        df_prod = filter_to_days(df_prod, window)
        window_info = {
            "challenger_trained_at": window.as_dict()["challenger_trained_at"],
            "production_changed_at": window.as_dict()["production_changed_at"],
            "window_days": len(window.days),
        }

    prod_hit = _compute_hit_rate(df_prod)
    prod_sharpe = _compute_sharpe(df_prod)
    chal_hit = _compute_hit_rate(df_chal)
    chal_sharpe = _compute_sharpe(df_chal)

    chal_base = baseline_metrics(_with_market_day(df_chal))
    prod_base = baseline_metrics(_with_market_day(df_prod))
    prod_rank_corr = prod_base["daily_rank_corr"]
    chal_rank_corr = chal_base["daily_rank_corr"]
    win_rule = SINGLE_SYMBOL_WIN_RULE if _single_symbol(df_prod, df_chal) else SHADOW_WIN_RULE
    if win_rule == SHADOW_WIN_RULE:
        # 向きの的中率と符号の Sharpe は、両方が「上がる」ばかりだと窓の相場の向きでそろって
        # 動く。劣らないことだけを求め、勝ちは同じ窓・同じ銘柄の日ごとの順位相関で決める（#1985）。
        # 日ごとに定数の production は並べの腕が無いので順位相関を 0 と比べ、
        # 行が無い・日の列が無いなど測れない None は勝ちにしない
        prod_rank_for_compare = (
            0.0
            if prod_rank_corr is None and is_constant_prediction(prod_base["prediction_std"])
            else prod_rank_corr
        )
        metric_wins = (
            _not_worse(chal_hit, prod_hit)
            and _not_worse(chal_sharpe, prod_sharpe)
            and _better(chal_rank_corr, prod_rank_for_compare)
        )
    else:
        # 銘柄が1つの比べでは銘柄の間の並べを測れないので、前の条件で決める
        metric_wins = _better(chal_hit, prod_hit) and _better(chal_sharpe, prod_sharpe)
    # 定数の予想の的中率は上がる割合そのものなので、比べに勝たせない（#1602）
    chal_std = chal_base["prediction_std"]
    chal_constant = is_constant_prediction(chal_std)
    challenger_wins = metric_wins and not chal_constant

    evaluated_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 順位相関と勝ちの条件の版は、比べる行があるときだけ載せる（週の記録の欄は常にある）
    rank_info: dict = (
        {
            "production_daily_rank_corr": prod_rank_corr,
            "challenger_daily_rank_corr": chal_rank_corr,
            "win_rule": win_rule,
        }
        if not (df_prod.empty and df_chal.empty)
        else {}
    )
    result = {
        "production_hit_rate": prod_hit,
        "production_sharpe": prod_sharpe,
        "challenger_hit_rate": chal_hit,
        "challenger_sharpe": chal_sharpe,
        **rank_info,
        "challenger_wins": challenger_wins,
        "challenger_prediction_std": chal_std,
        "challenger_constant": chal_constant,
        "evaluated_at": evaluated_at,
        "n_production": len(df_prod),
        "n_challenger": len(df_chal),
        **window_info,
    }

    logger.info(
        f"評価結果: production=hit_rate={prod_hit}, sharpe={prod_sharpe} / "
        f"challenger=hit_rate={chal_hit}, sharpe={chal_sharpe} / "
        f"daily_rank_corr=production {prod_rank_corr}, challenger {chal_rank_corr} / "
        f"challenger_wins={challenger_wins} / "
        f"challenger_constant={chal_constant} std={chal_std} / "
        f"日数={window_info.get('window_days')} 件数=production {len(df_prod)}, "
        f"challenger {len(df_chal)}"
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
        challenger_constant=chal_constant,
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
    challenger_constant: bool = False,
) -> None:
    """A/B テスト評価結果を experiment_runs テーブルに保存する。"""
    run_id = generate_run_id()
    params = {
        "role": role,
        "challenger_wins": challenger_wins,
        "sharpe_ratio": sharpe,
        "challenger_constant": challenger_constant,
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
# Unified モデル専用: Challenger 予測・昇格フロー
# ---------------------------------------------------------------------------

_UNIFIED_PRODUCTION_NAMES = [
    "UnifiedStockXGBoost",
    "UnifiedStockLightGBM",
    "UnifiedStockTransformer",
]
_UNIFIED_CHALLENGER_NAMES = [
    "UnifiedStockXGBoost_challenger",
    "UnifiedStockLightGBM_challenger",
    "UnifiedStockTransformer_challenger",
]


def available_unified_challengers() -> list[str]:
    """ファイルが在る challenger unified モデルの名前を返す（一部欠損は許容）。"""
    from src.utils.data_path_utils import get_models_dir

    unified_dir = os.path.join(get_models_dir(), "unified")
    return [
        name
        for name in _UNIFIED_CHALLENGER_NAMES
        if os.path.exists(os.path.join(unified_dir, f"{name}.joblib"))
    ]


def predict_with_challenger_unified(
    on_error: Callable[[str, str], None] | None = None,
) -> list:
    """Challenger unified モデルで全銘柄予測を実行し PredictionResult リストを返す。

    UnifiedStockXGBoost_challenger / UnifiedStockLightGBM_challenger を使用する。
    モデルファイルが存在しない場合は空リストを返す。

    Args:
        on_error: 例外で予想を出せなかった銘柄ごとに (market, symbol) で呼ぶ（#1139）

    Returns:
        list[PredictionResult]: 各銘柄の予測結果
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.prediction.predict_unified import get_cached_model, predict_with_unified_model
    from src.prediction.prediction_run_stats import collect_prediction_run_stats
    from src.prediction.unified_run_summary import log_unified_prediction_summary
    from src.utils.db import get_active_symbols

    available_challengers = available_unified_challengers()
    missing = [n for n in _UNIFIED_CHALLENGER_NAMES if n not in available_challengers]
    if missing:
        logger.info(f"一部の Challenger モデルが見つかりません（スキップ）: {missing}")
    if not available_challengers:
        logger.info("利用可能な Challenger モデルが見つかりません（スキップ）")
        return []

    # モデルをキャッシュに事前ロード
    for name in available_challengers:
        get_cached_model(name)

    all_keys = get_active_symbols()
    logger.info(f"Challenger shadow 予測開始: {len(all_keys)} 銘柄")

    # 予想を出せなかった銘柄（list.append はスレッドをまたいでも安全）
    stale_keys: list[tuple[str, str]] = []
    error_keys: list[tuple[str, str]] = []

    def on_stale(market: str, symbol: str) -> None:
        stale_keys.append((market, symbol))

    def _predict(args):
        market, symbol = args
        try:
            return predict_with_unified_model(
                market, symbol, model_types=available_challengers, on_stale=on_stale
            )
        except Exception:
            logger.warning(
                "Challenger 予測スキップ: market=%s symbol=%s", market, symbol, exc_info=True
            )
            error_keys.append((market, symbol))
            return None

    results = []
    # クリップと0埋めの数を本番と同じ形で回の終わりにログに出す（#1431）
    with collect_prediction_run_stats("challenger"):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_predict, key): key for key in all_keys}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

    log_unified_prediction_summary(
        len(results), len(all_keys), stale_keys, error_keys, label="challenger 統合モデル"
    )
    if on_error is not None:
        for market, symbol in error_keys:
            on_error(market, symbol)
    return results


def promote_unified_challenger(dry_run: bool = False) -> dict:
    """Challenger unified モデルファイルを production unified モデルファイルに昇格する。

    UnifiedStockXGBoost_challenger.joblib → UnifiedStockXGBoost.joblib
    UnifiedStockLightGBM_challenger.joblib → UnifiedStockLightGBM.joblib

    evaluate_shadow_models() で challenger_wins=True かつ
    evaluate_promotion() で eligible=True を確認してから呼び出すこと。

    上書きの前に、今の本番のファイルを `<本番の名前>_YYYYMMDD_HHMMSS.joblib`（JST）に写す。
    写しに1つでも失敗したら、どの本番のファイルも上書きせずに例外を投げる（#824）。

    Args:
        dry_run: True の場合はファイル操作を行わずログのみ出力する

    Returns:
        dict:
            promoted: list[str] — 昇格に成功した production モデル名のリスト
            skipped: list[str] — challenger ファイルが存在しなかった challenger 名
            backups: list[str] — 写した前の本番のファイルのパス
            dry_run: bool
    """
    from src.prediction.backtest_promotion.state import copy_with_backup
    from src.utils.data_path_utils import get_models_dir
    from src.utils.japan_time import now_jst

    unified_dir = os.path.join(get_models_dir(), "unified")
    stamp = now_jst().strftime("%Y%m%d_%H%M%S")
    pairs = []
    skipped = []

    for challenger_name, production_name in zip(
        _UNIFIED_CHALLENGER_NAMES, _UNIFIED_PRODUCTION_NAMES
    ):
        src = os.path.join(unified_dir, f"{challenger_name}.joblib")
        dst = os.path.join(unified_dir, f"{production_name}.joblib")
        if not os.path.exists(src):
            logger.warning(f"Challenger モデルが見つかりません: {src}")
            skipped.append(challenger_name)
            continue
        pairs.append((challenger_name, production_name, src, dst))

    if dry_run:
        for challenger_name, production_name, _, _ in pairs:
            logger.info(f"[dry_run] 昇格スキップ: {challenger_name} -> {production_name}")
        return {
            "promoted": [p[1] for p in pairs],
            "skipped": skipped,
            "backups": [],
            "dry_run": True,
        }

    # 先に全部の本番のファイルを写す。失敗したら上書きの前に止まる（例外は呼ぶ側が知らせる）
    backups = copy_with_backup([(src, dst) for _, _, src, dst in pairs], stamp)

    promoted = []
    for challenger_name, production_name, _, _ in pairs:
        logger.info(f"統合モデル昇格完了: {challenger_name} -> {production_name}")
        promoted.append(production_name)

    return {"promoted": promoted, "skipped": skipped, "backups": backups, "dry_run": False}
