"""
ドメイン共有型定義。

複数の Bounded Context から参照される汎用型を集約する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SymbolTask:
    """バッチ処理の単位タスク（market + symbol + horizon）。

    batch_runner.load_target_symbols() / run_parallel() で使用する。
    複数 BC から参照される共有型のため domain/ に配置する。
    """

    market: str
    symbol: str
    horizon: int = 1


@dataclass
class HorizonResult:
    """単一ホライズンの予測結果。

    PredictionResult.horizons dict の値として使用する。
    """

    horizon_days: int
    pred_price: float
    diff_ratio: float


@dataclass
class PredictionResult:
    """1銘柄の予測結果。

    predict_single_stock / predict_with_unified_model の戻り値として使い、
    prediction_pipeline → DB保存 → Discord出力まで全層を型安全に繋ぐ。
    """

    market: str
    symbol: str
    current_price: float
    avg_pred_price: float
    diff_ratio: float
    model_count: int

    # 多ホライズン（省略可）
    avg_pred_price_3d: Optional[float] = None
    avg_pred_price_5d: Optional[float] = None
    avg_pred_price_10d: Optional[float] = None
    diff_ratio_3d: Optional[float] = None
    diff_ratio_5d: Optional[float] = None
    diff_ratio_10d: Optional[float] = None
    confluence_score: Optional[int] = None
    confidence_ratio: Optional[float] = (
        None  # 1/(1+model_std); 1.0=最大信頼度（モデル間分散が小さい）
    )

    # 信頼区間（Quantile Regression）
    pred_lower_10: Optional[float] = None  # P10予測価格（下側10%分位点）
    pred_upper_90: Optional[float] = None  # P90予測価格（上側90%分位点）

    # A/B テスト（シャドーモード）
    model_version: Optional[str] = None  # "production" / "challenger" / 任意バージョン文字列

    # マルチホライズン集約（新設計: horizon_days → HorizonResult）
    horizons: dict[int, HorizonResult] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 変換メソッド（変換知識はここに1箇所）
    # ------------------------------------------------------------------

    @classmethod
    def to_dataframe(cls, results: list["PredictionResult"]) -> pd.DataFrame:
        """PredictionResult のリストを保存・集計用 DataFrame に変換する。"""
        rows = []
        for r in results:
            row: dict = {
                "market": r.market,
                "symbol": r.symbol,
                "current_price": r.current_price,
                "avg_pred_price": r.avg_pred_price,
                "diff_ratio": r.diff_ratio,
                "model_count": r.model_count,
            }
            if r.avg_pred_price_3d is not None:
                row["avg_pred_price_3d"] = r.avg_pred_price_3d
            if r.avg_pred_price_5d is not None:
                row["avg_pred_price_5d"] = r.avg_pred_price_5d
            if r.avg_pred_price_10d is not None:
                row["avg_pred_price_10d"] = r.avg_pred_price_10d
            if r.diff_ratio_3d is not None:
                row["diff_ratio_3d"] = r.diff_ratio_3d
            if r.diff_ratio_5d is not None:
                row["diff_ratio_5d"] = r.diff_ratio_5d
            if r.diff_ratio_10d is not None:
                row["diff_ratio_10d"] = r.diff_ratio_10d
            if r.confluence_score is not None:
                row["confluence_score"] = r.confluence_score
            if r.confidence_ratio is not None:
                row["confidence_ratio"] = r.confidence_ratio
            if r.pred_lower_10 is not None:
                row["pred_lower_10"] = r.pred_lower_10
            if r.pred_upper_90 is not None:
                row["pred_upper_90"] = r.pred_upper_90
            if r.model_version is not None:
                row["model_version"] = r.model_version
            # horizons dict の内容をフラットカラムに展開（h=1 は主フィールドと重複するためスキップ）
            for h, hr in r.horizons.items():
                if h > 1:
                    row[f"avg_pred_price_{h}d"] = hr.pred_price
                    row[f"diff_ratio_{h}d"] = hr.diff_ratio
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @classmethod
    def from_dataframe_row(cls, row: "pd.Series") -> "PredictionResult":
        """DataFrame の1行から PredictionResult を復元する（DB読み込み時に使用）。"""

        def _opt_float(key: str) -> Optional[float]:
            val = row.get(key)
            return float(val) if val is not None and not pd.isna(val) else None

        def _opt_int(key: str) -> Optional[int]:
            val = row.get(key)
            return int(val) if val is not None and not pd.isna(val) else None

        def _opt_str(key: str) -> Optional[str]:
            val = row.get(key)
            return str(val) if val is not None and not pd.isna(val) else None

        horizons_dict: dict[int, HorizonResult] = {}
        for h in [3, 5, 10]:
            pp = _opt_float(f"avg_pred_price_{h}d")
            dr = _opt_float(f"diff_ratio_{h}d")
            if pp is not None and dr is not None:
                horizons_dict[h] = HorizonResult(horizon_days=h, pred_price=pp, diff_ratio=dr)

        return cls(
            market=str(row["market"]),
            symbol=str(row["symbol"]),
            current_price=float(row["current_price"]),
            avg_pred_price=float(row["avg_pred_price"]),
            diff_ratio=float(row["diff_ratio"]),
            model_count=int(row["model_count"]),
            avg_pred_price_3d=_opt_float("avg_pred_price_3d"),
            avg_pred_price_5d=_opt_float("avg_pred_price_5d"),
            avg_pred_price_10d=_opt_float("avg_pred_price_10d"),
            diff_ratio_3d=_opt_float("diff_ratio_3d"),
            diff_ratio_5d=_opt_float("diff_ratio_5d"),
            diff_ratio_10d=_opt_float("diff_ratio_10d"),
            confluence_score=_opt_int("confluence_score"),
            confidence_ratio=_opt_float("confidence_ratio"),
            pred_lower_10=_opt_float("pred_lower_10"),
            pred_upper_90=_opt_float("pred_upper_90"),
            model_version=_opt_str("model_version"),
            horizons=horizons_dict,
        )


@dataclass
class ShapFeatureContribution:
    """SHAP 説明の1特徴量。"""

    feature: str
    shap_value: float


@dataclass
class SignalSnapshot:
    """Discord signal コマンド向けの予測スナップショット。"""

    prediction: PredictionResult
    shap_direction: Optional[str] = None
    top_features: list[ShapFeatureContribution] = field(default_factory=list)


@dataclass
class BatchFailure:
    """バッチ実行の単一失敗エントリ（例外 / タイムアウト起因）。"""

    market: str
    symbol: str
    error: str


@dataclass
class BatchResult:
    """run_parallel() の集約結果。

    succeeded: 正常完了（status=="success"）した結果オブジェクトのリスト
    failed:    例外・タイムアウト・status=="error" による失敗エントリのリスト
    skipped:   status=="skip" として返された結果オブジェクトのリスト
    """

    succeeded: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


@dataclass
class HoldRules:
    """保有/撤退エンジンのルール（差し替え可能なパラメータ）。"""

    trail_ma_weeks: int = 40  # 40週(=200日)線割れで撤退
    trail_stop_pct: float = 0.35  # 高値から -35% で撤退（MA割れと OR）
    # 2倍/5倍で部分利確
    scale_out_multiples: list[float] = field(default_factory=lambda: [2.0, 5.0])
    scale_out_fraction: float = 0.20  # 各利確で保有の20%を売却（[] や 0 で「放置」=利確しない）


@dataclass
class PositionEvent:
    """保有シミュレーション中に発生した約定イベント。"""

    date: str  # YYYY-MM-DD
    action: str  # "entry" | "hold" | "scale_out" | "exit"
    price: float
    held_fraction: float  # 約定後の保有比率（1.0=フル, 0.0=撤退完了）
    reason: str  # "entry" | "ma_break" | "trail_stop" | "scale_2x" | "scale_5x" | "thesis_break"
    multiple: float  # エントリー価格に対する現在倍率（price / entry_price）


@dataclass
class TrendCandidate:
    """長期トレンド・スクリーナーが返す候補銘柄。"""

    market: str
    symbol: str
    score: float  # 合成スコア（高いほど上位）
    close: float  # 直近終値
    dist_from_52w_high: float  # 52週高値からの下落率（0=高値更新, 負値=下にある）
    above_200dma: bool  # 終値 > 200日SMA
    sma200_rising: bool  # 200日SMAが上向き（直近20日で上昇）
    return_6m: float  # 6ヶ月リターン
    return_12m: float  # 12ヶ月リターン
    avg_volume: float  # 平均出来高（流動性）


@dataclass(frozen=True)
class OrderRunSummary:
    """発注実行 1 回分のサマリー（order_run_summary テーブルの 1 行に対応）。

    run_id: 実行ごとに採番される短縮 UUID
    mode: "paper" または "live"
    min_change_ratio: この実行で適用された最小変化率しきい値
    """

    run_id: str
    market: str
    mode: str
    buy_orders: int
    sell_orders: int
    short_orders: int
    skipped: int
    skipped_min_change: int
    total_turnover: float
    min_change_ratio: float
