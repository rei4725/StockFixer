"""
prediction BC の型定義。

AI モデル学習・予測パイプラインで使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.domain.types import HorizonResult as HorizonResult  # noqa: F401
from src.domain.types import PredictionResult as PredictionResult  # noqa: F401
from src.domain.types import ShapFeatureContribution as ShapFeatureContribution  # noqa: F401
from src.domain.types import SignalSnapshot as SignalSnapshot  # noqa: F401

# 統合モデル（production, horizon=1d）の canonical なモデル名リスト。
#
# 「要求リスト」（orchestration/jobs/daily.py の requested_models）と
# 「実際に推論へ使うリスト」（prediction_pipeline.py の predict_all_unified 内
# model_types）が別々のリテラルだと、将来モデルを1本追加したときに片方だけ
# 更新されて出力 invariant の A-1/A-2 がサイレントに沈黙・誤発火する
# （設定ドリフトを検知する機構が設定ドリフトで壊れる、I-4 対策）。
# 両方がこの定数を参照することで単一の情報源にする。
#
# shadow_evaluation.py の _UNIFIED_PRODUCTION_NAMES（Transformer を含む別集合）
# とは別物なので混同しないこと。
UNIFIED_PREDICTION_MODEL_NAMES: list[str] = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

# 手で流す銘柄別のモデルの回（run_predict.py の --individual・single・watchlist）を
# prediction_results に保存するときの model_version。production で保存すると
# Delete-Insert で統合モデルの行を上書きし、日次の採点に混ざる（#1457）。
INDIVIDUAL_MODEL_VERSION = "individual"


@dataclass
class TrainingMetrics:
    """モデル学習後の in-sample 評価指標。

    save_model_metrics() がこの型を受け取る。
    """

    rmse: float
    directional_accuracy: float
    n_samples: int


@dataclass
class FeatureLoadResult:
    """load_features_for_training の戻り値。

    status:
        "success" — X / y 付き
        "skip"    — データ不足（reason に理由）
        "error"   — 例外発生（error に文字列）
    """

    status: str
    market: str
    symbol: str
    X: Optional[pd.DataFrame] = field(default=None, repr=False)
    y: Optional[pd.Series] = field(default=None, repr=False)
    reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == "success" and self.X is not None


@dataclass
class MarketPredictionSnapshot:
    """Discord などの表示用にまとめた市場別予測スナップショット。"""

    market: str
    top_results: list[PredictionResult] = field(default_factory=list)
    worst_results: list[PredictionResult] = field(default_factory=list)


@dataclass
class EnsembleOutcome:
    """統合モデルのアンサンブルの芯（ensemble_diff_ratio）の結果。

    Attributes:
        diff_ratio_raw: 重みで混ぜた予想の変化率（ATR クリップ前）
        diff_ratio: ATR クリップ後の予想の変化率
        clip_max_ratio: クリップの上限（ATR が取れなければ None）
        model_count: 予想に使えたモデルの数
        zero_filled: 0で埋めた特徴量（名前の順）
    """

    diff_ratio_raw: float
    diff_ratio: float
    clip_max_ratio: Optional[float]
    model_count: int
    zero_filled: list[str] = field(default_factory=list)


@dataclass
class WinDecision:
    """バックテストの候補が production に勝ったかの判定。

    Attributes:
        won: すべての条件を満たしたとき True
        reasons: 落ちた条件ごとに1つ（日本語）。勝ちなら空
    """

    won: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class HoldoutEvaluation:
    """取っておいた期間で候補と production を比べた結果（evaluate_holdout）。

    Attributes:
        cutoff: 締め日（取っておいた期間の先頭の日、YYYY-MM-DD）
        dates: 取っておいた日（YYYY-MM-DD、古い順）
        n_symbols: 取っておいた期間に y の入った行を持つ (market, symbol) の数
        expected_rows: 採点の行の見込み（markets の expected_rows の合計。覆う割合の分母）
        markets: universe の市場ごとの n_universe_symbols・n_symbols・n_dates（y のある日の数）・
            expected_rows・n_compared（比べられた行の数）・candidate・production（その市場の物差し）
        candidate: 候補の baseline_metrics()（両方の予想がそろった行だけ）
        production: production の baseline_metrics()（同じ行）
        daily: 日ごとの date・candidate・production・diff（hit_rate_minus_majority）
        path_check: 道の一致の検査の compared・max_abs_diff・passed・compared_by_market・
            skipped_by_reason（飛ばした銘柄の理由ごとの数）
        recorded_reference: prediction_accuracy の production の記録との突き合わせ（参考のみ）
        decision: 勝ちの判定
        degraded_rows: モデルが欠けて予想した行の数（candidate・production）
        skipped_rows: 飛ばした行の数（理由ごと）
        weights: 使った重みのまとめ（candidate・production ごとに calls・distinct・名前ごとの mean）
    """

    cutoff: str
    dates: list[str]
    n_symbols: int
    candidate: dict
    production: dict
    daily: list[dict]
    path_check: dict
    recorded_reference: dict
    decision: WinDecision
    expected_rows: int = 0
    markets: dict = field(default_factory=dict)
    degraded_rows: dict = field(default_factory=dict)
    skipped_rows: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)


@dataclass
class RollbackDecision:
    """仮の入れ替えのあとの見張りの判定。

    Attributes:
        rollback: 戻すとき True（triggered があり、かつ old_is_better）
        triggered: 当たった見張りの条件（日本語）
        old_is_better: 旧 production のバックテストが新 production の実測を両方で上回るか
    """

    rollback: bool
    triggered: list[str] = field(default_factory=list)
    old_is_better: bool = False


@dataclass
class BacktestPromotionState:
    """仮の入れ替えの状態（results/backtest_promotion/state.json）。

    Attributes:
        last_evaluated_version: 最後にバックテストで比べた予想のコードの版
        rolled_back_versions: 戻した候補のコードの版（この版からは再び入れ替えない）
        watch: 入れ替えたあとの見張り。無ければ None。
            version・swapped_at（JST の ISO）・swap_date（YYYY-MM-DD）・
            backups（戻せる写しのパス）・backtest_candidate・backtest_old_production を持つ
    """

    last_evaluated_version: Optional[str] = None
    rolled_back_versions: list[str] = field(default_factory=list)
    watch: Optional[dict] = None


@dataclass
class DriftMonitorResult:
    """週次 Hit Rate ドリフト検知結果。

    Attributes:
        checked_at: 検査実行日時（ISO 8601）
        current_week: 当週の week_start 文字列（例: "2026-05-11"）
        current_hit_rate: 当週の Hit Rate（方向正解率）
        avg_hit_rate: 過去 alert_weeks 週の平均 Hit Rate
        drop_ratio: (avg - current) / avg — 正の値は低下を示す
        is_drifted: drop_ratio >= alert_threshold のとき True
        alert_weeks: 比較対象の週数
        alert_threshold: ドリフト判定閾値（例: 0.05 = 5%）
    """

    checked_at: str
    current_week: Optional[str]
    current_hit_rate: Optional[float]
    avg_hit_rate: Optional[float]
    drop_ratio: Optional[float]
    is_drifted: bool
    alert_weeks: int
    alert_threshold: float
