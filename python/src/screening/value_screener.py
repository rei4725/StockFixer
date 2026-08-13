"""低PER・低配当性向・財務安定 バリュー・スクリーナー

`stock_fundamentals` の最新スナップショットのみを使い、割安・低配当性向・
財務健全な銘柄を抽出する前向きライブスクリーン。過去のPER・配当性向は
DBに残らない（PIT非対応）ため、本スクリーンにバックテストは存在しない。

レイヤー規約: screening BC は utils（``utils.db.stock_fundamentals``）のみ
参照し、market_data BC は import しない（財務は DB 経由で読む）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.screening.types import ValueCandidate
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.db.stock_fundamentals import load_all_fundamentals
from src.utils.logger import get_logger

logger = get_logger(__name__)


def screen_value_candidates(
    market: str = "jp",
    min_per: float = 1.0,  # 実績PER下限（極端に低いPERは一時的な特別利益・データ異常の可能性が高いため除外）
    max_per: float = 10.0,
    min_payout_ratio: float = 0.05,  # 配当性向下限（0だと無配当銘柄が入り、将来の増配トリガーが存在しないため除外）
    max_payout_ratio: float = 0.30,
    max_debt_to_equity: float = 100.0,
    top_n: int = 30,
) -> list[ValueCandidate]:
    """低PER・低配当性向・財務安定な銘柄を抽出し、PER昇順でランキングする。

    ハードゲート（すべて満たす銘柄のみ残す）:
        - trailing_pe が存在し min_per 以上 max_per 以下
        - payout_ratio が存在し min_payout_ratio 以上 max_payout_ratio 以下
        - debt_to_equity が存在し max_debt_to_equity 以下（パーセントポイント単位）
        - net_income が存在し 0 より大きい（黒字）

    ゲート対象フィールドが欠損（NaN/None）の銘柄は判定不能として除外する。

    Returns:
        trailing_pe 昇順の ValueCandidate リスト（最大 top_n 件）。
        該当なしなら空リスト。

    Raises:
        ValueError: min_per > max_per または min_payout_ratio > max_payout_ratio の場合。
    """
    if min_per > max_per:
        raise ValueError(f"min_per({min_per}) は max_per({max_per}) 以下にすること")
    if min_payout_ratio > max_payout_ratio:
        raise ValueError(
            f"min_payout_ratio({min_payout_ratio}) は "
            f"max_payout_ratio({max_payout_ratio}) 以下にすること"
        )

    df = load_all_fundamentals()
    if df.empty:
        logger.warning("stock_fundamentals が空です")
        return []

    universe = df[df["market"] == market]
    if universe.empty:
        logger.warning(f"財務データがありません market={market}")
        return []

    required = ["trailing_pe", "payout_ratio", "debt_to_equity", "net_income"]
    mask = pd.Series(True, index=universe.index)
    for col in required:
        mask &= universe[col].notna()
    universe = universe[mask]

    if universe.empty:
        logger.warning(f"財務データ未取得（欠損）で判定不能: market={market}")
        return []

    universe = universe[
        (universe["trailing_pe"] >= min_per)
        & (universe["trailing_pe"] <= max_per)
        & (universe["payout_ratio"] >= min_payout_ratio)
        & (universe["payout_ratio"] <= max_payout_ratio)
        & (universe["debt_to_equity"] <= max_debt_to_equity)
        & (universe["net_income"] > 0)
    ]

    if universe.empty:
        logger.warning(f"バリュー・スクリーン通過銘柄なし（条件不一致）: market={market}")
        return []

    universe = universe.sort_values(["trailing_pe", "symbol"], ascending=True, kind="stable").head(
        top_n
    )

    candidates = [
        ValueCandidate(
            market=row["market"],
            symbol=row["symbol"],
            trailing_pe=float(row["trailing_pe"]),
            payout_ratio=float(row["payout_ratio"]),
            debt_to_equity=float(row["debt_to_equity"]),
            net_income=float(row["net_income"]),
            market_cap=(float(row["market_cap"]) if pd.notna(row["market_cap"]) else None),
        )
        for _, row in universe.iterrows()
    ]

    logger.info(f"バリュー・スクリーン完了: {len(candidates)} 銘柄を選定 ({market})")
    return candidates


def save_value_candidates(candidates: list[ValueCandidate], market: str) -> str:
    """候補リストを CSV に保存しパスを返す。"""
    out_dir = ensure_dir(f"{get_results_dir()}/screening")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/value_candidates_{market}_{timestamp}.csv"

    df = pd.DataFrame([c.__dict__ for c in candidates])
    df.to_csv(path, index=False)
    logger.info(f"候補を保存: {path}")
    return path
