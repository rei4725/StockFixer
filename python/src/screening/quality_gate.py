"""質ゲート＋合成 multibagger スコア

#429 のトレンド候補に対し、#432 の財務スナップショット（DuckDB
``stock_fundamentals``）で「成長×質×小型（=10倍の余地）」のハードゲートをかけ、
合成 multibagger スコアで最終候補をランキングする。

レイヤー規約: screening BC は utils（``utils.db.stock_fundamentals``）のみ参照し、
market_data BC は import しない（財務は DB 経由で読む）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.screening.types import MultibaggerCandidate, TrendCandidate
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.db.stock_fundamentals import load_fundamentals
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 合成スコアの重み（合計1.0）。multibagger は成長が起点となるため成長を最重視し、
# トレンド（需給・モメンタム）を次点、質（収益性・低負債）を下支えとして評価する。
# 後で最適化できるよう定数として切り出す。
_W_TREND = 0.35
_W_GROWTH = 0.40
_W_QUALITY = 0.25

# 財務欠損銘柄（on_missing="penalize"）の合成スコアに掛ける減点係数。
# 質ゲートを検証できないため、トレンド由来のスコアを大きく割り引く。
_PENALTY_FACTOR = 0.5


def _passes_hard_gate(
    fund: dict,
    min_revenue_cagr: float,
    min_roe: float,
    max_debt_to_equity: float,
    max_market_cap: float,
    require_margin_not_declining: bool,
) -> bool:
    """1銘柄の財務がハードゲートをすべて満たすか判定する。

    ゲート対象フィールドが欠損（None）の場合は検証不能とみなし除外する。
    """
    revenue_cagr = fund.get("revenue_cagr_3y")
    roe = fund.get("roe")
    de = fund.get("debt_to_equity")
    market_cap = fund.get("market_cap")

    if revenue_cagr is None or revenue_cagr < min_revenue_cagr:
        return False
    if roe is None or roe < min_roe:
        return False
    if de is None or de > max_debt_to_equity:
        return False
    if market_cap is None or market_cap > max_market_cap:
        return False
    if require_margin_not_declining and _margin_declining(fund):
        return False
    return True


def _margin_declining(fund: dict) -> bool:
    """営業利益率が前期から悪化しているか判定する。

    ``stock_fundamentals`` は最新スナップショットのみ（PIT 非対応）で前期マージンを
    持たないため、前期値 ``op_margin_prev`` が record に存在する場合のみ比較する。
    前期値が無い（=判定不能）場合は悪化とみなさない。前期マージンは将来の PIT 対応で
    供給される想定。
    """
    cur = fund.get("op_margin")
    prev = fund.get("op_margin_prev")
    if cur is None or prev is None:
        return False
    return cur < prev


def _rank_norm(series: pd.Series) -> pd.Series:
    """ユニバース内の順位を 0〜1 に正規化する（昇順、欠損は最下位 0）。"""
    return series.rank(pct=True).fillna(0.0)


def apply_quality_gate(
    candidates: list[TrendCandidate],
    min_revenue_cagr: float = 0.15,  # 売上CAGR(3年) >= 15%
    min_roe: float = 0.10,  # ROE >= 10%
    max_debt_to_equity: float = 150.0,  # D/E <= 1.5相当（fund["debt_to_equity"]はパーセントポイント単位。#637）
    max_market_cap: float = 2e10,  # 時価総額上限（小型〜中型に限定, 例 $20B）
    require_margin_not_declining: bool = True,
    on_missing: str = "exclude",  # 財務欠損銘柄: "exclude" | "penalize"
) -> list[MultibaggerCandidate]:
    """トレンド候補に財務ハードゲートをかけ、合成スコアでランキングする。

    各候補の財務を ``load_fundamentals(market, symbol)`` で取得し、成長・質・小型の
    ハードゲートで判定する。財務が無い銘柄は ``on_missing`` に従う
    （"exclude"=除外 / "penalize"=減点して残す）。

    Returns:
        合成スコア降順の MultibaggerCandidate リスト。該当なしなら空リスト。
    """
    if on_missing not in ("exclude", "penalize"):
        raise ValueError(f"on_missing は 'exclude' か 'penalize'。received: {on_missing!r}")

    if not candidates:
        return []

    rows: list[dict] = []
    for c in candidates:
        fund = load_fundamentals(c.market, c.symbol)

        if fund is None:
            if on_missing == "exclude":
                continue
            rows.append({"candidate": c, "fund": None, "missing": True})
            continue

        if not _passes_hard_gate(
            fund,
            min_revenue_cagr,
            min_roe,
            max_debt_to_equity,
            max_market_cap,
            require_margin_not_declining,
        ):
            continue

        rows.append({"candidate": c, "fund": fund, "missing": False})

    if not rows:
        logger.warning("質ゲート通過銘柄なし")
        return []

    universe = pd.DataFrame(
        {
            "row": rows,
            "trend_score": [r["candidate"].score for r in rows],
            "revenue_cagr_3y": [(r["fund"] or {}).get("revenue_cagr_3y") for r in rows],
            "roe": [(r["fund"] or {}).get("roe") for r in rows],
            "net_margin": [(r["fund"] or {}).get("net_margin") for r in rows],
            "debt_to_equity": [(r["fund"] or {}).get("debt_to_equity") for r in rows],
        }
    )

    # 成長スコア: 売上CAGR の順位正規化（利益成長はスナップショットに無いため CAGR で代表）
    growth_norm = _rank_norm(universe["revenue_cagr_3y"])
    # 質スコア: ROE・純マージン・低負債（D/E反転）の順位正規化を等加重平均
    quality_norm = (
        _rank_norm(universe["roe"])
        + _rank_norm(universe["net_margin"])
        + _rank_norm(-universe["debt_to_equity"])
    ) / 3.0
    trend_norm = _rank_norm(universe["trend_score"])

    universe = universe.assign(
        growth_norm=growth_norm,
        quality_norm=quality_norm,
        trend_norm=trend_norm,
        score=(_W_TREND * trend_norm + _W_GROWTH * growth_norm + _W_QUALITY * quality_norm),
    )
    # 財務欠損（penalize）銘柄は合成スコアを減点する
    missing_mask = universe["row"].map(lambda r: r["missing"])
    universe.loc[missing_mask, "score"] *= _PENALTY_FACTOR

    universe = universe.sort_values("score", ascending=False)

    results = [_to_candidate(rec) for rec in universe.to_dict("records")]
    logger.info(f"質ゲート完了: {len(results)} 銘柄を選定")
    return results


def _to_candidate(rec: dict) -> MultibaggerCandidate:
    """スコアリング済み行を MultibaggerCandidate に変換する。"""
    row = rec["row"]
    c: TrendCandidate = row["candidate"]
    fund = row["fund"] or {}
    return MultibaggerCandidate(
        market=c.market,
        symbol=c.symbol,
        multibagger_score=float(rec["score"]),
        trend_score=float(rec["trend_norm"]),
        growth_score=float(rec["growth_norm"]),
        quality_score=float(rec["quality_norm"]),
        close=c.close,
        revenue_cagr_3y=_opt_float(fund.get("revenue_cagr_3y")),
        roe=_opt_float(fund.get("roe")),
        op_margin=_opt_float(fund.get("op_margin")),
        net_margin=_opt_float(fund.get("net_margin")),
        debt_to_equity=_opt_float(fund.get("debt_to_equity")),
        market_cap=_opt_float(fund.get("market_cap")),
        fundamentals_missing=bool(row["missing"]),
    )


def _opt_float(value) -> Optional[float]:
    """欠損（None/NaN）を None に、それ以外を float に正規化する。"""
    if value is None or pd.isna(value):
        return None
    return float(value)


def save_multibagger_candidates(candidates: list[MultibaggerCandidate], market: str) -> str:
    """最終候補リストを CSV に保存しパスを返す。"""
    out_dir = ensure_dir(f"{get_results_dir()}/screening")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/multibagger_candidates_{market}_{timestamp}.csv"

    df = pd.DataFrame([c.__dict__ for c in candidates])
    df.to_csv(path, index=False)
    logger.info(f"最終候補を保存: {path}")
    return path
