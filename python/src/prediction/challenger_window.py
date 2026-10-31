"""影の比べと昇格ゲートの窓を、今の challenger が出した予想の日に絞る（#1327）。

challenger は毎週学び直されるが、採点の行は版にかかわらず model_name="challenger" で書かれる。
直近 N 日の窓には捨てた前の challenger の予想が混ざるため、今の challenger のファイルの
更新時刻より後の predicted_at だけを残し、production もその日だけで比べる。
challenger が一部の銘柄に欠けた日は、production もその銘柄の行を落とし、
同じ（日・market・symbol）の組だけで比べる（#1441）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 版にかかわらず challenger の影の予想の採点に使うラベル
CHALLENGER_LABEL = "challenger"


@dataclass
class ChallengerWindow:
    """今の challenger の窓。trained_at が None なら challenger のファイルが無い。"""

    trained_at: Optional[datetime]
    days: list[str]
    n_challenger: int
    n_production: Optional[int] = None
    # challenger の行の（日・market・symbol）の組。None なら日だけでそろえる（#1441）
    keys: Optional[frozenset[tuple[str, str, str]]] = None

    def as_dict(self) -> dict:
        return {
            "challenger_trained_at": (
                self.trained_at.isoformat() if self.trained_at is not None else None
            ),
            "window_days": len(self.days),
            "window_n_challenger": self.n_challenger,
            "window_n_production": self.n_production,
        }


def current_challenger_trained_at() -> Optional[datetime]:
    """今の challenger unified モデルのファイルの更新時刻（UTC）の最も新しいものを返す。

    取り方は weekly_training._challengers_trained_this_week に合わせる。ファイルが無ければ None。
    """
    from src.prediction.shadow_evaluation import _UNIFIED_CHALLENGER_NAMES
    from src.utils.data_path_utils import get_models_dir

    unified_dir = os.path.join(get_models_dir(), "unified")
    mtimes = []
    for name in _UNIFIED_CHALLENGER_NAMES:
        path = os.path.join(unified_dir, f"{name}.joblib")
        if os.path.exists(path):
            mtimes.append(datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc))
    return max(mtimes) if mtimes else None


def _predicted_at_utc(predicted_at: object) -> Optional[datetime]:
    """predicted_at（UTC の 'YYYYMMDD_HHMMSS'）を読む。読めなければ None。"""
    digits = re.sub(r"[^0-9]", "", str(predicted_at))
    try:
        return datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _predicted_day(predicted_at: object) -> str:
    """predicted_at の日付（数字の先頭8字。load_prediction_accuracy の days の窓と同じ）。"""
    return re.sub(r"[^0-9]", "", str(predicted_at))[:8]


def _key_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """行ごとの（日・market・symbol）の組。market か symbol の列が無ければ None。"""
    if df.empty or not {"predicted_at", "market", "symbol"}.issubset(df.columns):
        return None
    days = df["predicted_at"].map(_predicted_day)
    return pd.Series(
        list(zip(days, df["market"].astype(str), df["symbol"].astype(str))), index=df.index
    )


def _row_keys(df: pd.DataFrame) -> Optional[frozenset[tuple[str, str, str]]]:
    keys = _key_series(df)
    return frozenset(keys) if keys is not None else None


def filter_current_challenger(df_chal: pd.DataFrame) -> tuple[pd.DataFrame, ChallengerWindow]:
    """challenger の行を、今の challenger のファイルの更新時刻より後の predicted_at に絞る。"""
    trained_at = current_challenger_trained_at()
    if trained_at is None or df_chal.empty or "predicted_at" not in df_chal.columns:
        kept = df_chal.iloc[0:0]
    else:
        times = df_chal["predicted_at"].map(_predicted_at_utc)
        kept = df_chal[times.map(lambda t: t is not None and t > trained_at).astype(bool)]
    days = sorted(set(kept["predicted_at"].map(_predicted_day))) if not kept.empty else []
    window = ChallengerWindow(
        trained_at=trained_at, days=days, n_challenger=len(kept), keys=_row_keys(kept)
    )
    if trained_at is None:
        logger.warning("今の challenger のファイルが無いため、challenger の採点の行を使わない")
    logger.info(
        "今の challenger の窓: trained_at=%s 日数=%d challenger の件数=%d（絞る前 %d 件）",
        window.as_dict()["challenger_trained_at"],
        len(days),
        len(kept),
        len(df_chal),
    )
    return kept, window


def filter_to_days(df: pd.DataFrame, window: ChallengerWindow) -> pd.DataFrame:
    """production の行を、今の challenger の行がある日・market・symbol の組に絞り、件数を記す。

    challenger の行に market・symbol の列が無いときは、日だけで絞る。
    """
    keys = _key_series(df) if window.keys is not None else None
    if df.empty or "predicted_at" not in df.columns or not window.days:
        kept = df.iloc[0:0]
    elif keys is not None and window.keys is not None:
        kept = df[keys.isin(window.keys)]
    else:
        kept = df[df["predicted_at"].map(_predicted_day).isin(window.days)]
    window.n_production = len(kept)
    logger.info(
        "production を今の challenger と同じ%sに絞る: 日数=%d production の件数=%d（絞る前 %d 件）",
        "日・market・symbol" if keys is not None else "日",
        len(window.days),
        len(kept),
        len(df),
    )
    return kept
