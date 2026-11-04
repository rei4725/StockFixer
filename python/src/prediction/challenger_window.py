"""影の比べと昇格ゲートの窓を、今の challenger が出した予想の日に絞る（#1327）。

challenger は毎週学び直されるが、採点の行は版にかかわらず model_name="challenger" で書かれる。
直近 N 日の窓には捨てた前の challenger の予想が混ざるため、今の challenger のファイルの
更新時刻より後の predicted_at だけを残し、production もその日だけで比べる。
challenger が一部の銘柄に欠けた日は、production もその銘柄の行を落とし、
同じ（日・market・symbol）の組だけで比べる（#1441）。

production の行も版にかかわらず model_name="production" で書かれ、仮の入れ替え・見張りの戻し・
昇格で週の途中に中身が変わる。影の比べと昇格ゲートは、今の challenger の更新時刻と production が
最後に変わった時刻の遅い方より後の行だけで比べる（#1559）。challenger だけを見る採点の JSON は
filter_current_challenger のまま、production の変わり目では絞らない。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.prediction.prediction_pipeline import _base_bar_date
from src.utils.logger import get_logger

logger = get_logger(__name__)

_JST = ZoneInfo("Asia/Tokyo")

# 版にかかわらず challenger の影の予想の採点に使うラベル
CHALLENGER_LABEL = "challenger"

# copy_with_backup が production を上書きする前に作る写しの名前の時刻（JST）の部分
_BACKUP_STAMP = r"_(\d{8}_\d{6})(?:_\d+)?\.joblib"


@dataclass
class ChallengerWindow:
    """今の challenger の窓。trained_at が None なら challenger のファイルが無い。"""

    trained_at: Optional[datetime]
    days: list[str]
    n_challenger: int
    n_production: Optional[int] = None
    # challenger の行の（日・market・symbol）の組。None なら日だけでそろえる（#1441）
    keys: Optional[frozenset[tuple[str, str, str]]] = None
    # production が最後に変わった時刻。None なら変わり目で絞らない（#1559）
    production_changed_at: Optional[datetime] = None

    def as_dict(self) -> dict:
        return {
            "challenger_trained_at": (
                self.trained_at.isoformat() if self.trained_at is not None else None
            ),
            "production_changed_at": (
                self.production_changed_at.isoformat()
                if self.production_changed_at is not None
                else None
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


def _backup_changed_at(filename: str, production_names: list[str]) -> Optional[datetime]:
    """production の写しの名前から、上書きした時刻（UTC）を読む。写しでなければ None。"""
    for name in production_names:
        match = re.fullmatch(rf"{re.escape(name)}{_BACKUP_STAMP}", filename)
        if match is None:
            continue
        try:
            stamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            logger.warning("production の写しの時刻を読めない: %s", filename)
            return None
        return stamp.replace(tzinfo=_JST).astimezone(timezone.utc)
    return None


def current_production_changed_at() -> Optional[datetime]:
    """production の統合モデルが最後に変わった時刻（UTC）を返す。分からなければ None。

    入れ替え・戻し・昇格は copy_with_backup で上書きし、ファイルの更新時刻は写し元のまま残る
    ので、上書きの前に作られる写しの名前の時刻を読む。写しを作らずに本番のファイルを書く道
    （手で流す学習）はファイルの更新時刻で分かるので、両方の遅い方を返す。
    """
    from src.prediction.shadow_evaluation import _UNIFIED_PRODUCTION_NAMES
    from src.utils.data_path_utils import get_models_dir

    unified_dir = os.path.join(get_models_dir(), "unified")
    if not os.path.isdir(unified_dir):
        return None
    times = []
    for name in _UNIFIED_PRODUCTION_NAMES:
        path = os.path.join(unified_dir, f"{name}.joblib")
        if os.path.exists(path):
            times.append(datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc))
    for filename in os.listdir(unified_dir):
        changed_at = _backup_changed_at(filename, _UNIFIED_PRODUCTION_NAMES)
        if changed_at is not None:
            times.append(changed_at)
    return max(times) if times else None


def _predicted_at_utc(predicted_at: object) -> Optional[datetime]:
    """predicted_at（UTC の 'YYYYMMDD_HHMMSS'）を読む。読めなければ None。"""
    digits = re.sub(r"[^0-9]", "", str(predicted_at))
    try:
        return datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _predicted_day(predicted_at: object, market: Optional[str] = None) -> str:
    """predicted_at の取引日（YYYYMMDD）を返す（#1493）。

    market があれば採点の基準の日（_base_bar_date）と同じく、予想の時刻に引けていた最後の
    足の日にする。market が無ければ JST の日付にする。UTC の 00:00 を過ぎたやり直しの回が、
    次の朝の回と同じ日に入らないようにするため。時刻が読めなければ数字の先頭8字を返す。
    """
    digits = re.sub(r"[^0-9]", "", str(predicted_at))
    utc = _predicted_at_utc(digits)
    if utc is None:
        return digits[:8]
    if market is None:
        return utc.astimezone(_JST).strftime("%Y%m%d")
    return _base_bar_date(utc.strftime("%Y%m%d_%H%M%S"), market).strftime("%Y%m%d")


def _day_series(df: pd.DataFrame) -> pd.Series:
    """行ごとの取引日。market の列が無ければ JST の日付。"""
    if "market" not in df.columns:
        return df["predicted_at"].map(_predicted_day)
    return pd.Series(
        [_predicted_day(pa, str(m)) for pa, m in zip(df["predicted_at"], df["market"])],
        index=df.index,
        dtype=object,
    )


def _key_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """行ごとの（取引日・market・symbol）の組。market か symbol の列が無ければ None。"""
    if df.empty or not {"predicted_at", "market", "symbol"}.issubset(df.columns):
        return None
    days = _day_series(df)
    return pd.Series(
        list(zip(days, df["market"].astype(str), df["symbol"].astype(str))), index=df.index
    )


def _row_keys(df: pd.DataFrame) -> Optional[frozenset[tuple[str, str, str]]]:
    keys = _key_series(df)
    return frozenset(keys) if keys is not None else None


def _rows_after(df: pd.DataFrame, since: datetime) -> pd.DataFrame:
    """predicted_at が since より後の行を残す。読めない時刻の行は落とす。"""
    times = df["predicted_at"].map(_predicted_at_utc)
    return df[times.map(lambda t: t is not None and t > since).astype(bool)]


def _filter_challenger(
    df_chal: pd.DataFrame, production_changed_at: Optional[datetime]
) -> tuple[pd.DataFrame, ChallengerWindow]:
    trained_at = current_challenger_trained_at()
    since = trained_at
    if trained_at is not None and production_changed_at is not None:
        since = max(trained_at, production_changed_at)
    if since is None or df_chal.empty or "predicted_at" not in df_chal.columns:
        kept = df_chal.iloc[0:0]
    else:
        kept = _rows_after(df_chal, since)
    days = sorted(set(_day_series(kept))) if not kept.empty else []
    window = ChallengerWindow(
        trained_at=trained_at,
        days=days,
        n_challenger=len(kept),
        keys=_row_keys(kept),
        production_changed_at=production_changed_at,
    )
    if trained_at is None:
        logger.warning("今の challenger のファイルが無いため、challenger の採点の行を使わない")
    logger.info(
        "今の challenger の窓: trained_at=%s production_changed_at=%s 日数=%d "
        "challenger の件数=%d（絞る前 %d 件）",
        window.as_dict()["challenger_trained_at"],
        window.as_dict()["production_changed_at"],
        len(days),
        len(kept),
        len(df_chal),
    )
    return kept, window


def filter_current_challenger(df_chal: pd.DataFrame) -> tuple[pd.DataFrame, ChallengerWindow]:
    """challenger の行を、今の challenger のファイルの更新時刻より後の predicted_at に絞る。"""
    return _filter_challenger(df_chal, None)


def filter_comparable_challenger(df_chal: pd.DataFrame) -> tuple[pd.DataFrame, ChallengerWindow]:
    """challenger の行を、今の challenger と今の production の予想がそろう時刻より後に絞る。

    時刻は challenger の更新時刻と production が最後に変わった時刻の遅い方（#1559）。
    production の行はこの窓を filter_to_days に渡して絞る。
    """
    return _filter_challenger(df_chal, current_production_changed_at())


def filter_to_days(df: pd.DataFrame, window: ChallengerWindow) -> pd.DataFrame:
    """production の行を、今の challenger の行がある日・market・symbol の組に絞り、件数を記す。

    challenger の行に market・symbol の列が無いときは、日だけで絞る。引けの後の回と翌朝の回は
    同じ日になるので、production が変わった窓では変わり目より前の行を先に落とす（#1559）。
    """
    n_before = len(df)
    changed_at = window.production_changed_at
    if changed_at is not None and not df.empty and "predicted_at" in df.columns:
        df = _rows_after(df, changed_at)
    keys = _key_series(df) if window.keys is not None else None
    if df.empty or "predicted_at" not in df.columns or not window.days:
        kept = df.iloc[0:0]
    elif keys is not None and window.keys is not None:
        kept = df[keys.isin(window.keys)]
    else:
        kept = df[_day_series(df).isin(window.days)]
    window.n_production = len(kept)
    logger.info(
        "production を今の challenger と同じ%sに絞る: 日数=%d production の件数=%d（絞る前 %d 件）",
        "日・market・symbol" if keys is not None else "日",
        len(window.days),
        len(kept),
        n_before,
    )
    return kept
