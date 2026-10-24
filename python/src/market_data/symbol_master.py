"""
日本株の銘柄マスタ（コード → 日本語社名・33業種）を JPX の東証上場銘柄一覧から作る。

出どころ: JPX「東証上場銘柄一覧」(data_j.xls、月1回更新)
    https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
取得は週の DB メンテナンスのジョブと run_symbol_master.py から流す（毎日のジョブには入れない）。
.xls の読み込みには pandas の xlrd エンジンを使う。
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import pandas as pd
import requests

from src.market_data.types import SymbolName
from src.utils.db.symbol_master import upsert_symbol_names
from src.utils.logger import get_logger

logger = get_logger(__name__)

JPX_LISTED_ISSUES_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
)
JPX_SOURCE = "jpx_listed_issues"
_TIMEOUT_SEC = 60

_COL_CODE = "コード"
_COL_NAME = "銘柄名"
_COL_SECTOR = "33業種区分"


@dataclass
class SymbolMasterBuildResult:
    """対応表の作成結果。missing は一覧に無かったウォッチリストの銘柄。"""

    records: List[SymbolName] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


def download_jpx_listed_issues(url: str = JPX_LISTED_ISSUES_URL) -> bytes:
    """JPX の東証上場銘柄一覧（.xls）をダウンロードしてバイト列で返す。"""
    resp = requests.get(url, timeout=_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.content


def _normalize_code(value: object) -> Optional[str]:
    """Excel 上で数値になったコード（7203.0）も文字列の "7203" にそろえる。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def parse_jpx_listed_issues(df: pd.DataFrame) -> List[SymbolName]:
    """JPX 一覧の DataFrame を SymbolName のリストにする。"""
    for col in (_COL_CODE, _COL_NAME):
        if col not in df.columns:
            raise ValueError(f"JPX 一覧に列がありません: {col}")
    records: List[SymbolName] = []
    for _, row in df.iterrows():
        code = _normalize_code(row[_COL_CODE])
        name = row[_COL_NAME]
        if code is None or pd.isna(name):
            continue
        sector = row.get(_COL_SECTOR)
        sector_text = None if sector is None or pd.isna(sector) else str(sector).strip()
        if sector_text in ("", "-"):
            sector_text = None
        records.append(
            SymbolName(
                market="jp",
                symbol=code,
                name_ja=str(name).strip(),
                sector=sector_text,
                source=JPX_SOURCE,
            )
        )
    return records


def fetch_jpx_symbol_names(
    downloader: Callable[[], bytes] = download_jpx_listed_issues,
) -> List[SymbolName]:
    """JPX 一覧を取得・解析して全上場銘柄の対応を返す。"""
    content = downloader()
    df = pd.read_excel(io.BytesIO(content), dtype={_COL_CODE: str})
    records = parse_jpx_listed_issues(df)
    logger.info("JPX 東証上場銘柄一覧: %d 件", len(records))
    return records


def build_jp_symbol_master(
    watchlist_symbols: Sequence[str],
    downloader: Callable[[], bytes] = download_jpx_listed_issues,
) -> SymbolMasterBuildResult:
    """ウォッチリストの jp 銘柄について、JPX 一覧から対応を引く。"""
    by_code = {r.symbol: r for r in fetch_jpx_symbol_names(downloader)}
    result = SymbolMasterBuildResult()
    for symbol in watchlist_symbols:
        record = by_code.get(str(symbol))
        if record is None:
            result.missing.append(str(symbol))
        else:
            result.records.append(record)
    if result.missing:
        logger.warning(
            "JPX 一覧に無い銘柄 %d 件: %s", len(result.missing), ", ".join(result.missing)
        )
    return result


def refresh_jp_symbol_master() -> int:
    """watchlist.json の jp 銘柄の社名を JPX 一覧から引き、symbol_master に保存する（#1080）。

    Returns:
        保存した件数
    """
    from src.utils.data_path_utils import get_watchlist_path

    with open(get_watchlist_path(), encoding="utf-8") as f:
        symbols = json.load(f).get("jp", [])
    result = build_jp_symbol_master(symbols)
    saved = upsert_symbol_names(result.records)
    logger.info("銘柄マスタ更新完了: %d/%d 銘柄 (missing=%s)", saved, len(symbols), result.missing)
    return saved
