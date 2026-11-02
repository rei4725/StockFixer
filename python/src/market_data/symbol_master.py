"""
銘柄マスタ（コード → 社名）を作る。

日本株: JPX「東証上場銘柄一覧」(data_j.xlsx、月1回更新) から日本語社名・33業種
    https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
米国株: Nasdaq Trader の Symbol Directory (nasdaqlisted.txt / otherlisted.txt、毎日更新) から英語名
    https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
取得は週の DB メンテナンスのジョブと run_symbol_master.py から流す（毎日のジョブには入れない）。
.xlsx は openpyxl、.xls は xlrd で読む（pandas が中身から見分ける）。
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence
from urllib.parse import urljoin

import pandas as pd
import requests

from src.market_data.types import SymbolName
from src.utils.db.symbol_master import upsert_symbol_names
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 一覧のファイルの URL は JPX 側で変わる（2026-10 に data_j.xls → data_j.xlsx になり 404 で落ちた）。
# 掲載ページからリンクを拾い、拾えなかったときだけ既定の URL を使う。
JPX_LISTED_ISSUES_PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
JPX_LISTED_ISSUES_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
)
_JPX_FILE_LINK = re.compile(r'href="([^"]*data_j\.xlsx?)"', re.IGNORECASE)
JPX_SOURCE = "jpx_listed_issues"
_TIMEOUT_SEC = 60

_COL_CODE = "コード"
_COL_NAME = "銘柄名"
_COL_SECTOR = "33業種区分"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NASDAQ_SOURCE = "nasdaqtrader_symdir"

# 銘柄名の末尾の株式種別（"Apple Inc. - Common Stock" / "Alcoa Corporation Common Stock"）を落とす
_US_NAME_SUFFIX = re.compile(
    r"(\s+-\s+.*|\s*(New\s+)?(Class\s+[A-Z]\s+)?"
    r"(Common\s+Stock|Common\s+Shares?|Ordinary\s+Shares?|Capital\s+Stock)\b.*)$",
    re.IGNORECASE,
)


@dataclass
class SymbolMasterBuildResult:
    """対応表の作成結果。missing は一覧に無かったウォッチリストの銘柄。"""

    records: List[SymbolName] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


def find_jpx_listed_issues_url(page_url: str = JPX_LISTED_ISSUES_PAGE_URL) -> str:
    """掲載ページから一覧のファイル（data_j.xls / data_j.xlsx）の URL を拾う。拾えなければ既定の URL。"""
    try:
        resp = requests.get(page_url, timeout=_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("JPX 掲載ページを読めない（既定の URL を使う）: %s", e)
        return JPX_LISTED_ISSUES_URL
    match = _JPX_FILE_LINK.search(resp.text)
    if match is None:
        logger.warning("JPX 掲載ページに一覧のリンクが無い（既定の URL を使う）: %s", page_url)
        return JPX_LISTED_ISSUES_URL
    return urljoin(page_url, match.group(1))


def download_jpx_listed_issues(url: Optional[str] = None) -> bytes:
    """JPX の東証上場銘柄一覧（.xls / .xlsx）をダウンロードしてバイト列で返す。"""
    if url is None:
        url = find_jpx_listed_issues_url()
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
                name_ja=unicodedata.normalize("NFKC", str(name)).strip(),
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


def _clean_us_name(name: str) -> str:
    """Symbol Directory の銘柄名から株式種別の飾りを落とす。落としきって空なら元のまま。"""
    text = name.strip()
    cleaned = _US_NAME_SUFFIX.sub("", text).strip()
    return cleaned or text


def parse_nasdaq_symbol_directory(text: str) -> List[SymbolName]:
    """Symbol Directory の1ファイル（| 区切り）を SymbolName のリストにする。

    nasdaqlisted.txt は "Symbol" 列、otherlisted.txt は "ACT Symbol" 列にコードがある。
    コードの "." は yfinance／watchlist の書き方の "-" にそろえる（BRK.B → BRK-B）。
    テスト銘柄（Test Issue = Y）と末尾の "File Creation Time" 行は捨てる。
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    symbol_col = "Symbol" if "Symbol" in header else "ACT Symbol"
    for col in (symbol_col, "Security Name"):
        if col not in header:
            raise ValueError(f"Symbol Directory に列がありません: {col}")
    i_sym = header.index(symbol_col)
    i_name = header.index("Security Name")
    i_test = header.index("Test Issue") if "Test Issue" in header else None
    records: List[SymbolName] = []
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        cols = line.split("|")
        if len(cols) <= max(i_sym, i_name):
            continue
        if i_test is not None and len(cols) > i_test and cols[i_test].strip() == "Y":
            continue
        symbol = cols[i_sym].strip().replace(".", "-")
        name = cols[i_name].strip()
        if not symbol or not name:
            continue
        records.append(
            SymbolName(
                market="us",
                symbol=symbol,
                name_ja=None,
                name_en=_clean_us_name(name),
                source=NASDAQ_SOURCE,
            )
        )
    return records


def download_nasdaq_symbol_directory() -> List[str]:
    """nasdaqlisted.txt と otherlisted.txt を取得してテキストで返す。"""
    texts = []
    for url in (NASDAQ_LISTED_URL, OTHER_LISTED_URL):
        resp = requests.get(url, timeout=_TIMEOUT_SEC)
        resp.raise_for_status()
        texts.append(resp.text)
    return texts


def build_us_symbol_master(
    watchlist_symbols: Sequence[str],
    downloader: Callable[[], List[str]] = download_nasdaq_symbol_directory,
) -> SymbolMasterBuildResult:
    """ウォッチリストの us 銘柄について、Symbol Directory から英語名を引く。"""
    by_code: dict[str, SymbolName] = {}
    for text in downloader():
        for parsed in parse_nasdaq_symbol_directory(text):
            by_code.setdefault(parsed.symbol, parsed)
    logger.info("Nasdaq Symbol Directory: %d 件", len(by_code))
    result = SymbolMasterBuildResult()
    for symbol in watchlist_symbols:
        record = by_code.get(str(symbol).upper())
        if record is None:
            result.missing.append(str(symbol))
        else:
            result.records.append(record)
    if result.missing:
        logger.warning(
            "Symbol Directory に無い銘柄 %d 件: %s", len(result.missing), ", ".join(result.missing)
        )
    return result


def _load_watchlist_symbols(market: str) -> List[str]:
    from src.utils.data_path_utils import get_watchlist_path

    with open(get_watchlist_path(), encoding="utf-8") as f:
        return list(json.load(f).get(market, []))


def refresh_us_symbol_master() -> int:
    """watchlist.json の us 銘柄の英語名を Symbol Directory から引き、symbol_master に保存する。

    Returns:
        保存した件数
    """
    symbols = _load_watchlist_symbols("us")
    result = build_us_symbol_master(symbols)
    saved = upsert_symbol_names(result.records)
    logger.info(
        "銘柄マスタ(us)更新完了: %d/%d 銘柄 (missing=%s)", saved, len(symbols), result.missing
    )
    return saved


def refresh_jp_symbol_master() -> int:
    """watchlist.json の jp 銘柄の社名を JPX 一覧から引き、symbol_master に保存する（#1080）。

    Returns:
        保存した件数
    """
    symbols = _load_watchlist_symbols("jp")
    result = build_jp_symbol_master(symbols)
    saved = upsert_symbol_names(result.records)
    logger.info("銘柄マスタ更新完了: %d/%d 銘柄 (missing=%s)", saved, len(symbols), result.missing)
    return saved
