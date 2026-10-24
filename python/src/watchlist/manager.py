"""
ウォッチリスト自動更新マネージャー

指数構成銘柄（S&P500 / 日経225）を定期的に取得し、
watchlist.json の追加・削除・監査ログ保存を行う。

安全弁:
  - 削除候補は yfinance で取引可能か確認してから除外
  - 1回の更新で削除できる上限 = 全体の10%まで（大量誤削除防止）
  - 削除した銘柄は config/watchlist_audit_log.json に日付付きで記録
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.utils.data_path_utils import get_watchlist_audit_log_path, get_watchlist_path
from src.utils.db import save_index_membership_snapshot
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 1回の更新で削除できる上限割合（全体の10%）
MAX_REMOVAL_RATIO = 0.10

# マーケットキー → ウォッチリストキーのマッピング
MARKET_KEY_MAP = {
    "us": "us",
    "jp": "jp",
}

# Wikipedia は既定の User-Agent を 403 で拒否するため、連絡先を含む UA を明示する。
# （Wikimedia の User-Agent ポリシーに従い、プロジェクト名と URL を含める）
WIKIPEDIA_STORAGE_OPTIONS = {"User-Agent": "StockFixer/1.0 (https://github.com/rei4725/StockFixer)"}

# 東証の銘柄コード（4文字。2文字目と4文字目は英字になりうる。例: 7203, 285A, 130A）
_TSE_CODE_RE = re.compile(r"^[0-9][0-9A-Z][0-9][0-9A-Z]$")


# ── データクラス ──────────────────────────────────────────


@dataclass
class WatchlistDiff:
    """ウォッチリスト差分結果"""

    market: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    removed_unverified: list[str] = field(default_factory=list)  # 確認できず保留した銘柄
    kept: list[str] = field(default_factory=list)
    capped: bool = False  # 安全弁が発動したか
    fetch_failed: bool = False  # 指数銘柄の取得自体に失敗したか

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)


# ── 指数銘柄取得 ──────────────────────────────────────────


def fetch_sp500_symbols() -> list[str]:
    """
    S&P500構成銘柄をWikipediaから取得する。

    Returns:
        ティッカーシンボルのリスト（例: ["AAPL", "MSFT", ...]）
    """
    import pandas as pd

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(
            url, attrs={"id": "constituents"}, storage_options=WIKIPEDIA_STORAGE_OPTIONS
        )
        df = tables[0]
        symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P500構成銘柄取得完了: {len(symbols)}件")
        return symbols
    except Exception as e:
        logger.error(f"S&P500銘柄取得失敗: {e}", exc_info=True)
        return []


def fetch_nikkei225_symbols() -> list[str]:
    """
    日経225構成銘柄をWikipediaから取得する。

    英語版の Nikkei_225 から構成銘柄の表が消えた（2026-10 に確認）ため、日本語版の
    「日経平均株価」の業種ごとの表（「証券コード」列）から取る。

    Returns:
        銘柄コードのリスト（例: ["7203", "9984", "285A", ...]）
    """
    import pandas as pd

    url = "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1"
    try:
        tables = pd.read_html(url, storage_options=WIKIPEDIA_STORAGE_OPTIONS)
        codes: list[str] = []
        for df in tables:
            if "証券コード" not in df.columns:
                continue
            for code in df["証券コード"].dropna().astype(str).str.strip():
                if _TSE_CODE_RE.match(code) and code not in codes:
                    codes.append(code)
        if codes:
            logger.info(f"日経225構成銘柄取得完了: {len(codes)}件")
            return codes
        logger.warning("日経225テーブルに証券コードのカラムが見つかりませんでした")
        return []
    except Exception as e:
        logger.error(f"日経225銘柄取得失敗: {e}", exc_info=True)
        return []


def fetch_index_symbols(market: str) -> list[str]:
    """
    マーケットに対応する指数構成銘柄を取得する。

    Args:
        market: "us" または "jp"

    Returns:
        銘柄コードのリスト
    """
    market_lower = market.lower()
    if market_lower == "us":
        return fetch_sp500_symbols()
    elif market_lower == "jp":
        return fetch_nikkei225_symbols()
    else:
        logger.warning(f"未対応マーケット: {market}")
        return []


# ── 銘柄有効確認 ──────────────────────────────────────────


def verify_symbol_active(symbol: str, market: str) -> bool:
    """
    yfinanceで銘柄が現在も取引可能かを確認する。

    直近5日間のデータが1件以上取得できれば「有効」と判定する。

    Args:
        symbol: 銘柄コード
        market: "us" または "jp"

    Returns:
        取引可能ならTrue、廃止・取得不可ならFalse
    """
    import yfinance as yf

    from src.utils.data_path_utils import get_ticker

    ticker_str = get_ticker(market, symbol)
    try:
        ticker = yf.Ticker(ticker_str)
        hist = ticker.history(period="5d")
        active = not hist.empty
        if not active:
            logger.info(f"非アクティブ確認: {ticker_str} (データなし)")
        return active
    except Exception as e:
        logger.warning(f"銘柄確認失敗: {ticker_str}: {e}", exc_info=True)
        return False


# ── 差分計算 ──────────────────────────────────────────────


def diff_watchlist(
    current: list[str],
    fetched: list[str],
    market: str,
) -> WatchlistDiff:
    """
    現在のウォッチリストと取得した指数構成銘柄を比較し差分を返す。

    削除候補はyfinanceで検証し、取引可能なものは除外しない。
    また1回の更新で削除できる上限（MAX_REMOVAL_RATIO）を超える場合は安全弁を発動する。

    Args:
        current: 現在のウォッチリスト
        fetched: 指数から取得した最新銘柄リスト
        market: マーケット識別子

    Returns:
        WatchlistDiff
    """
    current_set = set(s.upper() for s in current)
    fetched_set = set(s.upper() for s in fetched)

    added = sorted(fetched_set - current_set)
    removal_candidates = sorted(current_set - fetched_set)
    kept = sorted(current_set & fetched_set)

    diff = WatchlistDiff(market=market, added=added, kept=kept)

    if not removal_candidates:
        return diff

    # 削除候補をyfinanceで検証
    logger.info(f"[{market}] 削除候補 {len(removal_candidates)}件 をyfinanceで確認中...")
    confirmed_removed = []
    unverified = []
    for symbol in removal_candidates:
        if verify_symbol_active(symbol, market):
            # まだ取引可能 → 指数から除外されただけの可能性があるので保留
            unverified.append(symbol)
            logger.info(f"[{market}] {symbol}: 取引可能のため削除保留（指数リムーブ候補）")
        else:
            confirmed_removed.append(symbol)
            logger.info(f"[{market}] {symbol}: 取引不可を確認 → 削除対象")

    # 安全弁: 削除上限チェック
    max_remove = max(1, int(len(current) * MAX_REMOVAL_RATIO))
    capped = False
    if len(confirmed_removed) > max_remove:
        logger.warning(
            f"[{market}] 削除上限({max_remove}件)を超えたため {len(confirmed_removed)}件中"
            f" 先頭{max_remove}件のみ削除します"
        )
        confirmed_removed = confirmed_removed[:max_remove]
        capped = True

    diff.removed = confirmed_removed
    diff.removed_unverified = unverified
    diff.capped = capped

    return diff


# ── watchlist.json 更新 ───────────────────────────────────


def _load_watchlist() -> dict:
    path = get_watchlist_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_watchlist(data: dict) -> None:
    path = get_watchlist_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"watchlist.json 保存完了: {path}")


def _append_audit_log(diffs: list[WatchlistDiff]) -> None:
    """監査ログ（watchlist_audit_log.json）に変更履歴を追記する"""
    log_path = get_watchlist_audit_log_path()
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []

    entry: dict = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "markets": {},
    }
    for diff in diffs:
        entry["markets"][diff.market] = {
            "added": diff.added,
            "removed": diff.removed,
            "removed_unverified_kept": diff.removed_unverified,
            "capped": diff.capped,
        }

    log.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    logger.info(f"監査ログ追記完了: {log_path}")


def apply_watchlist_update(diffs: list[WatchlistDiff]) -> None:
    """
    WatchlistDiffのリストをwatchlist.jsonに適用する。

    Args:
        diffs: 各マーケットのWatchlistDiff
    """
    data = _load_watchlist()

    for diff in diffs:
        if not diff.has_changes:
            continue

        market_key = MARKET_KEY_MAP.get(diff.market.lower(), diff.market.lower())
        current = data.get(market_key, [])
        current_set = set(s.upper() for s in current)

        # 追加
        for sym in diff.added:
            current_set.add(sym.upper())

        # 削除（確認済みのみ）
        for sym in diff.removed:
            current_set.discard(sym.upper())

        # 元の大文字小文字を保持しつつソート
        # usはそのまま、jpは数字コードなのでソート
        new_list = sorted(current_set)
        data[market_key] = new_list
        logger.info(
            f"[{diff.market}] 追加={len(diff.added)}, 削除={len(diff.removed)}, "
            f"合計={len(new_list)}銘柄"
        )

    _save_watchlist(data)
    _append_audit_log(diffs)


# ── メイン実行 ────────────────────────────────────────────


def run_watchlist_refresh(markets: Optional[list[str]] = None) -> list[WatchlistDiff]:
    """
    ウォッチリストを指数構成銘柄と同期する。

    Args:
        markets: 対象マーケットのリスト。Noneの場合は ["us", "jp"]

    Returns:
        各マーケットのWatchlistDiffリスト
    """
    if markets is None:
        markets = ["us", "jp"]

    logger.info(f"=== ウォッチリスト更新開始 (対象: {markets}) ===")

    data = _load_watchlist()
    diffs: list[WatchlistDiff] = []

    for market in markets:
        market_key = MARKET_KEY_MAP.get(market.lower(), market.lower())
        current = data.get(market_key, [])
        logger.info(f"[{market}] 現在: {len(current)}銘柄 → 指数から最新取得中...")

        fetched = fetch_index_symbols(market)
        if not fetched:
            # 取得失敗時に差分を作ると全銘柄が削除候補になるため更新はしない。
            # ただし黙って握り潰すと失敗が誰にも見えないので diff として報告する。
            logger.warning(f"[{market}] 指数銘柄の取得に失敗したためスキップ")
            diffs.append(WatchlistDiff(market=market, fetch_failed=True))
            continue
        try:
            save_index_membership_snapshot(market=market, symbols=fetched)
        except Exception as e:
            logger.error(f"[{market}] 指数構成銘柄履歴の保存失敗: {e}", exc_info=True)

        diff = diff_watchlist(current, fetched, market)
        diffs.append(diff)

        logger.info(
            f"[{market}] 差分: 追加={len(diff.added)}, 削除={len(diff.removed)}, "
            f"保留={len(diff.removed_unverified)}"
        )

    # 変更があるマーケットのみ適用
    changed = [d for d in diffs if d.has_changes]
    if changed:
        apply_watchlist_update(changed)
        logger.info(f"=== ウォッチリスト更新完了 (変更あり: {len(changed)}マーケット) ===")
    else:
        logger.info("=== ウォッチリスト更新完了 (変更なし) ===")

    return diffs


# ── 実行結果の要約 ────────────────────────────────────────

_SUMMARY_LIST_LIMIT = 20


def failed_markets(diffs: list[WatchlistDiff]) -> list[str]:
    """指数銘柄の取得に失敗した市場の一覧"""
    return [d.market for d in diffs if d.fetch_failed]


def _format_symbols(label: str, symbols: list[str]) -> str:
    line = f"- **{label}** ({len(symbols)}): {', '.join(symbols[:_SUMMARY_LIST_LIMIT])}"
    if len(symbols) > _SUMMARY_LIST_LIMIT:
        line += " …他"
    return line


def format_refresh_summary(diffs: list[WatchlistDiff]) -> str:
    """GitHub Actions の Step Summary に書く Markdown（取得失敗を「変更なし」と書かない）"""
    lines = ["## Watchlist Refresh Results", ""]
    failed = failed_markets(diffs)
    for market in failed:
        lines.append(
            f"- **{market.upper()}**: 指数銘柄の取得失敗（watchlist.json は更新していない）"
        )
    if failed:
        lines.append("")

    changed = [d for d in diffs if d.has_changes or d.removed_unverified]
    if not changed and not failed:
        lines.append("変更なし — watchlist.json はすでに最新です。")
    for d in changed:
        lines.append(f"### {d.market.upper()}")
        if d.added:
            lines.append(_format_symbols("追加", d.added))
        if d.removed:
            lines.append(_format_symbols("削除", d.removed))
        if d.removed_unverified:
            lines.append(_format_symbols("保留", d.removed_unverified))
    return "\n".join(lines) + "\n"
