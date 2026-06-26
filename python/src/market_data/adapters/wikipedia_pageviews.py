"""
WikipediaPageviewsClient — Wikimedia REST API から銘柄関連記事の日次ページビューを取得する。

Wikipedia のページビュー数は、銘柄・製品への一般の注目度を反映する先行指標として
学術研究で有意性が確認されている（注目度の急増 → ボラティリティ・出来高の先行シグナル）。

API ドキュメント: https://wikimedia.org/api/rest_v1/#/Pageviews%20data
エンドポイント:
    https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
        {project}/all-access/user/{article}/daily/{start}/{end}

注意:
  - キー不要・無料。ただし Wikimedia ポリシーにより User-Agent ヘッダが必須（未設定は 403）。
  - ページビューデータは 2015-07-01 以降のみ提供される。それ以前の日付はデータなし扱い。
  - 銘柄→記事名の対応は既知辞書で管理し、未登録銘柄はスキップ（None を返す）する。
  - agent=user により bot/クローラのアクセスを除外し、人間の注目度に近づける。
"""

from __future__ import annotations

import threading
import time
from datetime import date as _date
from typing import Optional
from urllib.parse import quote

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

_API_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

# Wikimedia ポリシー上 User-Agent は必須（識別可能な文字列を送る）
_USER_AGENT = "StockFixer/1.0 (https://github.com/rei4725/StockFixer; market-data pipeline)"

# 429/5xx リトライ設定
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0

# バッチパイプラインは複数銘柄を並列スレッドで処理するため、Wikimedia への
# 同時バーストを避けてリクエスト開始間隔を最小 _MIN_REQUEST_INTERVAL 秒に空ける。
_MIN_REQUEST_INTERVAL = 0.2
_rate_lock = threading.Lock()
_last_request_ts = 0.0


def _throttle() -> None:
    """並列スレッド間でリクエスト開始間隔を空ける（レート制限の予防）。"""
    global _last_request_ts
    with _rate_lock:
        wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


# 銘柄コード → Wikipedia 記事名（既知辞書）。
# 注目度シグナルが効きやすいメガキャップ／高知名度銘柄を中心に収録する。
# 未登録の銘柄は resolver が None を返し、特徴量は中立値（0）で埋められる。
_US_ARTICLE_MAP: dict[str, str] = {
    "AAPL": "Apple_Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "GOOG": "Google",
    "AMZN": "Amazon_(company)",
    "NVDA": "Nvidia",
    "META": "Meta_Platforms",
    "TSLA": "Tesla,_Inc.",
    "AMD": "Advanced_Micro_Devices",
    "NFLX": "Netflix",
    "INTC": "Intel",
    "DIS": "The_Walt_Disney_Company",
    "KO": "The_Coca-Cola_Company",
    "PEP": "PepsiCo",
    "WMT": "Walmart",
    "JPM": "JPMorgan_Chase",
    "BAC": "Bank_of_America",
    "V": "Visa_Inc.",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "BA": "Boeing",
    "NKE": "Nike,_Inc.",
    "MCD": "McDonald's",
    "SBUX": "Starbucks",
    "COST": "Costco",
    "ADBE": "Adobe_Inc.",
    "CRM": "Salesforce",
    "ORCL": "Oracle_Corporation",
    "IBM": "IBM",
    "CSCO": "Cisco",
    "QCOM": "Qualcomm",
    "T": "AT&T",
    "VZ": "Verizon",
    "XOM": "ExxonMobil",
    "CVX": "Chevron_Corporation",
    "PFE": "Pfizer",
    "JNJ": "Johnson_&_Johnson",
    "UNH": "UnitedHealth_Group",
    "HD": "The_Home_Depot",
    "LOW": "Lowe's",
    "GM": "General_Motors",
    "F": "Ford_Motor_Company",
    "GE": "General_Electric",
}

# 日本株用（現状ウォッチリストは米国株のみだが拡張余地として保持）
_JP_ARTICLE_MAP: dict[str, str] = {
    "7203": "トヨタ自動車",
    "6758": "ソニーグループ",
    "9984": "ソフトバンクグループ",
    "6861": "キーエンス",
    "8306": "三菱UFJフィナンシャル・グループ",
}

# 市場ごとの Wikipedia プロジェクト（言語版）
_MARKET_PROJECT: dict[str, str] = {
    "us": "en.wikipedia",
    "jp": "ja.wikipedia",
}

_MARKET_ARTICLE_MAP: dict[str, dict[str, str]] = {
    "us": _US_ARTICLE_MAP,
    "jp": _JP_ARTICLE_MAP,
}


def resolve_article(market: str, symbol: str) -> Optional[tuple[str, str]]:
    """銘柄コードを (project, article) に解決する。

    Args:
        market: 市場識別子 ("us" / "jp")
        symbol: 銘柄コード

    Returns:
        (project, article) のタプル。未登録の場合は None。
    """
    article_map = _MARKET_ARTICLE_MAP.get(market)
    project = _MARKET_PROJECT.get(market)
    if article_map is None or project is None:
        return None
    article = article_map.get(symbol.strip())
    if not article:
        return None
    return project, article


class WikipediaPageviewsClient:
    """Wikimedia Pageviews API クライアント。

    銘柄に対応する Wikipedia 記事の日次ページビュー数を取得し、
    注目度ベースのオルタナティブ特徴量のソースとして利用する。
    """

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    def fetch_daily_views(
        self,
        market: str,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[dict[str, int]]:
        """指定銘柄・期間の日次ページビューを取得する。

        Args:
            market: 市場識別子 ("us" / "jp")
            symbol: 銘柄コード（例: "AAPL", "7203"）
            start_date: 取得開始日 (YYYY-MM-DD)
            end_date: 取得終了日 (YYYY-MM-DD)

        Returns:
            {date_str: views} 形式の辞書。
            記事未登録・データなし・取得失敗の場合は None。
        """
        resolved = resolve_article(market, symbol)
        if resolved is None:
            logger.debug("Wikipedia 記事未登録のためスキップ: %s/%s", market, symbol)
            return None
        project, article = resolved

        try:
            start = _date.fromisoformat(start_date)
            end = _date.fromisoformat(end_date)
        except ValueError:
            logger.warning("Wikipedia: 不正な日付形式 start=%s end=%s", start_date, end_date)
            return None

        encoded_article = quote(article, safe="")
        url = (
            f"{_API_BASE}/{project}/all-access/user/{encoded_article}"
            f"/daily/{start:%Y%m%d}/{end:%Y%m%d}"
        )

        items = self._get_items_with_retry(url, article)
        if items is None:
            return None

        records: dict[str, int] = {}
        for item in items:
            timestamp = item.get("timestamp", "")
            # timestamp は "YYYYMMDDHH" 形式
            if len(timestamp) >= 8:
                date_str = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                records[date_str] = int(item.get("views", 0) or 0)

        return records if records else None

    def _get_items_with_retry(self, url: str, article: str) -> Optional[list]:
        """URL を叩き items リストを返す。429/5xx/ネットワークエラーは指数バックオフでリトライ。

        404（データなし）は即 None。レート制限（429）でリトライ尽きた場合は
        「データなし」と区別できるよう warning ログを出してから None を返す。
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            _throttle()
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                    timeout=self._timeout,
                )
            except requests.exceptions.RequestException as e:
                if attempt == _MAX_RETRIES:
                    logger.warning("Wikipedia API ネットワークエラー (%s): %s", article, e)
                    return None
                time.sleep(self._backoff_seconds(attempt))
                continue

            if resp.status_code == 404:
                # 該当記事・期間にデータがない（記事名誤り / 2015-07 以前など）
                return None

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == _MAX_RETRIES:
                    logger.warning(
                        "Wikipedia API レート制限/サーバエラー (status=%s) によりスキップ: %s",
                        resp.status_code,
                        article,
                    )
                    return None
                wait = self._retry_after_seconds(resp) or self._backoff_seconds(attempt)
                logger.warning(
                    "Wikipedia API status=%s 検出、%.1f秒待機しリトライ (%d/%d): %s",
                    resp.status_code,
                    wait,
                    attempt,
                    _MAX_RETRIES,
                    article,
                )
                time.sleep(wait)
                continue

            try:
                resp.raise_for_status()
                return resp.json().get("items", [])
            except Exception as e:
                logger.warning("Wikipedia API 予期しないエラー (%s): %s", article, e)
                return None

        return None

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """指数バックオフ待機時間（上限 _MAX_BACKOFF_SECONDS）。"""
        return min(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> Optional[float]:
        """Retry-After ヘッダ（秒数指定のみ対応）を解釈する。"""
        value = resp.headers.get("Retry-After")
        if not value:
            return None
        try:
            return min(float(value), _MAX_BACKOFF_SECONDS)
        except ValueError:
            return None
