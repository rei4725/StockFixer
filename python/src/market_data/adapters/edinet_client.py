"""
EdinetClient — EDINET API v2 から日本株の開示情報タイトルを取得するクライアント

EDINET (Electronic Disclosure for Investors' NETwork) は金融庁が運営する電子開示システム。
日付ごとに提出された書類一覧を取得し、指定銘柄の開示タイトルを抽出する。

API ドキュメント: https://disclosure2.edinet-fsa.go.jp/weee0020.aspx
エンドポイント: https://api.edinet-fsa.go.jp/api/v2/documents.json

注意:
  - 日付ごとに 1 回の API コールが必要なため、長期間の取得は max_days でキャップする。
  - EDINET_API_KEY 環境変数が設定されている場合は Subscription-Key として送信する。
  - 未設定でも基本アクセスは可能（レートリミットが厳しい場合がある）。
"""

import os
from datetime import date as _date
from datetime import timedelta
from typing import Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EDINET_API_BASE = "https://api.edinet-fsa.go.jp/api/v2"

# 取得対象の書類種別コード（センチメント分析に有用な開示種別）
_TARGET_DOC_TYPES = frozenset(
    {
        "120",  # 有価証券報告書
        "130",  # 四半期報告書
        "140",  # 半期報告書
        "160",  # 臨時報告書
        "350",  # 訂正臨時報告書
    }
)

# デフォルトで取得する最大日数（API コール数を制限）
_DEFAULT_MAX_DAYS = 30


def _to_edinet_sec_code(symbol: str) -> str:
    """4桁の JP 銘柄コードを EDINET 5桁証券コードに変換する。

    EDINET は "72030" のように末尾に "0" を付加した 5 桁コードを使用する。
    既に 5 桁の場合はそのまま返す。
    """
    symbol = symbol.strip()
    if len(symbol) == 4 and symbol.isdigit():
        return symbol + "0"
    return symbol


class EdinetClient:
    """EDINET API v2 クライアント。

    日本株の適時開示・有価証券報告書などのタイトルを取得し、
    センチメントスコアリングのテキストソースとして利用する。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 15,
    ) -> None:
        self._api_key = api_key or os.environ.get("EDINET_API_KEY", "")
        self._timeout = timeout

    def fetch_document_titles(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        max_days: int = _DEFAULT_MAX_DAYS,
    ) -> Optional[dict[str, list[str]]]:
        """指定銘柄・期間の開示書類タイトルを日付別に取得する。

        API コール数を抑えるため、期間末尾から max_days 日分のみ取得する。

        Args:
            symbol: JP 銘柄コード（例: "7203", "72030"）
            start_date: 取得開始日 (YYYY-MM-DD)
            end_date: 取得終了日 (YYYY-MM-DD)
            max_days: 最大取得日数。期間が長い場合は end_date 側から遡る。

        Returns:
            {date_str: [doc_title, ...]} 形式の辞書。
            開示なし・取得失敗の場合は None。
        """
        sec_code = _to_edinet_sec_code(symbol)

        try:
            end = _date.fromisoformat(end_date)
            start = _date.fromisoformat(start_date)
        except ValueError:
            logger.warning("EDINET: 不正な日付形式 start=%s end=%s", start_date, end_date)
            return None

        # 期間が max_days を超える場合は末尾 (最新) を優先
        effective_start = max(start, end - timedelta(days=max_days - 1))

        records: dict[str, list[str]] = {}
        current = effective_start
        while current <= end:
            titles = self._fetch_day(current.isoformat(), sec_code)
            if titles:
                records[current.isoformat()] = titles
            current += timedelta(days=1)

        return records if records else None

    def _fetch_day(self, date_str: str, sec_code: str) -> list[str]:
        """1日分の EDINET 書類一覧から指定銘柄のタイトルを抽出する。"""
        params: dict[str, str] = {"date": date_str, "type": "2"}
        if self._api_key:
            params["Subscription-Key"] = self._api_key

        try:
            resp = requests.get(
                f"{_EDINET_API_BASE}/documents.json",
                params=params,
                timeout=self._timeout,
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.debug("EDINET API リクエストエラー (%s): %s", date_str, e)
            return []
        except Exception as e:
            logger.warning("EDINET API 予期しないエラー (%s): %s", date_str, e)
            return []

        results = data.get("results", [])
        titles: list[str] = []
        for doc in results:
            if doc.get("secCode", "") != sec_code:
                continue
            if doc.get("docTypeCode", "") not in _TARGET_DOC_TYPES:
                continue
            desc = doc.get("docDescription") or doc.get("filerName") or ""
            if desc:
                titles.append(desc)

        return titles
