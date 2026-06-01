import unittest
from unittest.mock import MagicMock, patch

import requests

from src.market_data.adapters.edinet_client import EdinetClient, _to_edinet_sec_code


class TestToEdinetSecCode(unittest.TestCase):
    def test_4digit_appends_zero(self):
        self.assertEqual(_to_edinet_sec_code("7203"), "72030")

    def test_5digit_unchanged(self):
        self.assertEqual(_to_edinet_sec_code("72030"), "72030")

    def test_strips_whitespace(self):
        self.assertEqual(_to_edinet_sec_code(" 7203 "), "72030")

    def test_non_numeric_5char_unchanged(self):
        self.assertEqual(_to_edinet_sec_code("AAPL0"), "AAPL0")


class TestEdinetClientFetchDay(unittest.TestCase):
    def setUp(self):
        self.client = EdinetClient()

    def _mock_response(self, results: list) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"results": results}
        return mock_resp

    @patch("requests.get")
    def test_returns_titles_for_matching_sec_code(self, mock_get):
        mock_get.return_value = self._mock_response(
            [
                {
                    "secCode": "72030",
                    "docTypeCode": "120",
                    "docDescription": "有価証券報告書",
                    "filerName": "トヨタ自動車株式会社",
                },
                {
                    "secCode": "67580",
                    "docTypeCode": "120",
                    "docDescription": "有価証券報告書（ソニー）",
                    "filerName": "ソニーグループ株式会社",
                },
            ]
        )
        titles = self.client._fetch_day("2024-01-02", "72030")
        self.assertEqual(titles, ["有価証券報告書"])

    @patch("requests.get")
    def test_returns_empty_for_404(self, mock_get):
        mock_get.return_value.status_code = 404
        titles = self.client._fetch_day("2024-01-02", "72030")
        self.assertEqual(titles, [])

    @patch("requests.get")
    def test_returns_empty_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("failed")
        titles = self.client._fetch_day("2024-01-02", "72030")
        self.assertEqual(titles, [])

    @patch("requests.get")
    def test_returns_empty_on_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        titles = self.client._fetch_day("2024-01-02", "72030")
        self.assertEqual(titles, [])

    @patch("requests.get")
    def test_filters_non_target_doc_types(self, mock_get):
        mock_get.return_value = self._mock_response(
            [
                {
                    "secCode": "72030",
                    "docTypeCode": "999",
                    "docDescription": "非対象書類",
                    "filerName": "トヨタ",
                },
            ]
        )
        titles = self.client._fetch_day("2024-01-02", "72030")
        self.assertEqual(titles, [])

    @patch("requests.get")
    def test_target_doc_types_all_accepted(self, mock_get):
        """取得対象の全書類種別コードが正しく受け入れられる"""
        for doc_type in ("120", "130", "140", "160", "350"):
            mock_get.return_value = self._mock_response(
                [
                    {
                        "secCode": "72030",
                        "docTypeCode": doc_type,
                        "docDescription": f"書類種別{doc_type}",
                        "filerName": "トヨタ",
                    }
                ]
            )
            titles = self.client._fetch_day("2024-01-02", "72030")
            self.assertEqual(len(titles), 1, f"doc_type={doc_type} が取得されていない")

    @patch("requests.get")
    def test_falls_back_to_filer_name_when_no_description(self, mock_get):
        """docDescription が空のとき filerName を使用する"""
        mock_get.return_value = self._mock_response(
            [
                {
                    "secCode": "72030",
                    "docTypeCode": "120",
                    "docDescription": "",
                    "filerName": "トヨタ自動車株式会社",
                },
            ]
        )
        titles = self.client._fetch_day("2024-01-02", "72030")
        self.assertEqual(titles, ["トヨタ自動車株式会社"])

    @patch("requests.get")
    def test_api_key_sent_as_subscription_key(self, mock_get):
        """EDINET_API_KEY が設定されているとき Subscription-Key がパラメータに含まれる"""
        client = EdinetClient(api_key="test-key-abc")
        mock_get.return_value = self._mock_response([])
        client._fetch_day("2024-01-02", "72030")

        call_kwargs = mock_get.call_args
        params = call_kwargs[1].get("params") or call_kwargs[0][1] if call_kwargs[0] else {}
        if not isinstance(params, dict):
            params = call_kwargs.kwargs.get("params", {})
        self.assertEqual(params.get("Subscription-Key"), "test-key-abc")


class TestEdinetClientFetchDocumentTitles(unittest.TestCase):
    def setUp(self):
        self.client = EdinetClient()

    def test_returns_none_for_invalid_start_date(self):
        result = self.client.fetch_document_titles("7203", "invalid-date", "2024-01-31")
        self.assertIsNone(result)

    def test_returns_none_for_invalid_end_date(self):
        result = self.client.fetch_document_titles("7203", "2024-01-01", "not-a-date")
        self.assertIsNone(result)

    def test_returns_none_when_no_disclosures(self):
        self.client._fetch_day = MagicMock(return_value=[])
        result = self.client.fetch_document_titles("7203", "2024-01-01", "2024-01-03")
        self.assertIsNone(result)

    def test_aggregates_titles_by_date(self):
        def _fake_fetch_day(date_str: str, sec_code: str) -> list:
            mapping = {
                "2024-01-02": ["有価証券報告書"],
                "2024-01-04": ["臨時報告書"],
            }
            return mapping.get(date_str, [])

        self.client._fetch_day = _fake_fetch_day
        result = self.client.fetch_document_titles("7203", "2024-01-01", "2024-01-05")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("2024-01-02", result)
        self.assertIn("2024-01-04", result)
        self.assertNotIn("2024-01-01", result)
        self.assertNotIn("2024-01-03", result)
        self.assertEqual(result["2024-01-02"], ["有価証券報告書"])

    def test_max_days_caps_api_calls(self):
        """max_days が指定された場合、期間末尾から max_days 日分のみ呼び出される"""
        called_dates: list[str] = []

        def _fake_fetch_day(date_str: str, sec_code: str) -> list:
            called_dates.append(date_str)
            return []

        self.client._fetch_day = _fake_fetch_day
        self.client.fetch_document_titles("7203", "2023-01-01", "2023-04-10", max_days=5)

        self.assertEqual(len(called_dates), 5)
        self.assertIn("2023-04-10", called_dates)
        self.assertIn("2023-04-06", called_dates)
        self.assertNotIn("2023-01-01", called_dates)

    def test_symbol_converted_to_edinet_sec_code(self):
        """4桁シンボルが EDINET 5桁証券コードに変換されてから API が呼ばれる"""
        received_sec_codes: list[str] = []

        def _fake_fetch_day(date_str: str, sec_code: str) -> list:
            received_sec_codes.append(sec_code)
            return []

        self.client._fetch_day = _fake_fetch_day
        self.client.fetch_document_titles("7203", "2024-01-01", "2024-01-02")

        for code in received_sec_codes:
            self.assertEqual(code, "72030")

    def test_short_range_within_max_days(self):
        """max_days 以内の期間ではすべての日付が取得される"""
        called_dates: list[str] = []

        def _fake_fetch_day(date_str: str, sec_code: str) -> list:
            called_dates.append(date_str)
            return []

        self.client._fetch_day = _fake_fetch_day
        self.client.fetch_document_titles("7203", "2024-01-01", "2024-01-03", max_days=30)

        self.assertEqual(len(called_dates), 3)
        self.assertIn("2024-01-01", called_dates)
        self.assertIn("2024-01-02", called_dates)
        self.assertIn("2024-01-03", called_dates)


if __name__ == "__main__":
    unittest.main()
