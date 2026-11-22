"""domain の sentiment_sources の直接のテスト（#2098）。

統合モデルの学習でセンチメントの列を残すか外すかを決める判定を確かめる。
キーの値は作り物だけを使い、本物の環境の値は読まない。
"""

from __future__ import annotations

import pytest

from src.domain.sentiment_sources import (
    any_sentiment_source_configured,
    is_sentiment_feature,
    sentiment_source_configured,
)

_KEY_BY_MARKET = (("us", "NEWSAPI_KEY"), ("jp", "EDINET_API_KEY"))


@pytest.mark.parametrize("market,key_name", _KEY_BY_MARKET)
def test_configured_when_key_has_value(market: str, key_name: str) -> None:
    assert sentiment_source_configured(market, {key_name: "dummy"}) is True


@pytest.mark.parametrize("market,key_name", _KEY_BY_MARKET)
@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_not_configured_when_key_blank(market: str, key_name: str, value: str) -> None:
    assert sentiment_source_configured(market, {key_name: value}) is False


@pytest.mark.parametrize("market", ["us", "jp"])
def test_not_configured_when_key_missing(market: str) -> None:
    assert sentiment_source_configured(market, {}) is False


def test_other_market_key_does_not_count() -> None:
    assert sentiment_source_configured("us", {"EDINET_API_KEY": "dummy"}) is False
    assert sentiment_source_configured("jp", {"NEWSAPI_KEY": "dummy"}) is False


def test_unknown_market_is_not_configured() -> None:
    env = {"NEWSAPI_KEY": "dummy", "EDINET_API_KEY": "dummy"}
    assert sentiment_source_configured("eu", env) is False


def test_any_configured_true_when_one_market_has_key() -> None:
    env = {"EDINET_API_KEY": "dummy"}
    assert any_sentiment_source_configured(["us", "jp"], env) is True
    assert any_sentiment_source_configured(["jp"], env) is True


def test_any_configured_false_when_no_market_has_key() -> None:
    assert any_sentiment_source_configured(["us", "jp"], {}) is False


def test_any_configured_false_for_empty_markets() -> None:
    env = {"NEWSAPI_KEY": "dummy", "EDINET_API_KEY": "dummy"}
    assert any_sentiment_source_configured([], env) is False


@pytest.mark.parametrize(
    "column,expected",
    [
        ("sentiment_score_lag1", True),
        ("news_count_lag1", True),
        ("close_lag1", False),
    ],
)
def test_is_sentiment_feature(column: str, expected: bool) -> None:
    assert is_sentiment_feature(column) is expected


def test_env_given_ignores_os_environ_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "dummy")
    monkeypatch.setenv("EDINET_API_KEY", "dummy")
    assert sentiment_source_configured("us", {}) is False
    assert sentiment_source_configured("jp", {}) is False
    assert any_sentiment_source_configured(["us", "jp"], {}) is False


def test_env_given_ignores_os_environ_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("EDINET_API_KEY", raising=False)
    assert sentiment_source_configured("us", {"NEWSAPI_KEY": "dummy"}) is True
    assert sentiment_source_configured("jp", {"EDINET_API_KEY": "dummy"}) is True
