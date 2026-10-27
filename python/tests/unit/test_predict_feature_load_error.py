"""日次の予想で特徴量の読み込みの例外を行が無いと分け、on_error に渡すテスト（#1272）"""

from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import pandas as pd
import pytest

_SF = "src.utils.db.stock_features"
_SYMBOLS = [("jp", "5202"), ("jp", "7203")]


def _fake_read_sql(_sql, _con, params=None):
    """5202 は DB の例外、7203 は行が無い。"""
    if params["symbol"] == "5202":
        raise RuntimeError("could not read: No space left on device")
    return pd.DataFrame()


def _run_predict_all():
    from src.prediction.prediction_pipeline import predict_all_unified

    errors: list[tuple[str, str]] = []
    with ExitStack() as stack:
        stack.enter_context(patch(f"{_SF}._db_connection", side_effect=lambda: nullcontext()))
        stack.enter_context(patch(f"{_SF}.pd.read_sql", side_effect=_fake_read_sql))
        stack.enter_context(patch("src.prediction.predict_unified.preload_models"))
        stack.enter_context(
            patch("src.prediction.prediction_pipeline.get_active_symbols", return_value=_SYMBOLS)
        )
        rows = predict_all_unified(max_workers=2, on_error=lambda m, s: errors.append((m, s)))
    return rows, errors


def test_load_exception_is_passed_to_on_error():
    """読み込みが例外を投げた銘柄は on_error に渡り、行が無いだけの銘柄は渡らない"""
    rows, errors = _run_predict_all()

    assert rows == []
    assert errors == [("jp", "5202")]


def test_load_stock_features_without_flag_still_returns_none_on_exception():
    """ほかの呼び出し元（既定の呼び方）は今までどおり例外を None で返す"""
    from src.utils.db.stock_features import load_stock_features

    with (
        patch(f"{_SF}._db_connection", side_effect=lambda: nullcontext()),
        patch(f"{_SF}.pd.read_sql", side_effect=_fake_read_sql),
    ):
        assert load_stock_features("jp", "5202") is None
        assert load_stock_features("jp", "7203") is None


def test_load_stock_features_raise_on_error_raises_only_on_exception():
    """raise_on_error=True は例外を投げ、行が無いときは None を返す"""
    from src.utils.db.stock_features import load_stock_features

    with (
        patch(f"{_SF}._db_connection", side_effect=lambda: nullcontext()),
        patch(f"{_SF}.pd.read_sql", side_effect=_fake_read_sql),
    ):
        with pytest.raises(RuntimeError):
            load_stock_features("jp", "5202", raise_on_error=True)
        assert load_stock_features("jp", "7203", raise_on_error=True) is None
