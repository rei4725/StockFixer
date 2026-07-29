import pandas as pd

from src.utils.db.stock_features import (
    delete_stock_features,
    get_all_symbols,
    load_all_stock_features,
    load_stock_features,
    upsert_stock_features,
)


def test_upsert_and_load_roundtrip():
    df = pd.DataFrame({"close": [100.0, 101.0], "rsi": [50.0, 55.0]})
    upsert_stock_features("us", "TEST", df)

    loaded = load_stock_features("us", "TEST")

    assert loaded is not None
    assert list(loaded["close"]) == [100.0, 101.0]
    assert list(loaded["rsi"]) == [50.0, 55.0]


def test_upsert_adds_new_column_dynamically():
    df1 = pd.DataFrame({"close": [100.0]})
    upsert_stock_features("us", "TEST2", df1)

    df2 = pd.DataFrame({"close": [101.0], "new_indicator": [42.0]})
    upsert_stock_features("us", "TEST2", df2)

    loaded = load_stock_features("us", "TEST2")
    assert "new_indicator" in loaded.columns


def test_get_all_symbols_includes_saved_symbol():
    upsert_stock_features("jp", "9999", pd.DataFrame({"close": [1.0]}))
    assert ("jp", "9999") in get_all_symbols()


def test_delete_removes_data():
    upsert_stock_features("us", "TEST3", pd.DataFrame({"close": [1.0]}))
    delete_stock_features("us", "TEST3")
    assert load_stock_features("us", "TEST3") is None


def test_load_all_stock_features_combines_symbols():
    upsert_stock_features("us", "A1", pd.DataFrame({"close": [1.0]}))
    upsert_stock_features("us", "A2", pd.DataFrame({"close": [2.0]}))
    all_df = load_all_stock_features()
    assert set(all_df["symbol"]) >= {"A1", "A2"}


def test_load_stock_features_coerces_all_null_column_to_numeric():
    """Issue: ある銘柄で全行NULLの列が object dtype で返り、XGBoost予測が
    KeyError: 'object' でスキップされていた（#Capital_Gains_lag系バグ）。

    upsert 済みの他銘柄に数値列が存在する状態でも、対象銘柄のスライスが
    全行NULLなら pd.read_sql は object dtype で返す。load_stock_features は
    これを float64（NaN）へ強制変換すること。
    """
    upsert_stock_features("us", "HASVALUE", pd.DataFrame({"close": [1.0], "capital_gains": [0.5]}))
    upsert_stock_features("us", "ALLNULL", pd.DataFrame({"close": [2.0], "capital_gains": [None]}))

    loaded = load_stock_features("us", "ALLNULL")

    assert loaded is not None
    assert pd.api.types.is_float_dtype(loaded["capital_gains"])
    assert loaded["capital_gains"].isna().all()


def test_load_all_stock_features_coerces_all_null_column_to_numeric():
    upsert_stock_features("us", "B1", pd.DataFrame({"close": [1.0], "capital_gains": [None]}))

    all_df = load_all_stock_features()

    assert pd.api.types.is_float_dtype(all_df["capital_gains"])
