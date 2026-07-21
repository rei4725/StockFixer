import pandas as pd

from src.utils.db._bulk import bulk_insert, bulk_upsert
from src.utils.db._connection import _db_connection


def test_bulk_insert_writes_all_rows():
    df = pd.DataFrame({"key": ["strategy-factory-idea"], "value": ["1"]})
    with _db_connection() as con:
        con.execute("CREATE TEMP TABLE _t_insert (key VARCHAR, value VARCHAR)")
        bulk_insert(con, "_t_insert", df)
        rows = con.execute("SELECT key, value FROM _t_insert").fetchall()
    assert rows == [("strategy-factory-idea", "1")]


def test_bulk_upsert_updates_existing_key():
    with _db_connection() as con:
        con.execute("CREATE TEMP TABLE _t_upsert (k VARCHAR PRIMARY KEY, v INTEGER)")
        con.execute("INSERT INTO _t_upsert VALUES ('a', 1)")
        df = pd.DataFrame({"k": ["a", "b"], "v": [99, 2]})
        bulk_upsert(con, "_t_upsert", df, key_cols=["k"])
        rows = dict(con.execute("SELECT k, v FROM _t_upsert ORDER BY k").fetchall())
    assert rows == {"a": 99, "b": 2}
