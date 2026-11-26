"""prediction/unified_frame.py の行と列の選び出しの直のテスト（#2186）。"""

import numpy as np
import pandas as pd
import pytest

from src.prediction.unified_frame import (
    _VALID_CHUNK_COLUMNS,
    rows_without_nan,
    select_rows_and_columns,
)

_N_ROWS = 6


def _float_table(n_cols: int, index=None) -> pd.DataFrame:
    rng = np.random.default_rng(n_cols)
    data = rng.normal(size=(_N_ROWS, n_cols))
    return pd.DataFrame(data, columns=[f"c{i:03d}" for i in range(n_cols)], index=index)


def test_chunk_size_is_64():
    assert _VALID_CHUNK_COLUMNS == 64


@pytest.mark.parametrize("n_cols", [64, 65, 130])
def test_rows_without_nan_marks_rows_with_nan_in_any_chunk(n_cols):
    df = _float_table(n_cols, index=[10, 3, 7, 7, 42, 0])
    cols = list(df.columns)
    # 行1: 1つ目のまとまりの最初の列、行2: 最後の列
    df.iloc[1, 0] = np.nan
    df.iloc[2, n_cols - 1] = np.nan
    if n_cols > _VALID_CHUNK_COLUMNS:
        # 行4: 2つ目のまとまりの最初の列（65列目）
        df.iloc[4, _VALID_CHUNK_COLUMNS] = np.nan
    if n_cols > 2 * _VALID_CHUNK_COLUMNS:
        # 行5: 3つ目のまとまりの最初の列（129列目）
        df.iloc[5, 2 * _VALID_CHUNK_COLUMNS] = np.nan

    result = rows_without_nan(df, cols)

    expected = df[cols].notna().all(axis=1)
    pd.testing.assert_series_equal(result, expected)
    false_rows = [i for i, v in enumerate(result.to_numpy()) if not v]
    want = [1, 2]
    if n_cols > _VALID_CHUNK_COLUMNS:
        want.append(4)
    if n_cols > 2 * _VALID_CHUNK_COLUMNS:
        want.append(5)
    assert false_rows == want


def test_rows_without_nan_last_column_of_second_chunk_only():
    df = _float_table(130)
    df.iloc[3, 2 * _VALID_CHUNK_COLUMNS - 1] = np.nan  # 128列目（2つ目のまとまりの最後）
    result = rows_without_nan(df, list(df.columns))
    assert result.tolist() == [True, True, True, False, True, True]


def test_rows_without_nan_subset_of_columns_ignores_other_columns():
    df = _float_table(70)
    df.iloc[0, 69] = np.nan
    cols = list(df.columns[:66])
    df.iloc[2, 65] = np.nan
    result = rows_without_nan(df, cols)
    assert result.tolist() == [True, True, False, True, True, True]


def test_rows_without_nan_empty_columns_all_true():
    df = _float_table(3, index=[5, 1, 9, 2, 8, 4])
    df.iloc[0, 0] = np.nan
    result = rows_without_nan(df, [])
    pd.testing.assert_series_equal(result, pd.Series(True, index=df.index))


def _mixed_table(index) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "f32a": np.arange(n, dtype=np.float32) + 0.5,
            "f32b": np.arange(n, dtype=np.float32) * 2.25,
            "f64a": np.linspace(-1.0, 1.0, n),
            "i64": np.arange(n, dtype=np.int64) * 7,
            "f32c": np.arange(n, dtype=np.float32) - 3.125,
            "boo": np.arange(n) % 2 == 0,
            "nint": pd.array([1, pd.NA, 3, 4, pd.NA, 6][:n], dtype="Int64"),
            "f64b": np.arange(n, dtype=np.float64) / 3.0,
            "cat": pd.Categorical(list("abcabc")[:n], categories=["a", "b", "c"]),
            "nint2": pd.array([pd.NA, 2, 3, pd.NA, 5, 6][:n], dtype="Int64"),
            "str": pd.array(list("uvwxyz")[:n], dtype="string"),
            "i64b": np.arange(n, dtype=np.int64) - 100,
            "f64c": np.arange(n, dtype=np.float64) ** 2,
        },
        index=index,
    )


_COLUMN_ORDERS = [
    # 同じ型が続く・離れて戻る
    ["f32a", "f32b", "f64a", "i64", "f32c", "boo", "nint", "f64b", "cat", "nint2", "str"],
    ["nint", "nint2", "f32a", "cat", "f32b", "i64", "i64b", "str", "f64a", "f64c", "boo"],
    ["cat", "f64c", "f64b", "f64a", "str", "nint", "f32c", "f32a", "boo", "i64b"],
]

_ROWS = [
    [True, False, True, True, False, True],
    [True] * 6,
    [False] * 6,
]

_INDEXES = [
    list(range(6)),
    [3, 17, 4, 99, 0, 250],  # 飛び飛び
    [5, 5, 2, 9, 2, 5],  # 重なりあり
]


@pytest.mark.parametrize("index", _INDEXES)
@pytest.mark.parametrize("rows_list", _ROWS)
@pytest.mark.parametrize("columns", _COLUMN_ORDERS)
def test_select_rows_and_columns_matches_naive_selection(index, rows_list, columns):
    df = _mixed_table(index)
    rows = pd.Series(rows_list, index=df.index)

    result = select_rows_and_columns(df, rows, columns)

    expected = df[columns][rows]
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert list(result.columns) == columns
    assert list(result.dtypes) == list(expected.dtypes)


@pytest.mark.parametrize("index", _INDEXES)
@pytest.mark.parametrize("rows_list", _ROWS)
def test_select_rows_and_columns_empty_columns(index, rows_list):
    df = _mixed_table(index)
    rows = pd.Series(rows_list, index=df.index)

    result = select_rows_and_columns(df, rows, [])

    expected = df[[]][rows]
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert list(result.columns) == []


def test_select_rows_and_columns_wide_float_table():
    df = _float_table(130, index=[8, 1, 1, 30, 2, 2])
    df["nint"] = pd.array([1, 2, pd.NA, 4, 5, 6], dtype="Int64")
    columns = list(df.columns)
    rows = rows_without_nan(df, columns)
    assert rows.tolist() == [True, True, False, True, True, True]

    result = select_rows_and_columns(df, rows, columns)

    pd.testing.assert_frame_equal(result, df[columns][rows], check_exact=True)


@pytest.mark.parametrize("columns", _COLUMN_ORDERS)
def test_select_rows_and_columns_result_is_independent_copy(columns):
    df = _mixed_table(list(range(6)))
    before = df.copy(deep=True)
    rows = pd.Series([True, True, False, True, True, True], index=df.index)

    result = select_rows_and_columns(df, rows, columns)
    for k, name in enumerate(columns):
        # 先頭の行を2行目の値で書き換える（その場の書き込み）
        result.iloc[0, k] = result.iloc[1, k]
    result.loc[result.index[-1], columns[-1]] = result.iloc[0, len(columns) - 1]

    pd.testing.assert_frame_equal(df, before, check_exact=True)


def test_select_rows_and_columns_inplace_write_does_not_touch_source():
    df = _float_table(5)
    before = df.copy(deep=True)
    rows = pd.Series(True, index=df.index)

    result = select_rows_and_columns(df, rows, list(df.columns))
    result.iloc[:, :] = -1.0
    result.iloc[0, 0] = 123.0

    pd.testing.assert_frame_equal(df, before, check_exact=True)
