"""reporting/walk_forward_trend の流れの表のテスト（#1926）。期待値は手で計算した値。"""

import math

import pandas as pd
import pytest

from src.reporting.walk_forward_trend import (
    COLUMNS,
    build_walk_forward_trend,
    format_walk_forward_trend_csv,
    format_walk_forward_trend_markdown,
    load_walk_forward_summaries,
    load_walk_forward_trend,
)

# 今の形: error の行（jp/6758）を含む
_NEW_FORM = (
    "market,symbol,total_return,sharpe_ratio,max_drawdown,cost_impact_return,"
    "win_rate,num_trades,fold_count,error\n"
    "us,AAPL,0.10,1.0,-0.05,0.08,0.6,10,3,\n"
    "us,MSFT,0.30,2.0,-0.15,0.20,0.4,20,3,\n"
    "jp,7203,-0.20,-1.0,-0.25,-0.22,0.5,6,3,\n"
    "jp,6758,,,,,,,,wf_df is empty\n"
)
# 古い形: cost_impact_return と error の列が無い
_OLD_FORM = (
    "market,symbol,total_return,sharpe_ratio,max_drawdown,win_rate,num_trades\n"
    "us,AAPL,0.05,0.5,-0.1,0.5,4\n"
    "us,MSFT,0.15,1.5,-0.2,0.7,8\n"
    "us,NVDA,0.40,3.0,-0.3,0.6,12\n"
)


@pytest.fixture
def report_dir(tmp_path):
    (tmp_path / "wf_summary_20260401_000000.csv").write_text(_NEW_FORM, encoding="utf-8-sig")
    (tmp_path / "wf_summary_20260325_000000.csv").write_text(_OLD_FORM, encoding="utf-8")
    # 読めないファイル（空）と market の列の無いファイルは飛ばす
    (tmp_path / "wf_summary_20260408_000000.csv").write_text("", encoding="utf-8")
    (tmp_path / "wf_summary_20260415_000000.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    # 形の違う名前は読まない
    (tmp_path / "wf_comparison_20260401_000000.md").write_text("# x", encoding="utf-8")
    (tmp_path / "wf_summary_latest.csv").write_text(_OLD_FORM, encoding="utf-8")
    return tmp_path


def _row(df: pd.DataFrame, run: str, market: str) -> pd.Series:
    hit = df[(df["run"] == run) & (df["market"] == market)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_reads_only_valid_files_in_time_order(report_dir):
    runs = load_walk_forward_summaries(str(report_dir))
    assert [run for run, _ in runs] == ["20260325_000000", "20260401_000000"]


def test_rows_are_run_then_all_then_markets(report_dir):
    df = load_walk_forward_trend(str(report_dir))
    assert list(df.columns) == list(COLUMNS)
    assert list(zip(df["run"], df["market"])) == [
        ("20260325_000000", "ALL"),
        ("20260325_000000", "us"),
        ("20260401_000000", "ALL"),
        ("20260401_000000", "jp"),
        ("20260401_000000", "us"),
    ]


def test_new_form_all_markets(report_dir):
    row = _row(load_walk_forward_trend(str(report_dir)), "20260401_000000", "ALL")
    assert row["n_symbols"] == 4
    assert row["n_errors"] == 1
    # error の行の欠けを除いた3銘柄: (0.10 + 0.30 - 0.20) / 3、中央値 0.10
    assert row["total_return_mean"] == pytest.approx(0.2 / 3)
    assert row["total_return_median"] == pytest.approx(0.10)
    assert row["sharpe_ratio_mean"] == pytest.approx(2.0 / 3)
    assert row["sharpe_ratio_median"] == pytest.approx(1.0)
    assert row["max_drawdown_mean"] == pytest.approx(-0.45 / 3)
    assert row["max_drawdown_median"] == pytest.approx(-0.15)
    assert row["cost_impact_return_mean"] == pytest.approx(0.06 / 3)
    assert row["cost_impact_return_median"] == pytest.approx(0.08)
    assert row["win_rate_mean"] == pytest.approx(0.5)
    assert row["win_rate_median"] == pytest.approx(0.5)
    assert row["num_trades_mean"] == pytest.approx(12.0)
    assert row["num_trades_median"] == pytest.approx(10.0)


def test_new_form_per_market(report_dir):
    df = load_walk_forward_trend(str(report_dir))
    us = _row(df, "20260401_000000", "us")
    assert (us["n_symbols"], us["n_errors"]) == (2, 0)
    assert us["total_return_mean"] == pytest.approx(0.20)
    assert us["total_return_median"] == pytest.approx(0.20)
    assert us["num_trades_mean"] == pytest.approx(15.0)
    jp = _row(df, "20260401_000000", "jp")
    assert (jp["n_symbols"], jp["n_errors"]) == (2, 1)
    assert jp["total_return_mean"] == pytest.approx(-0.20)
    assert jp["total_return_median"] == pytest.approx(-0.20)
    assert jp["win_rate_median"] == pytest.approx(0.5)


def test_old_form_missing_columns_are_nan(report_dir):
    row = _row(load_walk_forward_trend(str(report_dir)), "20260325_000000", "ALL")
    assert (row["n_symbols"], row["n_errors"]) == (3, 0)
    assert row["total_return_mean"] == pytest.approx(0.20)
    assert row["total_return_median"] == pytest.approx(0.15)
    assert row["sharpe_ratio_mean"] == pytest.approx(5.0 / 3)
    assert row["sharpe_ratio_median"] == pytest.approx(1.5)
    assert row["win_rate_mean"] == pytest.approx(0.6)
    assert row["num_trades_mean"] == pytest.approx(8.0)
    assert row["num_trades_median"] == pytest.approx(8.0)
    assert math.isnan(row["cost_impact_return_mean"])
    assert math.isnan(row["cost_impact_return_median"])


def test_blank_error_text_is_not_counted():
    df = pd.DataFrame(
        {"market": ["us", "us"], "symbol": ["A", "B"], "error": ["  ", "boom"]},
    )
    out = build_walk_forward_trend([("20260101_000000", df)])
    assert out.iloc[0]["n_errors"] == 1


def test_missing_dir_returns_empty(tmp_path):
    df = load_walk_forward_trend(str(tmp_path / "nope"))
    assert df.empty
    assert list(df.columns) == list(COLUMNS)
    assert format_walk_forward_trend_markdown(df) == "データなし"


def test_formats_write_null_for_missing(report_dir):
    df = load_walk_forward_trend(str(report_dir))
    text = format_walk_forward_trend_csv(df)
    lines = text.splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert "null" in lines[1]
    md = format_walk_forward_trend_markdown(df)
    md_lines = md.splitlines()
    assert len(md_lines) == 2 + len(df)
    assert md_lines[2].startswith("| 20260325_000000 | ALL | 3 | 0 | 0.2 | 0.15 |")
    assert "null" in md_lines[2]
