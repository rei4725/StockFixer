"""monthly_report_trend（月次レポートの流れの表）のテスト（#2025）。"""

import pandas as pd

from src.reporting.monthly_report_trend import (
    TREND_COLUMNS,
    build_monthly_report_trend,
    format_monthly_trend_markdown,
    load_monthly_report_trend,
    load_monthly_reports,
    parse_monthly_report,
)

# 5月の形: 週次スナップショットなし、Slippage の説明の行なし
MAY_REPORT = "\n".join(
    [
        "# StockFixer 月次レポート 2026-05",
        "",
        "生成日時: 2026-06-01T00:00:00+00:00",
        "",
        "## KPI サマリー",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        "| Net Return | 3.10% |",
        "| Max Drawdown | -4.20% |",
        "| Sharpe Ratio | 1.05 |",
        "| Hit Rate | 55.00% |",
        "| Avg Slippage | 0.05% |",
        "| 集計銘柄数 | 120 |",
        "| WF スナップショット | N/A |",
        "",
        "## 週次 Hit Rate トレンド",
        "",
        "週次スナップショットデータなし（2 週分以上必要）",
        "",
        "## paper/real 乖離（直近30日）",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        "| 追跡件数 | 0 |",
        "| 比較可能件数 | 0 |",
        "| 平均price乖離 | N/A |",
        "| 平均乖離率 | N/A |",
        "| Paper Slippage | N/A |",
        "| Real Slippage | N/A |",
        "",
        "## ドリフト状況（直近30日）",
        "",
        "閾値超過銘柄数: 703",
        "",
    ]
)

# 10月の形: 週次の表あり、Slippage の説明の行あり、乖離は N/A
OCT_REPORT = "\n".join(
    [
        "# StockFixer 月次レポート 2026-10",
        "",
        "生成日時: 2026-11-01T00:00:00+00:00",
        "",
        "## KPI サマリー",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        "| Net Return | -22.15% |",
        "| Max Drawdown | -25.00% |",
        "| Sharpe Ratio | -2.08 |",
        "| Hit Rate | 48.00% |",
        "| Avg Slippage | 0.10% |",
        "| 集計銘柄数 | 98 |",
        "| WF スナップショット | wf_20261031.json |",
        "",
        "## 市場ごとの基準の物差し",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        "| Hit Rate | 99.00% |",
        "",
        "## 週次 Hit Rate トレンド",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        "| 当週 | 2026-W44 |",
        "| 当週 Hit Rate | 45.00% |",
        "| 過去 4 週平均 Hit Rate | 52.00% |",
        "| 低下率 | 13.46% |",
        "| ドリフト検知 | **YES ⚠️** |",
        "| アラート閾値 | 10.00% |",
        "",
        "## paper/real 乖離（直近30日）",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        "| 追跡件数 | 9 |",
        "| 比較可能件数 | 0 |",
        "| 平均price乖離 | N/A |",
        "| 平均乖離率 | N/A |",
        "| Paper Slippage | 0.12% |",
        "| Real Slippage | N/A |",
        "",
        "Slippage は損の向きを正にした値（買いは高く、売りは安く約定するほど正）。",
        "",
        "## ドリフト状況（直近30日）",
        "",
        "閾値超過銘柄数: 10",
        "",
    ]
)


def test_parse_may_form_blanks_na_and_missing_weekly():
    row = parse_monthly_report(MAY_REPORT, "2026-05")
    assert row is not None
    assert row["month"] == "2026-05"
    assert row["net_return"] == "3.10%"
    assert row["sharpe_ratio"] == "1.05"
    assert row["symbol_count"] == "120"
    assert row["wf_snapshot"] == ""
    assert row["week_hit_rate"] == ""
    assert row["past_avg_hit_rate"] == ""
    assert row["drift_detected"] == ""
    assert row["tracked_count"] == "0"
    assert row["paper_slippage"] == ""
    assert row["drift_symbol_count"] == "703"


def test_parse_oct_form_reads_weekly_and_ignores_baseline_section():
    row = parse_monthly_report(OCT_REPORT, "2026-10")
    assert row is not None
    assert row["net_return"] == "-22.15%"
    assert row["max_drawdown"] == "-25.00%"
    assert row["sharpe_ratio"] == "-2.08"
    # 基準の物差しの節の Hit Rate で KPI の Hit Rate を上書きしない
    assert row["hit_rate"] == "48.00%"
    assert row["avg_slippage"] == "0.10%"
    assert row["wf_snapshot"] == "wf_20261031.json"
    assert row["week_hit_rate"] == "45.00%"
    assert row["past_avg_hit_rate"] == "52.00%"
    assert row["drift_detected"] == "YES"
    assert row["tracked_count"] == "9"
    assert row["comparable_count"] == "0"
    assert row["paper_slippage"] == "0.12%"
    assert row["drift_symbol_count"] == "10"


def test_parse_rejects_non_report_text():
    assert parse_monthly_report("hello\n| a | b |", "2026-01") is None
    assert parse_monthly_report("", "2026-01") is None


def test_load_skips_broken_and_misnamed_files(tmp_path):
    (tmp_path / "2026-10_report.md").write_text(OCT_REPORT, encoding="utf-8")
    (tmp_path / "2026-05_report.md").write_text(MAY_REPORT, encoding="utf-8")
    (tmp_path / "2026-06_report.md").write_bytes(b"\xff\xfe\x00broken")
    (tmp_path / "2026-07_report.md").write_text("not a report", encoding="utf-8")
    (tmp_path / "latest_report.md").write_text(MAY_REPORT, encoding="utf-8")

    rows = load_monthly_reports(str(tmp_path))
    assert [r["month"] for r in rows] == ["2026-05", "2026-10"]

    df = load_monthly_report_trend(str(tmp_path))
    assert list(df.columns) == TREND_COLUMNS
    assert list(df["month"]) == ["2026-05", "2026-10"]
    oct_row = df[df["month"] == "2026-10"].iloc[0]
    assert oct_row["net_return"] == "-22.15%"
    assert oct_row["drift_symbol_count"] == "10"


def test_load_empty_dir_gives_empty_table(tmp_path):
    df = load_monthly_report_trend(str(tmp_path / "missing"))
    assert df.empty
    assert list(df.columns) == TREND_COLUMNS
    assert format_monthly_trend_markdown(df) == "データなし"


def test_build_sorts_by_month_and_formats_markdown():
    rows = [
        parse_monthly_report(OCT_REPORT, "2026-10"),
        parse_monthly_report(MAY_REPORT, "2026-05"),
    ]
    df = build_monthly_report_trend([r for r in rows if r is not None])
    assert isinstance(df, pd.DataFrame)
    assert list(df["month"]) == ["2026-05", "2026-10"]
    text = format_monthly_trend_markdown(df)
    lines = text.splitlines()
    assert lines[0].startswith("| month | net_return |")
    assert len(lines) == 4
    assert "| 2026-10 | -22.15% |" in lines[3]
