"""紙の往復のまとめの総利益・総損失・プロフィットファクター・平均の騰落率のテスト（#2179）"""

from datetime import datetime

import pandas as pd
import pytest
import run_paper_roundtrips

from src.reporting.paper_roundtrips import (
    ROUTE_PREDICTION,
    ROUTE_RULE,
    RoundTrip,
    summarize_roundtrips,
)


def _trip(qty: int, entry: float, exit_: float, route: str = ROUTE_PREDICTION) -> RoundTrip:
    return RoundTrip(
        "A", qty, route, entry, exit_, datetime(2026, 9, 1, 23, 50), datetime(2026, 9, 3, 7, 30)
    )


def test_mixed_wins_losses_and_flat_match_hand_values():
    trips = [
        _trip(10, 100.0, 110.0),  # 勝ち 損益 +100、騰落率 +10%
        _trip(5, 200.0, 260.0),  # 勝ち 損益 +300、騰落率 +30%
        _trip(20, 50.0, 45.0),  # 負け 損益 -100、騰落率 -10%
        _trip(10, 100.0, 100.0),  # 損益 0。勝ちにも負けにも数えない
        _trip(1, 0.0, 10.0),  # 勝ち 損益 +10。入口が 0 なので平均の騰落率に入れない
    ]
    summary = summarize_roundtrips(trips)
    for g in (summary.by_route[0], summary.by_month[0]):
        assert g.trips == 5 and g.wins == 3
        assert g.pnl == pytest.approx(310.0)
        assert g.gross_profit == pytest.approx(410.0)
        assert g.gross_loss == pytest.approx(100.0)
        assert g.profit_factor == pytest.approx(4.1)
        assert g.avg_win_return == pytest.approx(0.20)
        assert g.avg_loss_return == pytest.approx(-0.10)


def test_group_without_losses_has_no_profit_factor_and_no_avg_loss():
    g = summarize_roundtrips([_trip(10, 100.0, 110.0), _trip(10, 100.0, 100.0)]).by_route[0]
    assert g.gross_loss == 0.0
    assert g.profit_factor is None
    assert g.avg_loss_return is None
    assert g.avg_win_return == pytest.approx(0.10)


def test_group_without_wins_has_zero_profit_factor_and_no_avg_win():
    g = summarize_roundtrips([_trip(10, 100.0, 90.0)]).by_route[0]
    assert g.gross_profit == 0.0
    assert g.gross_loss == pytest.approx(100.0)
    assert g.profit_factor == 0.0
    assert g.avg_win_return is None
    assert g.avg_loss_return == pytest.approx(-0.10)


def test_main_prints_new_columns_with_dash_for_none(monkeypatch, capsys):
    rows = [
        # ルールの道は勝ちだけ（PF と負けの平均が -）
        ("b1", "jp", "R", 1, 10, 100.0, datetime(2026, 9, 1, 7, 30), datetime(2026, 9, 2)),
        ("s1", "jp", "R", 2, 10, 110.0, datetime(2026, 9, 3, 7, 30), datetime(2026, 9, 4)),
        # 予想の道は負けだけ（PF 0.00、勝ちの平均が -）
        ("b2", "jp", "P", 1, 10, 100.0, datetime(2026, 9, 1, 23, 50), datetime(2026, 9, 2)),
        ("s2", "jp", "P", 2, 10, 90.0, datetime(2026, 9, 3, 7, 30), datetime(2026, 9, 4)),
    ]
    orders = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "market",
            "symbol",
            "side",
            "qty",
            "fill_price",
            "created_at",
            "filled_at",
        ],
    )

    class _Con:
        def close(self):
            pass

    monkeypatch.setattr(run_paper_roundtrips, "get_readonly_connection", lambda: _Con())
    monkeypatch.setattr(run_paper_roundtrips, "load_paper_roundtrip_orders", lambda con: orders)
    monkeypatch.setattr(run_paper_roundtrips, "_default_split_loader", lambda m, s: None)
    run_paper_roundtrips.main()
    out = capsys.readouterr().out
    lines = out.splitlines()
    route_part = out.split("=== 紙の往復 道と月 ===")[0]
    month_part = out.split("=== 紙の往復 道と月 ===")[1]

    def _line(part: str, route: str) -> str:
        return next(x for x in part.splitlines() if x.strip().startswith(route))

    rule = _line(route_part, ROUTE_RULE)
    pred = _line(route_part, ROUTE_PREDICTION)
    assert rule.endswith("PF -  勝ち平均 +10.00%  負け平均 -")
    assert pred.endswith("PF 0.00  勝ち平均 -  負け平均 -10.00%")
    # 既にある欄はそのまま前に並ぶ
    assert "勝率 100.0%" in rule and "平均保有   2.0日" in rule
    assert _line(month_part, ROUTE_RULE).endswith("PF -  勝ち平均 +10.00%  負け平均 -")
    assert _line(month_part, ROUTE_PREDICTION).endswith("PF 0.00  勝ち平均 -  負け平均 -10.00%")
    assert any("往復の損益の合計: 0" in x for x in lines)
