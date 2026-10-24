"""ユニットテスト: 影の運用の比べの日報"""

import json
from datetime import date
from unittest.mock import MagicMock

from src.trading.brokers.account.gateway import AccountGatewayBroker, AccountGatewayUnavailable
from src.trading.brokers.account.shadow_report import (
    build_shadow_report,
    read_shadow_log,
    write_shadow_report,
)

DAY = date(2026, 10, 26)


def _rec(symbol, account):
    return {
        "job": "auto",
        "symbol": symbol,
        "side": "buy",
        "qty": 100,
        "paper": {"order_id": "p", "status": "pending"},
        "account": account,
    }


def _paper():
    p = MagicMock()
    p.get_positions.return_value = [
        {"symbol": "7203", "qty": 100, "avg_price": 2500.0, "current_price": 2600.0},
        {"symbol": "6758", "qty": 50, "avg_price": 3000.0, "current_price": 3000.0},
    ]
    p.get_balance.return_value = 500000.0
    return p


def _gateway():
    g = MagicMock(spec=AccountGatewayBroker)
    g.get_positions.return_value = [
        {
            "symbol": "7203",
            "market": "jp",
            "qty": 100,
            "avg_price": 2510.0,
            "current_price": 2600.0,
        },
    ]
    g.get_balance.return_value = 640000.0
    g.get_equity.return_value = 895000.0  # 口座アプリの値（朝の値で評価）
    g.get_orders_on.return_value = [{"id": 1, "status": "filled"}]
    return g


def test_report_counts_rejections_by_code():
    records = [
        _rec("7203", {"status": "pending"}),
        _rec("6758", {"status": "rejected", "reason_code": "max_positions", "message": "m"}),
        _rec("9984", {"status": "rejected", "reason_code": "max_positions", "message": "m"}),
        _rec("8306", {"status": "error", "message": "down"}),
        _rec("4063", {"status": "not_forwarded", "message": "short"}),
    ]
    r = build_shadow_report(DAY, records, _paper(), _gateway())
    assert r["summary"] == {
        "orders": 5,
        "account_accepted": 1,
        "account_rejected": {"max_positions": 2},
        "account_error": 1,
        "not_forwarded": 1,
        "account_deduplicated": 0,
    }
    assert [m["symbol"] for m in r["mismatched_orders"]] == ["6758", "9984", "8306"]


def test_positions_and_equity_diff():
    r = build_shadow_report(DAY, [], _paper(), _gateway())
    assert r["positions_diff"] == [
        {
            "symbol": "6758",
            "paper_qty": 50,
            "account_qty": 0,
            "paper_avg_price": 3000.0,
            "account_avg_price": None,
        },
        {
            "symbol": "7203",
            "paper_qty": 100,
            "account_qty": 100,
            "paper_avg_price": 2500.0,
            "account_avg_price": 2510.0,
        },
    ]
    # 紙の評価額 = 現金 500000 + 100*2600 + 50*3000 = 910000
    # 口座アプリも紙と同じ値で評価し直す = 現金 640000 + 100*2600 = 900000
    assert r["equity"] == {
        "paper": 910000.0,
        "account": 900000.0,
        "diff": -10000.0,
        "account_reported": 895000.0,
    }
    assert r["account_orders"] == [{"id": 1, "status": "filled"}]
    assert r["account_error"] is None


def test_report_survives_account_down():
    g = _gateway()
    g.get_positions.side_effect = AccountGatewayUnavailable("down")
    r = build_shadow_report(DAY, [_rec("7203", {"status": "pending"})], _paper(), g)
    assert "down" in r["account_error"]
    assert r["positions_diff"] is None and r["equity"] is None
    assert r["summary"]["orders"] == 1


def test_read_and_write(tmp_path):
    (tmp_path / "2026-10-26.jsonl").write_text(
        json.dumps({"symbol": "7203"}) + "\n" + "{broken\n" + json.dumps({"symbol": "6758"}) + "\n",
        encoding="utf-8",
    )
    assert [x["symbol"] for x in read_shadow_log(DAY, str(tmp_path))] == ["7203", "6758"]
    assert read_shadow_log(date(2026, 10, 27), str(tmp_path)) == []
    path = write_shadow_report(DAY, {"day": "2026-10-26"}, str(tmp_path))
    assert path.endswith("2026-10-26.json")
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"day": "2026-10-26"}


def test_same_day_duplicate_is_counted_as_deduplicated():
    # 同じジョブ・銘柄・売買の2度目は、口座アプリが最初の答え（同じ order_id）を返す。注文は1回だけ
    first = _rec("7203", {"order_id": "5", "status": "pending"})
    again = _rec("7203", {"order_id": "5", "status": "pending"})
    other_job = {**_rec("7203", {"order_id": "6", "status": "pending"}), "job": "rule"}
    r = build_shadow_report(DAY, [first, again, other_job], _paper(), _gateway())
    assert r["summary"]["account_accepted"] == 2
    assert r["summary"]["account_deduplicated"] == 1
    assert len(r["mismatched_orders"]) == 1
    assert r["mismatched_orders"][0]["why"] == "deduplicated"
