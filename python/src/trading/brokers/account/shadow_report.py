"""
影の運用の比べの日報 — results/account_shadow/YYYY-MM-DD.json

紙と口座アプリの、その日の注文の通り・断り（理由の符号ごと）、持ち高（株数と平均の値）、評価額を並べる。
段3の合否は人間が、食い違いが上限の違いか約定の値の違いで説明できるかで決める。
"""

import json
import os
from collections import Counter
from datetime import date
from typing import Any

from src.trading.brokers.account.gateway import AccountGatewayBroker
from src.trading.brokers.account.shadow import shadow_log_dir
from src.trading.brokers.base import BrokerBase
from src.utils.japan_time import isoformat_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_shadow_log(day: date, log_dir: str | None = None) -> list[dict[str, Any]]:
    path = os.path.join(log_dir or shadow_log_dir(), f"{day.isoformat()}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                logger.warning("[shadow] %s の %s 行目を読めない。飛ばす", path, n)
    return records


def _summarize(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rejected: Counter[str] = Counter()
    accepted = errors = not_forwarded = 0
    mismatched = []
    for r in records:
        account = r.get("account") or {}
        status = account.get("status")
        if status == "pending":
            accepted += 1
        elif status == "not_forwarded":
            not_forwarded += 1
        else:
            if status == "rejected":
                rejected[str(account.get("reason_code"))] += 1
            else:
                errors += 1
            mismatched.append(r)
    summary = {
        "orders": len(records),
        "account_accepted": accepted,
        "account_rejected": dict(rejected),
        "account_error": errors,
        "not_forwarded": not_forwarded,
    }
    return summary, mismatched


def _positions_diff(
    paper: list[dict[str, Any]], account: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    p = {str(x["symbol"]).replace(".T", ""): x for x in paper}
    a = {str(x["symbol"]): x for x in account}
    rows = []
    for sym in sorted(set(p) | set(a)):
        pq = int(p[sym]["qty"]) if sym in p else 0
        aq = int(a[sym]["qty"]) if sym in a else 0
        pa = float(p[sym]["avg_price"]) if sym in p else None
        aa = float(a[sym]["avg_price"]) if sym in a else None
        if pq != aq or (pa is not None and aa is not None and abs(pa - aa) > 1e-6):
            rows.append(
                {
                    "symbol": sym,
                    "paper_qty": pq,
                    "account_qty": aq,
                    "paper_avg_price": pa,
                    "account_avg_price": aa,
                }
            )
    return rows


def build_shadow_report(
    day: date,
    records: list[dict[str, Any]],
    paper: BrokerBase,
    gateway: AccountGatewayBroker,
) -> dict[str, Any]:
    summary, mismatched = _summarize(records)
    report: dict[str, Any] = {
        "day": day.isoformat(),
        "generated_at": isoformat_jst(),
        "summary": summary,
        "mismatched_orders": mismatched,
        "positions_diff": None,
        "equity": None,
        "account_orders": None,
        "account_error": None,
        "orders": records,
    }
    paper_positions = paper.get_positions()
    paper_equity = float(paper.get_balance()) + sum(
        float(x["qty"]) * float(x["current_price"]) for x in paper_positions
    )
    try:
        account_positions = gateway.get_positions()
        account_equity = gateway.get_equity()
        report["account_orders"] = gateway.get_orders_on(day)
    except Exception as e:
        logger.warning("[shadow] 口座アプリを読めない。注文の数えだけ書く: %r", e, exc_info=True)
        report["account_error"] = repr(e)[:300]
        return report
    report["positions_diff"] = _positions_diff(paper_positions, account_positions)
    report["equity"] = {
        "paper": round(paper_equity, 2),
        "account": round(account_equity, 2),
        "diff": round(account_equity - paper_equity, 2),
    }
    return report


def write_shadow_report(day: date, report: dict[str, Any], out_dir: str | None = None) -> str:
    out = out_dir or shadow_log_dir()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{day.isoformat()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return path
