"""入れ替えの後の見張りが、市場と取引日ごとの並べで順位相関を測ることのテスト（#2159）。

jp と us は同じ predicted_at で書かれる。市場の予想の水準の差を、銘柄の並べの腕に数えない。
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.types import SymbolTask
from src.orchestration.jobs import backtest_promotion_watch as pw
from src.prediction.backtest_promotion import old_production as op
from src.prediction.backtest_promotion import state as st
from src.reporting.backtest_promotion_trend import load_backtest_promotion_records

_BACKUPS = [
    "/models/unified/UnifiedStockXGBoost_20261001_223000.joblib",
    "/models/unified/UnifiedStockLightGBM_20261001_223000.joblib",
]
_ACTUAL = {
    "jp": [0.01, 0.02, 0.03, 0.04],
    "us": [-0.04, -0.03, -0.02, -0.01],
}
# 市場の水準は当たり、市場の中の並べは逆
_NEW = {
    "jp": [0.04, 0.03, 0.02, 0.01],
    "us": [-0.01, -0.02, -0.03, -0.04],
}
# 市場の水準は逆で、市場の中の並べは当たる
_OLD = {
    "jp": [-0.04, -0.03, -0.02, -0.01],
    "us": [0.01, 0.02, 0.03, 0.04],
}


def _rows() -> pd.DataFrame:
    """UTC 22:30（JST の翌朝 07:30）に jp と us を同じ時刻で予想した、5日の採点済みの行。"""
    return pd.DataFrame(
        [
            {
                "market": market,
                "symbol": f"{market}{i}",
                "predicted_at": f"202610{day:02d}_223000",
                "predicted_ratio": _NEW[market][i],
                "actual_ratio": _ACTUAL[market][i],
            }
            for day in range(2, 7)
            for market in ("jp", "us")
            for i in range(4)
        ]
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """状態の置き場を tmp_path に向け、DB・モデル・写しの戻しを差し替える。"""
    monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(tmp_path))
    days = pd.date_range("2026-09-28", "2026-10-08")
    monkeypatch.setattr(op, "load_unified_model", lambda name, model_dir=None: f"model:{name}")
    monkeypatch.setattr(
        op,
        "load_stock_features",
        lambda market, symbol, raise_on_error=False: pd.DataFrame(
            {"date": days, "f": 1.0, "y": 0.0, "market_encoded": 1}
        ),
    )
    monkeypatch.setattr(
        op,
        "load_raw_ohlcv",
        lambda market, symbol, start_date=None: pd.DataFrame({"Close": 100.0}, index=days),
    )
    monkeypatch.setattr(
        op,
        "ensemble_diff_ratio",
        lambda latest_X, price, market, symbol, *args, **kwargs: SimpleNamespace(
            diff_ratio=_OLD[market][int(symbol[-1])]
        ),
    )
    monkeypatch.setattr(pw, "_local_now", lambda: datetime(2026, 10, 8, tzinfo=timezone.utc))
    monkeypatch.setattr(pw, "load_prediction_accuracy", MagicMock(return_value=_rows()))
    monkeypatch.setattr(
        pw,
        "load_target_symbols",
        lambda: [SymbolTask(market=m, symbol=f"{m}{i}") for m in ("jp", "us") for i in range(4)],
    )
    monkeypatch.setattr(pw, "restore_backups", MagicMock())
    return tmp_path


def _write_old_state() -> None:
    """daily_rank_corr の無い、前の形の state.json を生の JSON で置く。"""
    path = st._state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    old_metrics = {"hit_rate_minus_majority": 0.05, "mae": 0.01, "rank_corr": 0.1}
    path.write_text(
        json.dumps(
            {
                "last_evaluated_version": "abc",
                "rolled_back_versions": [],
                "watch": {
                    "version": "oldver000000",
                    "swapped_at": "2026-10-01T22:30:00+09:00",
                    "swap_date": "2026-10-01",
                    "backups": _BACKUPS,
                    "backtest_candidate": old_metrics,
                    "backtest_old_production": old_metrics,
                },
            }
        ),
        encoding="utf-8",
    )


def test_measure_old_production_ranks_inside_each_market(env):
    result = op.measure_old_production(_BACKUPS, _rows())

    assert result["reason"] is None and result["n_compared"] == 40
    # 混ぜた順位相関は市場の水準を数え、向きが逆になる（値は JSON に残る）
    assert result["new"]["rank_corr"] > 0 > result["old"]["rank_corr"]
    assert result["new"]["daily_rank_corr"] == pytest.approx(-1.0)
    assert result["old"]["daily_rank_corr"] == pytest.approx(1.0)


def test_check_watch_reads_old_state_and_ranks_inside_each_market(env):
    _write_old_state()
    result: dict = {}

    pw._check_watch(st.load_state(), result)

    watch = result["watch"]
    assert watch["judged"] is True
    assert watch["live"]["n_baseline"] == 40
    assert watch["live"]["rank_corr"] > 0
    assert watch["live"]["daily_rank_corr"] == pytest.approx(-1.0)
    assert watch["old_production"]["old"]["daily_rank_corr"] == pytest.approx(1.0)
    # 的中率と上げ率は当たらず、市場の中の並べだけで見張りに当たる
    assert len(watch["triggered"]) == 1
    assert "順位相関" in watch["triggered"][0] and "0以下" in watch["triggered"][0]
    # 的中率は旧が下なので戻さない。見張りは1回で終わる
    assert watch["old_is_better"] is False and watch["rolled_back"] is False
    pw.restore_backups.assert_not_called()
    assert st.load_state().watch is None


def test_old_result_json_without_daily_rank_corr_is_still_readable(env):
    folder = env / "backtest_promotion"
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "outcome": "skipped",
        "watch": {
            "judged": True,
            "live": {"hit_rate_minus_majority": 0.0, "rank_corr": 0.1},
            "triggered": ["実測の順位相関が0以下（-0.1）"],
        },
    }
    (folder / "20261101_220000.json").write_text(json.dumps(record), encoding="utf-8")

    records = load_backtest_promotion_records(str(folder))

    assert [r["watch"] for r in records] == [record["watch"]]


def test_day_key_is_market_and_base_bar_date():
    rows = pd.DataFrame(
        {
            "market": ["jp", "us", "jp", "us"],
            # 10-02 22:30 UTC は両市場の引けの後。10-03 00:30 UTC は jp の寄り（前の日の足）
            "predicted_at": ["20261002_223000"] * 2 + ["20261003_003000"] * 2,
            "_day": ["2026-10-03"] * 4,
        }
    )

    got = op.with_base_market_day(rows)

    assert got["date"].tolist() == ["jp:2026-10-02", "us:2026-10-02"] * 2
    assert got["_day"].tolist() == ["2026-10-03"] * 4
    assert "date" not in rows.columns
