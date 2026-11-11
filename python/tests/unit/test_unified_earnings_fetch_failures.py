"""統合の学習で決算日が取れなかった銘柄を数え、線を超えたら学習を止めることのテスト（#1700）。"""

import logging
from pathlib import Path
from typing import Callable, Iterable
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.market_data import loader
from src.market_data.prediction_adapter import PredictionMarketDataAdapter
from src.market_data.technical import add_earnings_flag
from src.prediction import unified_model_pipeline as ump
from src.prediction.manager import ModelManager

_N_ROWS = 30
_DATES = pd.bdate_range("2026-01-01", periods=_N_ROWS)
_EARNINGS_DAY = _DATES[15]
_OLD_BYTES = b"production model"


def _table(n_symbols: int, market: str = "jp", prefix: str = "S") -> pd.DataFrame:
    """1銘柄 _N_ROWS 行の銘柄を n_symbols 本並べた表。"""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_symbols):
        frame = pd.DataFrame(
            {
                "market": market,
                "symbol": f"{prefix}{i:03d}",
                "date": _DATES,
                "f1": rng.normal(size=_N_ROWS),
            }
        )
        frame["y"] = frame["f1"] * 0.01
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _mixed_table(n_jp: int, n_us: int) -> pd.DataFrame:
    """jp の銘柄（S000〜）と us の銘柄（U000〜）を並べた表。"""
    return pd.concat([_table(n_jp), _table(n_us, "us", "U")], ignore_index=True)


def _port(failed: Iterable[str] = (), no_dates: Iterable[str] = ()) -> MagicMock:
    """本物と同じく、raise_on_error が無ければ失敗を空にして返す port。"""
    failed_set, no_dates_set = set(failed), set(no_dates)

    def earnings_dates(market: str, symbol: str, raise_on_error: bool = False) -> pd.DatetimeIndex:
        if symbol in failed_set:
            if raise_on_error:
                raise ConnectionError("Too Many Requests")
            return pd.DatetimeIndex([])
        if symbol in no_dates_set:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex([_EARNINGS_DAY])

    port = MagicMock()
    port.get_earnings_dates.side_effect = earnings_dates
    port.add_earnings_flag.side_effect = add_earnings_flag
    return port


def _rows_kept(masked: pd.DataFrame, symbol: str) -> int:
    return int((masked["symbol"] == symbol).sum())


@pytest.fixture
def small_training(monkeypatch: pytest.MonkeyPatch) -> Callable[[pd.DataFrame, MagicMock], None]:
    """読み込みの結果と port を差し替え、木の少ないモデルで学習するようにする。"""
    monkeypatch.setattr(ump, "load_excluded_features", lambda *a, **k: set())
    monkeypatch.setattr(ump, "any_sentiment_source_configured", lambda markets: True)
    orig_create = ModelManager.create_model

    def small_create(self, model_type, model_name, **kwargs):
        return orig_create(self, model_type, model_name, n_estimators=5, **kwargs)

    monkeypatch.setattr(ModelManager, "create_model", small_create)

    def use(df: pd.DataFrame, port: MagicMock) -> None:
        monkeypatch.setattr(ump, "load_all_stock_features", lambda: df.copy())
        monkeypatch.setattr(ump, "get_market_data_port", lambda: port)

    return use


def _saved_model(save_dir: Path) -> Path:
    """今の本番のファイルに見立てて、中身の分かるファイルを置く。"""
    path = save_dir / "T.joblib"
    path.write_bytes(_OLD_BYTES)
    return path


def _train(save_dir: Path) -> None:
    ump.train_unified_model(model_name="T", save_dir=str(save_dir), save_selection=False)


def test_counts_failed_symbols_in_warning_and_keeps_training(
    small_training, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """一部の銘柄の取得が例外なら、数を WARNING に出して学習を続ける。"""
    small_training(_table(25), _port(failed=["S003"]))
    saved = _saved_model(tmp_path)

    with caplog.at_level(logging.WARNING, logger=ump.__name__):
        _train(tmp_path)

    assert saved.read_bytes() != _OLD_BYTES
    messages = [r.getMessage() for r in caplog.records if r.name == ump.__name__]
    failed_lines = [m for m in messages if "決算日の取得に失敗した銘柄: 1/25" in m]
    assert len(failed_lines) == 1
    assert "jp/S003" in failed_lines[0]


def test_failed_symbol_keeps_all_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """取れなかった銘柄は除く日が無いので全行残り、他の銘柄の除き方は変わらない。"""
    monkeypatch.setattr(ump, "get_market_data_port", lambda: _port(failed=["S003"]))

    masked = ump._mask_earnings_rows_for_unified(_table(25))

    assert _rows_kept(masked, "S003") == _N_ROWS
    assert _rows_kept(masked, "S004") == _N_ROWS - 7


@pytest.mark.parametrize(("n_failed", "n_symbols"), [(2, 25), (40, 788), (2, 2)])
def test_raises_and_keeps_model_file_when_failures_cross_line(
    small_training, tmp_path: Path, n_failed: int, n_symbols: int
) -> None:
    """失敗の割合が線を超えたら投げ、置いてあるモデルのファイルを上書きしない。"""
    failed = [f"S{i:03d}" for i in range(n_failed)]
    table = _table(n_symbols)
    small_training(table, _port(failed=failed))
    saved = _saved_model(tmp_path)

    with pytest.raises(RuntimeError, match=f"{n_failed}/{n_symbols}"):
        _train(tmp_path)

    assert saved.read_bytes() == _OLD_BYTES
    assert [p.name for p in tmp_path.iterdir()] == ["T.joblib"]


def test_does_not_create_a_model_file_when_failures_cross_line(
    small_training, tmp_path: Path
) -> None:
    """モデルのファイルが無いときも、線を超えた回は何も保存しない。"""
    small_training(_table(2), _port(failed=["S000", "S001"]))

    with pytest.raises(RuntimeError):
        _train(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_failures_exactly_on_line_do_not_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """線ちょうどは止めない（線を超えたときだけ止める）。"""
    monkeypatch.setattr(ump, "_EARNINGS_FETCH_MAX_FAILED_RATIO", 0.05)
    monkeypatch.setattr(ump, "get_market_data_port", lambda: _port(failed=["S000"]))

    masked = ump._mask_earnings_rows_for_unified(_table(20))

    assert masked["symbol"].nunique() == 20


def test_line_stops_a_rate_limit_but_not_a_short_blip_at_production_size() -> None:
    """今の本番の 788 銘柄で、39 銘柄までは止めず 40 銘柄で止める線である。"""
    assert 39 / 788 <= ump._EARNINGS_FETCH_MAX_FAILED_RATIO < 40 / 788


def test_raises_without_saving_when_only_jp_crosses_line(small_training, tmp_path: Path) -> None:
    """全市場では線の内でも、jp だけで線を超えたら投げ、モデルを保存しない（#1705）。"""
    small_training(_mixed_table(10, 50), _port(failed=["S000", "S001"]))
    saved = _saved_model(tmp_path)
    model = MagicMock()

    with (
        patch.object(ModelManager, "create_model", return_value=model),
        pytest.raises(RuntimeError, match="jp .*2/10"),
    ):
        _train(tmp_path)

    model.save_model.assert_not_called()
    assert saved.read_bytes() == _OLD_BYTES


def test_jp_line_at_production_size() -> None:
    """本番の 792 銘柄（jp 249）で、全部 jp の失敗は 12 銘柄まで通り 13 銘柄で止まる（#1705）。"""
    jp_failed = [f"jp/S{i:03d}" for i in range(13)]

    ump._check_earnings_fetch_failures(jp_failed[:12], 792, 249)
    with pytest.raises(RuntimeError, match="13/249"):
        ump._check_earnings_fetch_failures(jp_failed, 792, 249)


def test_warns_and_keeps_training_when_all_and_jp_are_within_line(
    small_training, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """全市場でも jp でも線の内なら、WARNING を出して学習を続ける（#1705）。"""
    small_training(_mixed_table(25, 15), _port(failed=["S000"]))
    saved = _saved_model(tmp_path)

    with caplog.at_level(logging.WARNING, logger=ump.__name__):
        _train(tmp_path)

    assert saved.read_bytes() != _OLD_BYTES
    assert any("決算日の取得に失敗した銘柄: 1/40" in r.getMessage() for r in caplog.records)


def test_jp_exactly_on_line_does_not_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """jp の割合が線ちょうどなら止めない（線を超えたときだけ止める）。"""
    monkeypatch.setattr(ump, "get_market_data_port", lambda: _port(failed=["S000"]))

    masked = ump._mask_earnings_rows_for_unified(_mixed_table(20, 20))

    assert masked["symbol"].nunique() == 40


def test_table_without_jp_does_not_stop_on_jp_line(small_training, tmp_path: Path) -> None:
    """jp の銘柄が無い表では、jp の線を見ずに学習を続ける（#1705）。"""
    small_training(_table(25, "us", "U"), _port(failed=["U003"]))
    saved = _saved_model(tmp_path)

    _train(tmp_path)

    assert saved.read_bytes() != _OLD_BYTES


def test_us_only_failures_within_all_market_line_keep_training(
    small_training, tmp_path: Path
) -> None:
    """us の失敗だけなら、us の中で 5% を超えても全市場が線の内なら学習を続ける（#1705）。"""
    small_training(_mixed_table(30, 10), _port(failed=["U000"]))
    saved = _saved_model(tmp_path)

    _train(tmp_path)

    assert saved.read_bytes() != _OLD_BYTES


def test_empty_responses_are_not_failures(
    small_training, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """全銘柄の応答が空でも失敗に数えず、今までどおり全行で学習する。"""
    table = _table(3)
    port = _port(no_dates=table["symbol"].unique())
    small_training(table, port)
    saved = _saved_model(tmp_path)

    with caplog.at_level(logging.WARNING, logger=ump.__name__):
        _train(tmp_path)

    assert saved.read_bytes() != _OLD_BYTES
    assert not any("決算日の取得に失敗" in r.getMessage() for r in caplog.records)


def test_removes_the_same_rows_as_before_without_failures(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """失敗の無い回は、決算の前後 EARNINGS_MASK_WINDOW_DAYS 営業日だけを除く。"""
    no_dates = "S001"
    monkeypatch.setattr(ump, "get_market_data_port", lambda: _port(no_dates=[no_dates]))
    table = _table(2)

    with caplog.at_level(logging.WARNING, logger=ump.__name__):
        masked = ump._mask_earnings_rows_for_unified(table)

    expected_days = [
        d for d in _DATES if abs(_DATES.get_loc(d) - 15) > ump.EARNINGS_MASK_WINDOW_DAYS
    ]
    kept_days = pd.DatetimeIndex(masked.loc[masked["symbol"] == "S000", "date"])
    assert list(kept_days) == expected_days
    assert _rows_kept(masked, no_dates) == _N_ROWS
    assert list(masked.columns) == list(table.columns)
    assert not caplog.records


def _ticker(earnings: object) -> MagicMock:
    ticker = MagicMock()
    ticker.get_earnings_dates.side_effect = earnings
    return ticker


def test_loader_default_still_returns_empty_on_failure() -> None:
    """既定の道は今までどおり、例外を空にして返す。"""
    with patch.object(loader.yf, "Ticker", return_value=_ticker(ConnectionError("boom"))):
        assert len(loader.get_earnings_dates("jp", "7203")) == 0


def test_loader_raises_when_asked() -> None:
    """raise_on_error=True なら、取得の例外を呼び手まで届ける。"""
    with (
        patch.object(loader.yf, "Ticker", return_value=_ticker(ConnectionError("boom"))),
        pytest.raises(ConnectionError),
    ):
        loader.get_earnings_dates("jp", "7203", raise_on_error=True)


@pytest.mark.parametrize("empty", [None, pd.DataFrame()], ids=["None", "empty"])
def test_loader_does_not_raise_on_empty_response(empty: object) -> None:
    """raise_on_error=True でも、空の応答は例外にしない。"""
    ticker = MagicMock()
    ticker.get_earnings_dates.return_value = empty
    with patch.object(loader.yf, "Ticker", return_value=ticker):
        assert len(loader.get_earnings_dates("jp", "1570", raise_on_error=True)) == 0


def test_adapter_passes_raise_on_error_through() -> None:
    """adapter が raise_on_error を loader まで渡す。"""
    with (
        patch.object(loader.yf, "Ticker", return_value=_ticker(ConnectionError("boom"))),
        pytest.raises(ConnectionError),
    ):
        PredictionMarketDataAdapter().get_earnings_dates("jp", "7203", raise_on_error=True)
