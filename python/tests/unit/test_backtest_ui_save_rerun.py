"""バックテスト UI の最適化の結果を再実行の後でも保存できる（Issue 1019）。

Streamlit はボタンを押すたびに上から流し直し、そのとき前の回のボタンは False に戻る。
1回目の流しで実行ボタンだけ、2回目の流しで保存ボタンだけを押し、st.session_state を
2回の流しで引き継ぐ形で、保存の道に入ることを確かめる。
"""

import importlib.util
import sys
import types
from typing import Any, Dict
from unittest.mock import patch

from tests.unit.test_backtest_ui_slippage import (
    _RUN_OPT,
    _SAVE_OPT,
    _SLIPPAGE,
    _USE_DEFAULT_SLIPPAGE,
    _FakeStreamlit,
    _opt_result,
    _UiMixin,
)

from src.backtest.optimizer import persistence
from src.orchestration import port_wiring

_SYMBOL = "銘柄コード"


class _RerunMixin(_UiMixin):
    """st.session_state を流しの間で引き継ぐ。"""

    def setUp(self) -> None:
        super().setUp()
        self.session_state: Dict[str, Any] = {}

    def _rerun(self, overrides: Dict[str, Any]) -> types.ModuleType:
        fake_st = _FakeStreamlit(overrides)
        fake_st.session_state = self.session_state
        spec = importlib.util.spec_from_file_location("_run_backtest_ui_under_test", self.ui_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"streamlit": fake_st}):
            with patch.object(port_wiring, "wire_ports"):
                spec.loader.exec_module(module)
        return module


class TestSaveOnRerun(_RerunMixin):
    def test_save_button_on_next_run_saves_once(self):
        with patch(
            "src.backtest.optimizer.run_optimization", return_value=_opt_result()
        ) as mock_opt:
            self._rerun({_RUN_OPT: True})
            with patch.object(
                persistence,
                "save_optimal_params_json",
                wraps=persistence.save_optimal_params_json,
            ) as spy:
                self._rerun({_SAVE_OPT: True})

        mock_opt.assert_called_once()
        spy.assert_called_once()
        entry = self._read()["jp_7203"]
        self.assertAlmostEqual(entry["threshold"], 0.02)

    def test_save_uses_settings_of_the_run_not_changed_widgets(self):
        """実行の後にサイドバーを変えても、結果を出したときの銘柄とスリッページで保存する。"""
        with patch("src.backtest.optimizer.run_optimization", return_value=_opt_result()):
            self._rerun({_USE_DEFAULT_SLIPPAGE: False, _SLIPPAGE: 0.0, _RUN_OPT: True})
            self._rerun({_SYMBOL: "9984", _SAVE_OPT: True})

        saved = self._read()
        self.assertNotIn("jp_9984", saved)
        entry = saved["jp_7203"]
        self.assertEqual(entry["slippage"], 0.0)
        self.assertIs(entry["slippage_is_market_default"], False)

    def test_save_without_run_does_not_save(self):
        with patch.object(persistence, "save_optimal_params_json") as spy:
            self._rerun({_SAVE_OPT: True})
        spy.assert_not_called()

    def test_new_run_replaces_kept_result(self):
        with patch(
            "src.backtest.optimizer.run_optimization", return_value=_opt_result()
        ) as mock_opt:
            self._rerun({_RUN_OPT: True})
        with patch("src.backtest.optimizer.run_optimization", return_value=_opt_result().head(0)):
            self._rerun({_RUN_OPT: True})
        mock_opt.assert_called_once()
        with patch.object(persistence, "save_optimal_params_json") as spy:
            self._rerun({_SAVE_OPT: True})
        spy.assert_not_called()
