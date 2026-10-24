"""バックテスト UI のスリッページの既定と最適パラメータの保存（Issue 1005）。

UI の既定では市場別の既定（slippage=None）を渡し、0.0 は人間が明示したときだけ渡す。
「JSON に保存」は save_optimal_params_json を通し、当てたスリッページの率を残す。

CI の環境に streamlit は無いため、ウィジェットが既定の値を返す偽の streamlit で
run_backtest_ui.py を上から流し、押したボタンの道だけを動かす。
"""

import glob
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch

import pandas as pd

from src.backtest.optimizer import persistence
from src.orchestration import port_wiring

_UI_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "run_backtest_ui.py")

_RUN_SINGLE = "▶ バックテスト実行"
_RUN_WF = "▶ Walk-Forward 実行"
_RUN_OPT = "▶ 最適化実行"
_SAVE_OPT = "💾 最適パラメータを JSON に保存"
_USE_DEFAULT_SLIPPAGE = "市場別の既定スリッページを使う"
_SLIPPAGE = "スリッページ"


class _Ctx:
    """with で使え、どの描画の呼び出しも黙って受ける入れ物。"""

    def __enter__(self) -> "_Ctx":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return _noop


def _noop(*args: Any, **kwargs: Any) -> _Ctx:
    return _Ctx()


class _FakeStreamlit(types.ModuleType):
    """ウィジェットは既定の値（overrides にあればその値）を返す。"""

    def __init__(self, overrides: Dict[str, Any]) -> None:
        super().__init__("streamlit")
        self._overrides = overrides
        self.session_state: Dict[str, Any] = {}
        self.sidebar = _Ctx()

    def __getattr__(self, name: str) -> Any:
        return _noop

    def _pick(self, label: str, default: Any) -> Any:
        return self._overrides.get(label, default)

    def number_input(self, label: str, *args: Any, value: Any = None, **kwargs: Any) -> Any:
        return self._pick(label, value)

    def slider(self, label: str, *args: Any, value: Any = None, **kwargs: Any) -> Any:
        return self._pick(label, value)

    def text_input(self, label: str, value: str = "", **kwargs: Any) -> Any:
        return self._pick(label, value)

    def date_input(self, label: str, value: Any = None, **kwargs: Any) -> Any:
        return self._pick(label, value)

    def checkbox(self, label: str, value: bool = False, **kwargs: Any) -> Any:
        return self._pick(label, value)

    def selectbox(self, label: str, options: Any, index: int = 0, **kwargs: Any) -> Any:
        return self._pick(label, list(options)[index])

    def button(self, label: str, *args: Any, **kwargs: Any) -> bool:
        return bool(self._pick(label, False))

    def tabs(self, labels: Any) -> list:
        return [_Ctx() for _ in labels]

    def columns(self, spec: Any, **kwargs: Any) -> list:
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def cache_data(self, *args: Any, **kwargs: Any) -> Any:
        if args and callable(args[0]):
            return args[0]
        return lambda func: func


def _opt_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "threshold": [0.01, 0.02],
            "stop_loss_pct": [None, None],
            "take_profit_pct": [None, None],
            "sharpe_ratio": [0.8, 1.5],
            "num_trades": [40, 50],
        }
    )


class _UiMixin(unittest.TestCase):
    """run_backtest_ui.py を一時フォルダへ写して流す（保存先 config/ も一時フォルダ）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        self.ui_path = os.path.join(root, "run_backtest_ui.py")
        shutil.copyfile(_UI_PATH, self.ui_path)
        # persistence も同じ root/config を向く（dirname を4回たどると root）
        fake_file = os.path.join(root, "src", "backtest", "optimizer", "persistence.py")
        file_patch = patch.object(persistence, "__file__", fake_file)
        file_patch.start()
        self.addCleanup(file_patch.stop)
        self.config_dir = os.path.join(root, "config")
        os.makedirs(self.config_dir, exist_ok=True)
        self.json_path = os.path.join(self.config_dir, "optimal_params.json")

    def _run_ui(self, overrides: Optional[Dict[str, Any]] = None) -> types.ModuleType:
        fake_st = _FakeStreamlit(overrides or {})
        spec = importlib.util.spec_from_file_location("_run_backtest_ui_under_test", self.ui_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"streamlit": fake_st}):
            with patch.object(port_wiring, "wire_ports"):
                spec.loader.exec_module(module)
        return module

    def _read(self) -> dict:
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)


class TestDefaultSlippageIsMarketDefault(_UiMixin):
    def test_single_backtest_gets_none_by_default(self):
        with patch(
            "src.backtest.pipeline.run_backtest_single",
            return_value=(pd.DataFrame(), {}, pd.Series(dtype=float)),
        ) as mock_single:
            self._run_ui({_RUN_SINGLE: True})
        mock_single.assert_called_once()
        self.assertIsNone(mock_single.call_args.kwargs["slippage"])

    def test_walk_forward_gets_none_by_default(self):
        with patch(
            "src.backtest.pipeline.run_backtest_walk_forward",
            return_value=(None, None, pd.DataFrame()),
        ) as mock_wf:
            self._run_ui({_RUN_WF: True})
        mock_wf.assert_called_once()
        self.assertIsNone(mock_wf.call_args.kwargs["slippage"])

    def test_optimization_gets_none_by_default(self):
        with patch(
            "src.backtest.optimizer.run_optimization", return_value=pd.DataFrame()
        ) as mock_opt:
            self._run_ui({_RUN_OPT: True})
        mock_opt.assert_called_once()
        self.assertIsNone(mock_opt.call_args.kwargs["slippage"])

    def test_initial_screen_renders_with_default(self):
        module = self._run_ui()
        self.assertIsNone(module.slippage)


class TestExplicitSlippageIsRespected(_UiMixin):
    _EXPLICIT_ZERO = {_USE_DEFAULT_SLIPPAGE: False, _SLIPPAGE: 0.0}

    def test_explicit_zero_is_passed_to_all_three(self):
        with (
            patch(
                "src.backtest.pipeline.run_backtest_single",
                return_value=(pd.DataFrame(), {}, pd.Series(dtype=float)),
            ) as mock_single,
            patch(
                "src.backtest.pipeline.run_backtest_walk_forward",
                return_value=(None, None, pd.DataFrame()),
            ) as mock_wf,
            patch(
                "src.backtest.optimizer.run_optimization", return_value=pd.DataFrame()
            ) as mock_opt,
        ):
            self._run_ui({**self._EXPLICIT_ZERO, _RUN_SINGLE: True, _RUN_WF: True, _RUN_OPT: True})
        for mock in (mock_single, mock_wf, mock_opt):
            slippage = mock.call_args.kwargs["slippage"]
            self.assertEqual(slippage, 0.0)
            self.assertIsNotNone(slippage)

    def test_explicit_rate_is_passed(self):
        with patch(
            "src.backtest.optimizer.run_optimization", return_value=pd.DataFrame()
        ) as mock_opt:
            self._run_ui({_USE_DEFAULT_SLIPPAGE: False, _SLIPPAGE: 0.002, _RUN_OPT: True})
        self.assertEqual(mock_opt.call_args.kwargs["slippage"], 0.002)


class TestSaveOptimalParamsRecordsSlippage(_UiMixin):
    def _save(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        with patch("src.backtest.optimizer.run_optimization", return_value=_opt_result()):
            self._run_ui({**(overrides or {}), _RUN_OPT: True, _SAVE_OPT: True})

    def test_default_records_market_default_rate_and_keeps_other_symbols(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"jp_1301": {"threshold": 0.01}}, f)

        with patch.object(
            persistence,
            "save_optimal_params_json",
            wraps=persistence.save_optimal_params_json,
        ) as spy:
            self._save()

        spy.assert_called_once()
        saved = self._read()
        self.assertEqual(saved["jp_1301"], {"threshold": 0.01})
        entry = saved["jp_7203"]
        self.assertAlmostEqual(entry["threshold"], 0.02)
        self.assertAlmostEqual(entry["slippage"], 0.001)
        self.assertIs(entry["slippage_is_market_default"], True)

    def test_explicit_zero_is_recorded_as_not_default(self):
        self._save({_USE_DEFAULT_SLIPPAGE: False, _SLIPPAGE: 0.0})

        entry = self._read()["jp_7203"]
        self.assertEqual(entry["slippage"], 0.0)
        self.assertIs(entry["slippage_is_market_default"], False)

    def test_corrupted_json_is_not_overwritten_with_only_this_symbol(self):
        broken = '{"jp_1301": {"threshold": 0.01}, "jp_13'
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write(broken)

        self._save()

        quarantined = glob.glob(f"{self.json_path}.corrupt-*")
        self.assertEqual(len(quarantined), 1)
        with open(quarantined[0], "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), broken)
        self.assertFalse(os.path.exists(self.json_path))


if __name__ == "__main__":
    unittest.main()
