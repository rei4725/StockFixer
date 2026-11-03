"""最適パラメータの保存先を一時フォルダへ向ける道具（#1081）。

save_optimal_params_json は results/optimal_params.json に書き、まだ無ければ
config/ の初めの値を引き継ぐ。保存の中身を確かめるテストは
`with redirect_optimal_params(root):` の内（または `.start()` から `.stop()` まで）で保存し、
`<root>/config/optimal_params.json` を読めば、本物の results/ と config/ に触れない
（初めの値は無いものとして扱う）。
"""

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch


class redirect_optimal_params:  # noqa: N801 (patch と同じく関数のように使う)
    """保存先を `<root>/config/optimal_params.json`、初めの値を無い場所に向ける。"""

    def __init__(self, root: str) -> None:
        self.config_dir = os.path.join(root, "config")
        self.json_path = os.path.join(self.config_dir, "optimal_params.json")
        self._seed = Path(root) / "seed" / "optimal_params.json"
        self._stack = ExitStack()

    def start(self) -> str:
        self._stack.enter_context(
            patch.dict(os.environ, {"STOCKFIXER_RESULTS_DIR": self.config_dir})
        )
        self._stack.enter_context(
            patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", self._seed)
        )
        return self.json_path

    def stop(self) -> None:
        self._stack.close()

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
