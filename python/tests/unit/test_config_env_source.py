"""pytest の実行で本番の .env を読まないことの確認（#1017 の再発防止）

デプロイの機械には本番の python/.env があり、CI には無い。.env を読むと
同じテストが場所によって違う結果になり、テストから本番の資源（口座アプリ・
Discord・healthchecks 等）に届く。tests/conftest.py が STOCKFIXER_IGNORE_DOTENV を
立て、設定のクラスと load_dotenv はそれを見て .env を読まない。
"""

import pytest

import config.data as data_cfg
import config.execution as execution_cfg
import config.feature_flags as flags_cfg
import config.settings as settings_cfg
from config.env_source import dotenv_path


def test_dotenv_path_is_none_when_ignore_flag_set(monkeypatch):
    monkeypatch.setenv("STOCKFIXER_IGNORE_DOTENV", "1")
    assert dotenv_path() is None


def test_dotenv_path_is_dotenv_when_flag_unset(monkeypatch):
    monkeypatch.delenv("STOCKFIXER_IGNORE_DOTENV", raising=False)
    assert dotenv_path() == ".env"


def test_pytest_session_sets_ignore_flag():
    import os

    assert os.environ.get("STOCKFIXER_IGNORE_DOTENV") == "1"


@pytest.mark.parametrize(
    "settings_cls",
    [
        settings_cfg.Settings,
        flags_cfg.FeatureFlags,
        data_cfg.DataSettings,
        execution_cfg.ExecutionSettings,
    ],
)
def test_settings_classes_do_not_read_dotenv_in_pytest(settings_cls):
    assert settings_cls.model_config.get("env_file") is None


def test_settings_ignore_dotenv_file_in_cwd(tmp_path, monkeypatch):
    """cwd に .env があっても、pytest の中では値を拾わない"""
    (tmp_path / ".env").write_text("ACCOUNT_SHADOW=true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACCOUNT_SHADOW", raising=False)

    assert flags_cfg.FeatureFlags().ACCOUNT_SHADOW is False
