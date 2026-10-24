"""
設定のクラスと load_dotenv が読む .env の場所

STOCKFIXER_IGNORE_DOTENV=1 のときは .env を読まない。tests/conftest.py が
pytest の起動時に立て、デプロイの機械の本番の .env がテストに漏れ込むのを防ぐ
（CI には .env が無く、同じテストが場所で結果を変えていた。#1017）。
環境変数そのものによる上書きは、立てていても効く。
"""

import os

IGNORE_DOTENV_ENV = "STOCKFIXER_IGNORE_DOTENV"


def dotenv_path() -> str | None:
    """読む .env のパス。STOCKFIXER_IGNORE_DOTENV=1 なら None（読まない）"""
    if os.environ.get(IGNORE_DOTENV_ENV) == "1":
        return None
    return ".env"
