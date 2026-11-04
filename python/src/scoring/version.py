"""アプリのバージョン文字列の読み出し（BC をまたげないので factory_report と同じ読み方を写す）。"""

from __future__ import annotations

from pathlib import Path


def read_app_version(path: Path | None = None) -> str | None:
    """`python/VERSION` を返す（読めない・空なら None）。

    utf-8-sig で読むのは BOM 対策（PowerShell 5.1 の Set-Content が BOM を付けうる）。
    """
    version_path = path or Path(__file__).resolve().parents[2] / "VERSION"
    try:
        with open(version_path, encoding="utf-8-sig") as f:
            return f.read().strip() or None
    except OSError:
        return None
