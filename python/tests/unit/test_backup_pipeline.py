"""PostgreSQL pg_dump ベースのバックアップパイプラインの単体テスト。

- _run_pg_dump() が pg_dump をカスタムフォーマット(-Fc)で呼び出し、
  パスワードをコマンドライン引数ではなく PGPASSWORD 環境変数で渡すこと
- pg_dump が失敗した場合に RuntimeError を送出すること
- run_db_backup() が pg_dump 経由でバックアップを作成し、
  失敗時はエラーを dict に格納して非致命的に処理すること
- _prune_old_backups() が世代管理ロジックとして引き続き機能すること
  （DBエンジンに依存しないため無変更で流用）
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

from src.orchestration.backup_pipeline import _prune_old_backups, _run_pg_dump, run_db_backup


def _fake_completed_process(returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = stderr
    return proc


class TestRunPgDump:
    def test_builds_command_with_fc_and_connection_params(self):
        with patch(
            "src.orchestration.backup_pipeline.subprocess.run",
            return_value=_fake_completed_process(0),
        ) as mock_run:
            _run_pg_dump(
                "postgresql://stockfixer:secret_pw@dbhost:5433/stockfixer", "/tmp/out.dump"
            )

        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "pg_dump"
        assert "-Fc" in cmd
        assert cmd[cmd.index("-h") + 1] == "dbhost"
        assert cmd[cmd.index("-p") + 1] == "5433"
        assert cmd[cmd.index("-U") + 1] == "stockfixer"
        assert cmd[cmd.index("-f") + 1] == "/tmp/out.dump"
        assert cmd[-1] == "stockfixer"

    def test_password_passed_via_env_not_command_line(self):
        with patch(
            "src.orchestration.backup_pipeline.subprocess.run",
            return_value=_fake_completed_process(0),
        ) as mock_run:
            _run_pg_dump(
                "postgresql://stockfixer:super_secret@dbhost:5432/stockfixer", "/tmp/out.dump"
            )

        cmd = mock_run.call_args.args[0]
        assert "super_secret" not in cmd
        assert not any("super_secret" in str(part) for part in cmd)
        env = mock_run.call_args.kwargs["env"]
        assert env["PGPASSWORD"] == "super_secret"

    def test_raises_runtime_error_on_nonzero_exit(self):
        with patch(
            "src.orchestration.backup_pipeline.subprocess.run",
            return_value=_fake_completed_process(1, stderr="connection refused"),
        ):
            try:
                _run_pg_dump("postgresql://u:p@h:5432/db", "/tmp/out.dump")
            except RuntimeError as e:
                assert "connection refused" in str(e)
            else:
                raise AssertionError("RuntimeError が送出されなかった")

    def test_passes_timeout_to_subprocess_run(self):
        with patch(
            "src.orchestration.backup_pipeline.subprocess.run",
            return_value=_fake_completed_process(0),
        ) as mock_run:
            _run_pg_dump("postgresql://u:p@h:5432/db", "/tmp/out.dump")

        assert mock_run.call_args.kwargs["timeout"] == 300


class TestRunDbBackup:
    def test_success_returns_dump_path_and_no_error(self, tmp_path):
        with (
            patch(
                "src.orchestration.backup_pipeline.get_backup_dir",
                return_value=str(tmp_path / "backups"),
            ),
            patch(
                "src.orchestration.backup_pipeline._run_pg_dump",
                side_effect=lambda url, dest: open(dest, "wb").write(b"x" * 2048),
            ),
            patch(
                "src.orchestration.backup_pipeline.get_database_url",
                return_value="postgresql://u:p@h:5432/db",
            ),
        ):
            result = run_db_backup()

        assert result["error"] is None
        assert result["backup_path"].endswith("stockfixer.dump")
        assert os.path.isfile(result["backup_path"])
        assert result["size_mb"] > 0
        assert result["pruned_count"] == 0
        assert result["elapsed_seconds"] >= 0

    def test_failure_captures_error_and_does_not_raise(self, tmp_path):
        with (
            patch(
                "src.orchestration.backup_pipeline.get_backup_dir",
                return_value=str(tmp_path / "backups"),
            ),
            patch(
                "src.orchestration.backup_pipeline._run_pg_dump",
                side_effect=RuntimeError("pg_dump 失敗 (code=1): auth failed"),
            ),
            patch(
                "src.orchestration.backup_pipeline.get_database_url",
                return_value="postgresql://u:p@h:5432/db",
            ),
        ):
            result = run_db_backup()

        assert result["error"] is not None
        assert "auth failed" in result["error"]
        assert result["size_mb"] == 0.0
        assert result["pruned_count"] == 0

    def _run_failing_backup(self, backup_root, side_effect):
        def _fail(url, dest):
            open(dest, "wb").write(b"partial")
            raise side_effect

        with (
            patch(
                "src.orchestration.backup_pipeline.get_backup_dir",
                return_value=str(backup_root),
            ),
            patch("src.orchestration.backup_pipeline._run_pg_dump", side_effect=_fail),
            patch(
                "src.orchestration.backup_pipeline.get_database_url",
                return_value="postgresql://u:p@h:5432/db",
            ),
        ):
            return run_db_backup()

    def test_failure_removes_folder_of_this_run(self, tmp_path):
        backup_root = tmp_path / "backups"
        _write_good_backup(backup_root / "20260101_000000")

        result = self._run_failing_backup(backup_root, RuntimeError("pg_dump 失敗 (code=1)"))

        assert result["error"] is not None
        assert not os.path.exists(os.path.dirname(result["backup_path"]))
        assert os.listdir(backup_root) == ["20260101_000000"]

    def test_timeout_removes_folder_of_this_run(self, tmp_path):
        backup_root = tmp_path / "backups"

        result = self._run_failing_backup(
            backup_root, subprocess.TimeoutExpired(cmd="pg_dump", timeout=300)
        )

        assert result["error"] is not None
        assert not os.path.exists(os.path.dirname(result["backup_path"]))
        assert os.listdir(backup_root) == []

    def test_success_keeps_newest_five_good_generations_among_failure_leftovers(self, tmp_path):
        backup_root = tmp_path / "backups"
        good = [
            "20260101_000000",
            "20260103_000000",
            "20260105_000000",
            "20260107_000000",
            "20260109_000000",
        ]
        for name in good:
            _write_good_backup(backup_root / name)
        (backup_root / "20260110_000000").mkdir()
        (backup_root / "20260111_000000").mkdir()
        (backup_root / "20260111_000000" / "other.txt").write_bytes(b"x")
        (backup_root / "20260112_000000").mkdir()
        (backup_root / "20260112_000000" / "stockfixer.dump").write_bytes(b"")

        with (
            patch(
                "src.orchestration.backup_pipeline.get_backup_dir", return_value=str(backup_root)
            ),
            patch(
                "src.orchestration.backup_pipeline._run_pg_dump",
                side_effect=lambda url, dest: open(dest, "wb").write(b"x" * 2048),
            ),
            patch(
                "src.orchestration.backup_pipeline.get_database_url",
                return_value="postgresql://u:p@h:5432/db",
            ),
        ):
            result = run_db_backup()

        assert result["error"] is None
        this_run = os.path.basename(os.path.dirname(result["backup_path"]))
        assert sorted(os.listdir(backup_root)) == good[1:] + [this_run]
        for name in good[1:]:
            assert (backup_root / name / "stockfixer.dump").read_bytes() == b"x" * 2048


def _write_good_backup(folder):
    folder.mkdir(parents=True)
    (folder / "stockfixer.dump").write_bytes(b"x" * 2048)


class TestPruneOldBackups:
    def test_returns_zero_when_backup_root_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert _prune_old_backups(str(missing), 5) == 0

    def test_deletes_oldest_beyond_max_generations(self, tmp_path):
        for name in [
            "20260101_000000",
            "20260102_000000",
            "20260103_000000",
            "20260104_000000",
            "20260105_000000",
            "20260106_000000",
            "20260107_000000",
        ]:
            _write_good_backup(tmp_path / name)

        deleted = _prune_old_backups(str(tmp_path), 5)

        assert deleted == 2
        remaining = sorted(os.listdir(tmp_path))
        assert remaining == [
            "20260103_000000",
            "20260104_000000",
            "20260105_000000",
            "20260106_000000",
            "20260107_000000",
        ]

    def test_no_deletion_when_within_max_generations(self, tmp_path):
        for name in ["20260101_000000", "20260102_000000"]:
            _write_good_backup(tmp_path / name)

        deleted = _prune_old_backups(str(tmp_path), 5)

        assert deleted == 0
        assert len(os.listdir(tmp_path)) == 2
