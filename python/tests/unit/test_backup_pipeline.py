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
            (tmp_path / name).mkdir()

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
            (tmp_path / name).mkdir()

        deleted = _prune_old_backups(str(tmp_path), 5)

        assert deleted == 0
        assert len(os.listdir(tmp_path)) == 2
