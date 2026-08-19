"""Unit tests for backend/showcases/file_transfer.py.

`subprocess.run`/`_docker_exec` are mocked, so this covers the module's own logic (command
construction, JSON parsing, counter parsing, error handling) rather than real container
behaviour, which needs a Docker-capable machine (see README.md).
"""
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.showcases import file_transfer as ft


@pytest.fixture(autouse=True)
def _non_root(monkeypatch):
    # Pin os.geteuid() so nsenter-command construction is deterministic
    # regardless of which user actually runs the test suite.
    monkeypatch.setattr(ft.os, "geteuid", lambda: 1000)


class TestSetCongestionControl:
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_skips_write_when_already_set(self, mock_docker_exec, mock_run):
        mock_docker_exec.return_value = MagicMock(returncode=0, stdout="cubic\n")
        ft.set_congestion_control("cubic", container="cosme-server")
        mock_run.assert_not_called()

    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_success_runs_expected_nsenter_command(self, mock_docker_exec, mock_run):
        mock_docker_exec.return_value = MagicMock(returncode=0, stdout="cubic\n")
        # first subprocess.run call: `docker inspect` for the host PID; second: the nsenter sysctl write.
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),
            MagicMock(returncode=0),
        ]
        ft.set_congestion_control("bbr", container="cosme-server")
        inspect_args = mock_run.call_args_list[0][0][0]
        nsenter_args = mock_run.call_args_list[1][0][0]
        assert inspect_args == ["docker", "inspect", "-f", "{{.State.Pid}}", "cosme-server"]
        assert nsenter_args == [
            "sudo", "-n", "nsenter", "-t", "12345", "-n", "/usr/sbin/sysctl", "-w",
            "net.ipv4.tcp_congestion_control=bbr",
        ]

    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_root_skips_sudo_prefix(self, mock_docker_exec, mock_run, monkeypatch):
        monkeypatch.setattr(ft.os, "geteuid", lambda: 0)
        mock_docker_exec.return_value = MagicMock(returncode=0, stdout="cubic\n")
        mock_run.side_effect = [MagicMock(returncode=0, stdout="12345\n"), MagicMock(returncode=0)]
        ft.set_congestion_control("bbr", container="cosme-server")
        nsenter_args = mock_run.call_args_list[1][0][0]
        assert nsenter_args == [
            "nsenter", "-t", "12345", "-n", "/usr/sbin/sysctl", "-w",
            "net.ipv4.tcp_congestion_control=bbr",
        ]

    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_failure_raises_showcase_error_with_bbr_hint(self, mock_docker_exec, mock_run):
        mock_docker_exec.return_value = MagicMock(returncode=0, stdout="cubic\n")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),
            subprocess.CalledProcessError(1, "sysctl", stderr="invalid value"),
        ]
        with pytest.raises(ft.ShowcaseError, match="tcp_bbr"):
            ft.set_congestion_control("bbr")

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_undetermined_current_cc_still_attempts_write(self, mock_docker_exec):
        # sysctl read itself fails -- must not be mistaken for "already set".
        mock_docker_exec.side_effect = subprocess.CalledProcessError(1, "sysctl")
        with patch("backend.showcases.file_transfer.subprocess.run") as mock_run:
            mock_run.side_effect = [MagicMock(returncode=0, stdout="1\n"), MagicMock(returncode=0)]
            ft.set_congestion_control("bbr")
            assert mock_run.call_count == 2


class TestEnsureHttpServerRunning:
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_skips_start_if_correct_server_already_running(self, mock_docker_exec, mock_run):
        mock_docker_exec.return_value = MagicMock(
            returncode=0, stdout=f"123 {' '.join(ft._HTTP_SERVER_CMD)}\n",
        )
        ft.ensure_http_server_running()
        mock_run.assert_not_called()  # no `docker exec -d` to (re)start it

    @patch("backend.showcases.file_transfer.time.sleep")
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_kills_stale_server_on_wrong_port_then_starts_new(self, mock_docker_exec, mock_run, mock_sleep):
        # pgrep finds a stale server (old port/dir) that must NOT satisfy the
        # "already running" check; then a kill call; then the post-start
        # curl probe succeeds on the first try.
        mock_docker_exec.side_effect = [
            MagicMock(returncode=0, stdout="99 python3 -m http.server 8080 --directory /srv/media\n"),
            MagicMock(returncode=0, stdout=""),  # the kill call
            MagicMock(returncode=0, stdout="200"),  # curl probe
        ]
        ft.ensure_http_server_running()
        kill_call = mock_docker_exec.call_args_list[1]
        assert kill_call[0][1:3] == ("kill", "99")
        start_cmd = mock_run.call_args_list[0][0][0]
        assert start_cmd == ["docker", "exec", "-d", ft.DEFAULT_SERVER_CONTAINER, *ft._HTTP_SERVER_CMD]

    @patch("backend.showcases.file_transfer.time.sleep")
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_raises_if_server_never_comes_up(self, mock_docker_exec, mock_run, mock_sleep):
        mock_docker_exec.side_effect = [
            MagicMock(returncode=1, stdout=""),  # pgrep: nothing running
        ] + [MagicMock(returncode=0, stdout="000")] * 5  # curl probe never answers
        with pytest.raises(ft.ShowcaseError, match="did not come up"):
            ft.ensure_http_server_running(retries=5, retry_interval_s=0)


class TestTcpRetransTotal:
    # /proc/net/snmp is two aligned lines: names, then values. RetransSegs is read BY NAME rather
    # than by column index, so a kernel that adds or reorders a counter cannot silently make this
    # report some neighbouring field.
    SNMP = (
        "Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens AttemptFails "
        "EstabResets CurrEstab InSegs OutSegs RetransSegs InErrs OutRsts InCsumErrors\n"
        "Tcp: 1 200 120000 -1 3 487 2 44 0 17346163 442847573 585525 0 2 0\n"
    )

    @patch("backend.showcases.file_transfer.subprocess.run")
    def test_reads_retranssegs_by_name(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=self.SNMP, stderr="")
        assert ft._tcp_retrans_total("cosme-server") == 585525

    @patch("backend.showcases.file_transfer.subprocess.run")
    def test_missing_counter_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert ft._tcp_retrans_total("cosme-server") is None

    @patch("backend.showcases.file_transfer.subprocess.run")
    def test_header_without_values_returns_none(self, mock_run):
        header_only = self.SNMP.splitlines()[0] + "\n"
        mock_run.return_value = MagicMock(returncode=0, stdout=header_only, stderr="")
        assert ft._tcp_retrans_total("cosme-server") is None


class TestLiveProgressHelpers:
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_content_length_parsed_from_head(self, mock_exec):
        mock_exec.return_value = MagicMock(stdout="HTTP/1.1 200 OK\r\nContent-Length: 12345\r\n\r\n")
        assert ft._content_length("cosme-client", "http://x/y") == 12345

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_content_length_missing_header_returns_none(self, mock_exec):
        mock_exec.return_value = MagicMock(stdout="HTTP/1.1 200 OK\r\n\r\n")
        assert ft._content_length("cosme-client", "http://x/y") is None

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_file_size_parses_stat_output(self, mock_exec):
        mock_exec.return_value = MagicMock(stdout="42\n")
        assert ft._file_size("cosme-client", "/tmp/x") == 42

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_file_size_defaults_to_zero_when_missing(self, mock_exec):
        mock_exec.return_value = MagicMock(stdout="0\n")
        assert ft._file_size("cosme-client", "/tmp/x") == 0

    @patch("backend.showcases.file_transfer.requests.post")
    def test_post_live_progress_is_best_effort(self, mock_post):
        mock_post.side_effect = requests.RequestException("unreachable")
        ft._post_live_progress(100, 1000, 5.0)  # must not raise
        mock_post.assert_called_once()


class TestEnsureSizedAsset:
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_builds_expected_idempotent_shell_script(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0)
        rel_path = ft._ensure_sized_asset("cosme-server", 214748365)
        assert rel_path == "sized/214748365.bin"
        args = mock_exec.call_args[0]
        assert args[0] == "cosme-server"
        assert args[1:3] == ("sh", "-c")
        script = args[3]
        # the guard must check the CACHED file's exact size before regenerating --
        # this is what makes repeated requests for the same size reuse the file
        # instead of rebuilding it every run.
        assert 'stat -c%s "/srv/media/sized/214748365.bin"' in script
        assert '!= "214748365"' in script
        assert 'truncate -s 214748365 "/srv/media/sized/214748365.bin"' in script
        assert '"/srv/media/bigbuckbunny.mp4"' in script
        # each cat is checked -- a real write failure (e.g. disk full) must exit
        # the loop immediately instead of spinning until the subprocess timeout
        # (regression: this is exactly what happened with a 5GB request against
        # a size-capped tmpfs mount before the mount was fixed to be disk-backed).
        assert '|| { echo' in script

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_timeout_scales_with_requested_size(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0)
        ft._ensure_sized_asset("cosme-server", 5 * 1024 ** 3)  # 5GB
        timeout_kwarg = mock_exec.call_args.kwargs["timeout_s"]
        assert timeout_kwarg > ft._SIZED_ASSET_MIN_TIMEOUT_S  # bigger than the fixed 120s floor

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_small_size_uses_the_minimum_timeout_floor(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0)
        ft._ensure_sized_asset("cosme-server", 1024)  # tiny -- must not underflow the floor
        timeout_kwarg = mock_exec.call_args.kwargs["timeout_s"]
        assert timeout_kwarg == ft._SIZED_ASSET_MIN_TIMEOUT_S

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_write_failure_raises_clear_showcase_error(self, mock_exec):
        mock_exec.side_effect = subprocess.CalledProcessError(1, "sh", stderr="No space left on device")
        with pytest.raises(ft.ShowcaseError, match="disk space"):
            ft._ensure_sized_asset("cosme-server", 5 * 1024 ** 3)

    @patch("backend.showcases.file_transfer._docker_exec")
    def test_generation_timeout_raises_clear_showcase_error(self, mock_exec):
        mock_exec.side_effect = subprocess.TimeoutExpired("sh", 120)
        with pytest.raises(ft.ShowcaseError, match="didn't finish within"):
            ft._ensure_sized_asset("cosme-server", 5 * 1024 ** 3)


class TestRunFileTransferShowcase:
    @patch("backend.showcases.file_transfer.ensure_containers_healthy")
    @patch("backend.showcases.file_transfer.time.sleep")
    @patch("backend.showcases.file_transfer.requests.post")
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._tcp_retrans_total", side_effect=[100, 102])
    @patch("backend.showcases.file_transfer.ensure_http_server_running")
    @patch("backend.showcases.file_transfer.set_congestion_control")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_parses_curl_output_into_result(self, mock_exec, mock_cc, mock_http, mock_retrans,
                                            mock_subproc_run, mock_post, mock_sleep, mock_healthy):
        curl_json = '{"time_total":2.5,"size_download":10000000,"speed_download":4000000,"http_code":200}'

        def exec_side_effect(container, *cmd, **kwargs):
            if cmd[0] == "curl":  # the HEAD request for total size
                return MagicMock(returncode=0, stdout="HTTP/1.1 200 OK\r\nContent-Length: 10000000\r\n\r\n")
            if cmd[0] == "rm":
                return MagicMock(returncode=0, stdout="")
            if cmd[0] == "sh":  # the stat-based size poll
                return MagicMock(returncode=0, stdout="10000000\n")
            if cmd[0] == "cat":  # the completion check -- done on the first poll
                return MagicMock(returncode=0, stdout=curl_json)
            raise AssertionError(f"unexpected _docker_exec call: {cmd}")

        mock_exec.side_effect = exec_side_effect
        result = ft.run_file_transfer_showcase(congestion_control="bbr")
        mock_cc.assert_called_once_with("bbr", ft.DEFAULT_SERVER_CONTAINER)
        assert result.duration_s == 2.5
        assert result.throughput_bps == pytest.approx(32_000_000)
        assert result.http_code == 200
        assert result.tcp_retransmits == 2
        assert result.congestion_control == "bbr"
        # the backgrounded curl launch went through plain subprocess.run, not _docker_exec
        mock_subproc_run.assert_called_once()
        assert mock_subproc_run.call_args[0][0][:3] == ["docker", "exec", "-d"]

    @patch("backend.showcases.file_transfer.ensure_containers_healthy")
    @patch("backend.showcases.file_transfer.time.sleep")
    @patch("backend.showcases.file_transfer.requests.post")
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer._tcp_retrans_total", return_value=None)
    @patch("backend.showcases.file_transfer.ensure_http_server_running")
    @patch("backend.showcases.file_transfer.set_congestion_control")
    @patch("backend.showcases.file_transfer._ensure_sized_asset")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_size_gb_downloads_a_generated_sized_asset_instead_of_asset_name(
        self, mock_exec, mock_ensure_sized, mock_cc, mock_http, mock_retrans, mock_subproc_run, mock_post, mock_sleep,
        mock_healthy,
    ):
        mock_ensure_sized.return_value = "sized/214748365.bin"
        curl_json = '{"time_total":0.3,"size_download":214748365,"speed_download":700000000,"http_code":200}'

        def exec_side_effect(container, *cmd, **kwargs):
            if cmd[0] == "curl":
                return MagicMock(returncode=0, stdout="HTTP/1.1 200 OK\r\nContent-Length: 214748365\r\n\r\n")
            if cmd[0] == "rm":
                return MagicMock(returncode=0, stdout="")
            if cmd[0] == "sh":
                return MagicMock(returncode=0, stdout="214748365\n")
            if cmd[0] == "cat":
                return MagicMock(returncode=0, stdout=curl_json)
            raise AssertionError(f"unexpected _docker_exec call: {cmd}")

        mock_exec.side_effect = exec_side_effect
        result = ft.run_file_transfer_showcase(size_gb=0.2)
        mock_ensure_sized.assert_called_once_with(ft.DEFAULT_SERVER_CONTAINER, round(0.2 * 1024 ** 3))
        # the HEAD/curl download URL must point at the generated sized asset, not bigbuckbunny.mp4
        head_call = next(c for c in mock_exec.call_args_list if c[0][1] == "curl")
        assert head_call[0][2] == "-sI"
        assert head_call[0][3] == f"http://{ft.SERVER_INTERNAL_IP}:{ft.HTTP_PORT}/sized/214748365.bin"
        assert result.size_bytes == 214748365

    @patch("backend.showcases.file_transfer.ensure_containers_healthy")
    @patch("backend.showcases.file_transfer.time.sleep")
    @patch("backend.showcases.file_transfer.requests.post")
    @patch("backend.showcases.file_transfer.subprocess.run")
    @patch("backend.showcases.file_transfer.ensure_http_server_running")
    @patch("backend.showcases.file_transfer.set_congestion_control")
    @patch("backend.showcases.file_transfer._docker_exec")
    def test_times_out_if_never_completes(self, mock_exec, mock_cc, mock_http,
                                          mock_subproc_run, mock_post, mock_sleep, mock_healthy):
        def exec_side_effect(container, *cmd, **kwargs):
            if cmd[0] == "cat":
                return MagicMock(returncode=1, stdout="")  # never produces a result
            return MagicMock(returncode=0, stdout="0\n")

        mock_exec.side_effect = exec_side_effect
        with pytest.raises(ft.ShowcaseError, match="did not finish"):
            ft.run_file_transfer_showcase(timeout_s=1.0, poll_interval_s=0.1)
