"""Integration tests against the REAL Docker stack (auto-skipped without it).

These exercise the paths the unit suite deliberately mocks: real `docker
exec ... tc` shaping, the nsenter-based congestion-control write, and real
application traffic. They assume the base stack is already up
(`cd docker && docker compose up -d`); see conftest.py for the marker
mechanics and scripts/verify_docker_stack.sh for the full bring-up checklist.

Kept intentionally quick (~1 min total, dominated by the two short showcase
runs) so they're cheap to run before every demo session.
"""
import os
import subprocess
import time

import pytest

from backend.netem_backend import DockerNetemBackend, NetemParams
from backend.showcases import file_transfer, surveillance

pytestmark = pytest.mark.docker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _tc_show(container: str) -> str:
    return subprocess.run(
        ["docker", "exec", container, "tc", "qdisc", "show", "dev", "eth0"],
        capture_output=True, text=True, timeout=10,
    ).stdout


@pytest.fixture()
def clean_qdiscs():
    """Guarantees no leftover shaping before and after a test."""
    for c in ("cosme-client", "cosme-server"):
        subprocess.run(["docker", "exec", c, "tc", "qdisc", "del", "dev", "eth0", "root"],
                       capture_output=True, timeout=10)
    yield
    for c in ("cosme-client", "cosme-server"):
        subprocess.run(["docker", "exec", c, "tc", "qdisc", "del", "dev", "eth0", "root"],
                       capture_output=True, timeout=10)


class TestDockerNetemBackend:
    def test_apply_and_reset_round_trip(self, clean_qdiscs):
        backend = DockerNetemBackend(containers={"client": "cosme-client", "server": "cosme-server"})
        backend.apply("client", NetemParams(loss_pct=0, delay_ms=100, jitter_ms=10, rate_mbit=50))
        out = _tc_show("cosme-client")
        assert "netem" in out and "delay 100ms" in out and "rate 50Mbit" in out
        backend.reset("client")
        assert "netem" not in _tc_show("cosme-client")

    def test_shaping_actually_changes_rtt(self, clean_qdiscs):
        backend = DockerNetemBackend(containers={"client": "cosme-client", "server": "cosme-server"})
        backend.apply("server", NetemParams(loss_pct=0, delay_ms=150, jitter_ms=0, rate_mbit=0))
        ping = subprocess.run(
            ["docker", "exec", "cosme-client", "ping", "-c", "2", "-W", "3", "10.42.0.10"],
            capture_output=True, text=True, timeout=15,
        )
        rtt_avg = float(ping.stdout.strip().splitlines()[-1].split("/")[4])
        assert rtt_avg >= 140, f"150ms netem delay not reflected in ping: {rtt_avg}ms"


class TestScenarioAgainstRealDocker:
    def test_scenario_runs_in_docker_mode(self, clean_qdiscs):
        # Import here so the TestClient app initializes fresh with Docker present.
        from backend.tests.test_api import client
        resp = client.post("/api/scenarios", json={
            "duration_s": 8, "speed": 8, "seed": 3,
            "custom_obstruction_trace": [{"timestamp": 2.0, "lossTime": 0.5}],
        })
        assert resp.status_code == 200
        sid = resp.json()["id"]
        deadline = time.monotonic() + 25
        status = client.get(f"/api/scenarios/{sid}/status").json()
        while status["running"] and time.monotonic() < deadline:
            time.sleep(0.5)
            status = client.get(f"/api/scenarios/{sid}/status").json()
        assert status["backend_mode"] == "docker"
        assert status["n_tc_commands_issued"] > 0
        assert any("docker exec" in c for c in status["recent_tc_commands"])


class TestCongestionControl:
    def test_set_and_read_back_via_nsenter(self):
        try:
            file_transfer.set_congestion_control("reno", "cosme-server")
        except file_transfer.ShowcaseError as e:
            pytest.skip(f"nsenter sudoers rule not installed on this host: {e}")
        out = subprocess.run(
            ["docker", "exec", "cosme-server", "sysctl", "-n", "net.ipv4.tcp_congestion_control"],
            capture_output=True, text=True, timeout=10,
        )
        assert out.stdout.strip() == "reno"
        file_transfer.set_congestion_control("cubic", "cosme-server")


class TestRealShowcases:
    def test_file_transfer_real_http(self, clean_qdiscs):
        result = file_transfer.run_file_transfer_showcase(congestion_control="cubic")
        assert result.http_code == 200
        assert result.throughput_bps > 0
        expected = os.path.getsize(os.path.join(REPO_ROOT, "docker", "media", "bigbuckbunny.mp4"))
        assert result.size_bytes == expected

    def test_surveillance_receives_frames(self, clean_qdiscs):
        result = surveillance.run_surveillance_showcase(duration_s=10)
        assert result.frames_received > 0
        assert result.mean_bitrate_kbps > 0


@pytest.mark.probe
class TestProbeContainerShowcases:
    """Showcases whose client probe lives in the dedicated cosme-probe container."""

    def test_remote_desktop_measures_keystrokes(self, clean_qdiscs):
        from backend.showcases import remote_desktop
        result = remote_desktop.run_remote_desktop_showcase(duration_s=12, congestion_control="cubic")
        assert result.n_keystrokes > 0
        assert result.keystroke_latency_ms_median is not None
        assert result.effective_fps_mean is not None


class TestWebrtcShowcase:
    def test_voip_call_flows_audio(self, clean_qdiscs):
        from backend.showcases import video_conferencing
        result = video_conferencing.run_voip_showcase(duration_s=10)
        assert result.audio_frames_received > 0
        assert result.video_frames_received == 0  # genuinely audio-only
