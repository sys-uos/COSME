"""Shared pytest configuration: the `docker` / `probe` markers and the
research-data auto-skip.

Division of labor (documented in README.md):
  - `pytest -m "not docker"` -- the pure-unit suite (mocked/dry-run), runs
    anywhere, no Docker needed.
  - `pytest -m docker` -- repeatable regression tests against an
    ALREADY-RUNNING stack (cosme-client/cosme-server up, backend venv).
    Auto-skipped when the containers aren't detected.
  - `scripts/verify_docker_stack.sh` -- the conference-laptop *bring-up*
    checklist (compose up --build, host sudoers/modprobe checks, OSRM data
    presence, probe container); not a pytest concern.

Tests that need the real measurement data under `models/` are skipped when that
directory is absent, which is how the repository arrives from a clone (`models/`
is gitignored -- see README.md "Research data"). Without this a fresh clone
reports dozens of FileNotFoundError failures that look like a broken checkout
rather than the documented, expected state.
"""
import subprocess
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _containers_running(*names: str) -> bool:
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0 and set(names).issubset(set(out.stdout.split()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "docker: needs the real cosme-client/cosme-server containers running")
    config.addinivalue_line(
        "markers", "probe: additionally needs the cosme-probe (VNC probe) container running")
    config.addinivalue_line(
        "markers", "research_data: needs the real measurement data under models/")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Skip, rather than fail, when a test needs absent research data.

    Narrow by design: fires only when `models/` is absent AND the error is a FileNotFoundError
    pointing inside it, so a genuine missing-file bug still fails.
    """
    outcome = yield
    if MODELS_DIR.exists():
        return
    call = outcome.excinfo
    if call is None or not issubclass(call[0], FileNotFoundError):
        return
    missing = getattr(call[1], "filename", None) or ""
    if str(MODELS_DIR) not in str(missing):
        return
    outcome.force_exception(
        pytest.skip.Exception(
            f"needs the research data under models/ (missing: {missing}) -- "
            "see README.md \"Research data\"",
            _use_item_location=True,
        )
    )


def pytest_collection_modifyitems(config, items):
    have_docker = _containers_running("cosme-client", "cosme-server")
    have_probe = have_docker and _containers_running("cosme-probe")
    have_data = MODELS_DIR.exists()
    skip_docker = pytest.mark.skip(reason="cosme-client/cosme-server containers not running")
    skip_probe = pytest.mark.skip(reason="cosme-probe container not running")
    # These assert on non-empty drive lists rather than opening a file, so they surface a
    # missing models/ as a bare assertion failure that pytest_runtest_call cannot recognise.
    skip_data = pytest.mark.skip(reason="needs the research data under models/ -- see README.md")
    for item in items:
        if "docker" in item.keywords and not have_docker:
            item.add_marker(skip_docker)
        if "probe" in item.keywords and not have_probe:
            item.add_marker(skip_probe)
        if "research_data" in item.keywords and not have_data:
            item.add_marker(skip_data)
