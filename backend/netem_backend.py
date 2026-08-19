"""tc/netem backends for applying link parameters to a running emulation.

Per user decision: Docker is the primary emulation target (two containers on
a bridge network, `NET_ADMIN` capability, driven via `docker exec ... tc`)
because it needs less host privilege than bare root network namespaces and
is what's actually available to set up portably for a conference laptop.
`DryRunNetemBackend` implements the identical interface but only logs the
commands it would run -- this is the backend exercised in this development
sandbox, which has neither `docker` nor `tc`/`ip` installed (verified: both
commands are absent). Real execution must be validated on the conference
laptop; see docs/VALIDATION.md and README.md for what has/hasn't been
exercised where.

Note this replaces `models/ObLoS/emulator` (the Rust/rtnetlink binary): we
confirmed (see plan research) that the emulator only ever configures
standard netem fields (loss/delay/jitter/rate/pareto-distribution), all of
which are reachable from the plain `tc` CLI -- nothing here needs rtnetlink
specifically, and a Python/Docker-based orchestrator matches what the paper
itself describes ("a Python orchestrator script").
"""
from __future__ import annotations

import abc
import shlex
import subprocess
from dataclasses import dataclass


@dataclass
class NetemParams:
    loss_pct: float  # 0-100
    delay_ms: float
    jitter_ms: float
    rate_mbit: float

    def as_tc_netem_args(self) -> list[str]:
        args = ["netem"]
        if self.loss_pct > 0:
            args += ["loss", f"{self.loss_pct:.4f}%"]
        args += ["delay", f"{max(self.delay_ms, 0):.3f}ms"]
        if self.jitter_ms > 0:
            args += [f"{self.jitter_ms:.3f}ms"]
        if self.rate_mbit > 0:
            args += ["rate", f"{self.rate_mbit:.3f}mbit"]
        return args


class NetemBackend(abc.ABC):
    """Applies NetemParams to a named endpoint ("client" or "server")."""

    @abc.abstractmethod
    def apply(self, endpoint: str, params: NetemParams) -> None: ...

    @abc.abstractmethod
    def reset(self, endpoint: str) -> None: ...


class DockerNetemBackend(NetemBackend):
    def __init__(self, containers: dict, interface: str = "eth0"):
        """`containers` maps endpoint name -> docker container name, e.g.
        {"client": "cosme-client", "server": "cosme-server"} (see docker/docker-compose.yml).
        """
        self.containers = containers
        self.interface = interface
        self.log: list[str] = []  # mirrors DryRunNetemBackend.log so callers can treat both uniformly

    # A real `tc qdisc replace/del` via `docker exec` completes in tens of ms; 10s is a generous
    # bound that only trips on a genuine hang (docker daemon wedged, etc.), not normal variance.
    # Previously had NO timeout at all on the single most frequently-invoked subprocess call in
    # the whole system (up to 10Hz during playback) -- a hang here would have blocked that
    # OS thread forever with nothing to show for it beyond a stuck-looking dashboard.
    _CMD_TIMEOUT_S = 10.0

    def _run(self, container: str, tc_args: list[str]) -> None:
        cmd = ["docker", "exec", container, "tc", "qdisc"]
        cmd += tc_args
        self.log.append(shlex.join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self._CMD_TIMEOUT_S)

    def apply(self, endpoint: str, params: NetemParams) -> None:
        # Always "replace", never "add" -- `add` fails ("RTNETLINK answers: File exists") if a
        # qdisc from an EARLIER scenario run is still on the container, which is the common case:
        # containers are long-lived across many scenario runs, and nothing tears down the qdisc
        # on a scenario finishing naturally (only an explicit Stop does, see Scenario.stop()).
        # A per-instance "already added one" flag cannot know about qdisc state left by a
        # DIFFERENT Scenario's backend instance, and the resulting CalledProcessError kills
        # playback from inside an un-awaited task. `replace` creates-or-replaces either way.
        container = self.containers[endpoint]
        self._run(container, ["replace", "dev", self.interface, "root"] + params.as_tc_netem_args())

    def reset(self, endpoint: str) -> None:
        container = self.containers[endpoint]
        subprocess.run(
            ["docker", "exec", container, "tc", "qdisc", "del", "dev", self.interface, "root"],
            capture_output=True, text=True, timeout=self._CMD_TIMEOUT_S,
        )


class DryRunNetemBackend(NetemBackend):
    """Logs the tc/docker commands it would issue instead of executing them.

    This is the backend actually exercised in this development sandbox
    (confirmed: no docker, no tc/ip binaries present here).
    """

    def __init__(self, containers: dict | None = None, interface: str = "eth0", verbose: bool = False):
        self.containers = containers or {"client": "cosme-client", "server": "cosme-server"}
        self.interface = interface
        self.verbose = verbose
        self.log: list[str] = []

    def _record(self, endpoint: str, tc_args: list[str]) -> None:
        container = self.containers.get(endpoint, endpoint)
        cmd = ["docker", "exec", container, "tc", "qdisc"] + tc_args
        line = shlex.join(cmd)
        self.log.append(line)
        if self.verbose:
            print(f"[dry-run] {line}")

    def apply(self, endpoint: str, params: NetemParams) -> None:
        self._record(endpoint, ["replace", "dev", self.interface, "root"] + params.as_tc_netem_args())

    def reset(self, endpoint: str) -> None:
        self._record(endpoint, ["del", "dev", self.interface, "root"])
