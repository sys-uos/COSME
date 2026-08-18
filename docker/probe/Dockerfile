# Probe container for the remote-desktop (VNC) showcase.
#
# Exists because vncdotool is not packaged in Debian (pip-only) and the
# minimal endpoint image deliberately has no pip. Shares cosme-client's
# network namespace (see docker-compose.yml) so its RFB traffic crosses the
# tc/netem-shaped interface exactly like any other client traffic.
FROM python:3.12-slim

# iproute2: not present in python:3.12-slim by default -- needed purely for the `ip route get`
# healthcheck in docker-compose.yml (see its own comment for why), which detects the real,
# documented failure mode where this container's shared network namespace with cosme-client goes
# stale (routing breaks with "Network is unreachable") after a cosme-client recreate/stop+start.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir vncdotool
COPY vnc_probe.py /app/vnc_probe.py

# Idles; backend/showcases/remote_desktop.py triggers probe runs on demand
# via `docker exec cosme-probe python3 /app/vnc_probe.py ...`.
CMD ["sleep", "infinity"]
