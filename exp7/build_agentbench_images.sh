#!/usr/bin/env bash
# Build and pull the Docker images needed by AgentBench dbbench/os workers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTBENCH="$ROOT/related_work/AgentBench"

if [[ ! -S /var/run/docker.sock && -z "${DOCKER_HOST:-}" ]]; then
  echo "[blocked] No Docker API endpoint. Ask the server administrator to enable Docker." >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[blocked] docker CLI is not installed." >&2
  exit 2
fi

cd "$AGENTBENCH"
docker pull mysql:8
docker pull ubuntu:22.04
docker build -f data/os_interaction/res/dockerfiles/default \
  data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages \
  data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu \
  data/os_interaction/res/dockerfiles --tag local-os/ubuntu
echo "[ok] AgentBench dbbench/os images are ready"
