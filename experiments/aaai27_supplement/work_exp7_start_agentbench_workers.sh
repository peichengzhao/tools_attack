#!/usr/bin/env bash
# Start the minimal AgentBench workers used by Exp7 (dbbench-std + os-std).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTBENCH="$ROOT/related_work/AgentBench"
PYTHON="${AGENTBENCH_PYTHON:-$ROOT/.venvs/agentbench/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing AgentBench Python: $PYTHON" >&2
  echo "Run: bash exp7/setup_benchmarks.sh agentbench" >&2
  exit 2
fi
if [[ ! -S /var/run/docker.sock && -z "${DOCKER_HOST:-}" ]]; then
  echo "[blocked] No Docker API endpoint. Ask the server administrator to enable Docker." >&2
  exit 2
fi

cd "$AGENTBENCH"
exec "$PYTHON" -m src.start_task -a \
  --config configs/start_task_lite.yaml "$@"
