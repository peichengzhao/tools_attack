#!/usr/bin/env bash
# Idempotent setup for AgentBench and AppWorld.
#
# This script installs Python dependencies and benchmark data. AgentBench task
# workers additionally require a working Docker daemon; it never tries to use
# sudo. If Docker is unavailable, setup stops with an explicit status.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
RW="$REPO_ROOT/related_work"
VENVS="$REPO_ROOT/.venvs"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
TARGET="${1:-check}"

mkdir -p "$RW" "$VENVS"

ensure_repo() {
  local url="$1" dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "[ok] $dir"
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

ensure_conda_env() {
  local prefix="$1" python_version="$2"
  if [[ -x "$prefix/bin/python" ]]; then
    echo "[ok] Python environment $prefix"
  else
    "$CONDA_BIN" create -y -p "$prefix" "python=$python_version" pip
  fi
}

setup_agentbench() {
  ensure_repo https://github.com/THUDM/AgentBench.git "$RW/AgentBench"
  ensure_conda_env "$VENVS/agentbench" "3.9"
  "$VENVS/agentbench/bin/python" -m pip install \
    torch --index-url https://download.pytorch.org/whl/cpu
  "$VENVS/agentbench/bin/python" -m pip install -r "$RW/AgentBench/requirements.txt"
  "$VENVS/agentbench/bin/python" -m pip install importlib-metadata
  echo "[ok] AgentBench Python dependencies installed"
}

setup_appworld() {
  ensure_repo https://github.com/StonyBrookNLP/appworld.git "$RW/appworld"
  ensure_conda_env "$VENVS/appworld" "3.11"
  "$VENVS/appworld/bin/python" -m pip install 'appworld>=0.1.3'
  APPWORLD_ROOT="$RW/appworld" "$VENVS/appworld/bin/appworld" install
  "$VENVS/appworld/bin/appworld" download data --root "$RW/appworld"
  APPWORLD_ROOT="$RW/appworld" "$VENVS/appworld/bin/appworld" verify tests --root "$RW/appworld"
}

check_agentbench_docker() {
  if [[ -S /var/run/docker.sock || -n "${DOCKER_HOST:-}" ]]; then
    echo "[ok] Docker API endpoint detected"
  else
    echo "[blocked] AgentBench workers need /var/run/docker.sock or DOCKER_HOST." >&2
    echo "          A system administrator must enable Docker for user $(id -un)." >&2
    return 1
  fi
}

case "$TARGET" in
  agentbench)
    setup_agentbench
    check_agentbench_docker
    ;;
  appworld)
    setup_appworld
    ;;
  all)
    setup_appworld
    setup_agentbench
    check_agentbench_docker
    ;;
  check)
    "$REPO_ROOT/experiments/aaai27_supplement/scripts/check_external_benchmarks.py"
    ;;
  *)
    echo "Usage: bash $0 [check|agentbench|appworld|all]" >&2
    exit 1
    ;;
esac
