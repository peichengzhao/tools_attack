#!/usr/bin/env bash
# Install optional Exp7 benchmark dependencies under related_work/ and .venvs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RW="$ROOT/related_work"
VENVS="$ROOT/.venvs"
TARGET="${1:-check}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"

mkdir -p "$RW" "$VENVS"

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "[ok] already cloned: $dir"
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

ensure_env() {
  local prefix="$1"
  local python_version="$2"
  if [[ -x "$prefix/bin/python" ]]; then
    echo "[ok] python environment: $prefix"
  else
    "$CONDA_BIN" create -y -p "$prefix" "python=$python_version" pip
  fi
}

setup_agentbench() {
  clone_if_missing "https://github.com/THUDM/AgentBench.git" "$RW/AgentBench"
  ensure_env "$VENVS/agentbench" "3.9"
  if ! (
    cd "$RW/AgentBench"
    "$VENVS/agentbench/bin/python" -c "import docker, src.client.agent_test"
  ) 2>/dev/null; then
    "$VENVS/agentbench/bin/python" -m pip install \
      torch --index-url https://download.pytorch.org/whl/cpu
    "$VENVS/agentbench/bin/python" -m pip install -r "$RW/AgentBench/requirements.txt"
    "$VENVS/agentbench/bin/python" -m pip install importlib-metadata
  fi
  echo "AgentBench Python environment ready: $VENVS/agentbench/bin/python"
  if [[ ! -S /var/run/docker.sock && -z "${DOCKER_HOST:-}" ]]; then
    echo "[blocked] Docker API endpoint is missing." >&2
    echo "          AgentBench dbbench-std/os-std workers cannot start." >&2
    return 2
  fi
}

setup_appworld() {
  clone_if_missing "https://github.com/StonyBrookNLP/appworld.git" "$RW/appworld"
  local py="${APPWORLD_PYTHON:-$HOME/anaconda3/envs/appworld/bin/python}"
  if [[ ! -x "$py" ]]; then
    ensure_env "$VENVS/appworld" "3.11"
    py="$VENVS/appworld/bin/python"
  fi
  if ! "$py" -c "import appworld" 2>/dev/null; then
    "$py" -m pip install 'appworld>=0.1.3'
  fi
  local cli
  cli="$(dirname "$py")/appworld"
  APPWORLD_ROOT="$RW/appworld" "$cli" install
  if [[ ! -f "$RW/appworld/data/datasets/test_normal.txt" ]]; then
    APPWORLD_ROOT="$RW/appworld" "$cli" download data --root "$RW/appworld"
  fi
  echo "AppWorld ready: python=$py root=$RW/appworld"
}

setup_bfcl() {
  local py="${PROJECT_PYTHON:-$HOME/anaconda3/envs/tools_attack/bin/python}"
  "$py" -m pip install 'bfcl-eval==2025.12.17' soundfile
  echo "BFCL ready. Exp7 registers tools_attack models via exp7/bfcl/model_overrides.json"
}

check_all() {
  "$ROOT/experiments/aaai27_supplement/scripts/check_external_benchmarks.py"
}

case "$TARGET" in
  check) check_all ;;
  agentbench) setup_agentbench ;;
  appworld) setup_appworld ;;
  bfcl) setup_bfcl ;;
  all)
    setup_bfcl
    setup_appworld
    setup_agentbench
    check_all
    ;;
  *)
    echo "Usage: bash exp7/setup_benchmarks.sh [check|all|bfcl|agentbench|appworld]"
    exit 1
    ;;
esac
