#!/usr/bin/env bash
# Cross-Eval experiment launcher (Experiment 10)
# Usage:
#   ./scripts/run_cross_eval.sh smoke          # 32 cases, qwen, ready attacks
#   ./scripts/run_cross_eval.sh ready          # 96 cells, all models, ready attacks
#   ./scripts/run_cross_eval.sh full           # 128 cells, all models (needs offline)
#   ./scripts/run_cross_eval.sh offline        # Enhanced + Adaptive poison per model
#   ./scripts/run_cross_eval.sh setup          # validate only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
RUN=( "$PYTHON" src/run_cross_eval_experiment.py )

mode="${1:-smoke}"

case "$mode" in
  setup)
    "${RUN[@]}" --phase setup
    ;;
  offline)
    "${RUN[@]}" --phase offline --model all --use_cache
    ;;
  smoke)
    "${RUN[@]}" --phase cross_eval --smoke --limit 32 \
      --model qwen3.5-27b --ready_attacks_only --use_cache
    "${RUN[@]}" --phase report --smoke
    ;;
  ready)
    "${RUN[@]}" --phase cross_eval --model all --ready_attacks_only --use_cache
    "${RUN[@]}" --phase report
    ;;
  full)
    "${RUN[@]}" --phase all --model all --use_cache
    ;;
  dry)
    "${RUN[@]}" --phase all --model all --dry_run
    ;;
  *)
    echo "Unknown mode: $mode"
    echo "Usage: $0 {setup|offline|smoke|ready|full|dry}"
    exit 1
    ;;
esac
