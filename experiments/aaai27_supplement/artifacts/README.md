# Frozen Artifacts

This directory contains small tracked artifacts needed to begin held-out
evaluation without retraining.

## `frozen_test_attack_211.jsonl`

One frozen Ours-Attack payload per held-out test case.

- Records: `211`
- Payload coverage: `211/211`
- Victim outputs and evaluation labels are stripped
- Source model: `deepseek-v4.1-flash-1017`

## `selected_harness.json`

Validation-selected defense harness and candidate ranking.

## `MANIFEST.json`

SHA-256 hashes and provenance for the tracked artifacts.

## Omitted Artifacts

Raw victim outputs, API logs, full trajectories, and offline poison caches are
not tracked. They are large, contain transient provider failures, or can be
regenerated with the documented commands.

## Safety

The attack JSONL contains research attack payloads. Use it only for authorized,
sandboxed evaluation.
