# Reproducibility Status

Generated: `2026-11-25T09:10:48.227900+00:00`

## Core checks

| Check | Status |
|---|---|
| Split manifest hash | PASS |
| Split counts | PASS |
| Frozen artifacts | PASS |
| Table 2 cross-eval contract | PASS |
| Baseline provenance | PASS |

## Paper tables

| Table | Code | Artifact/data state | Current status |
|---|---|---|---|
| Table 1 | READY | frozen attack 211/211; Adaptive caches incomplete | partial |
| Table 2 | READY | 225 cells; Adaptive caches incomplete | partial |
| Table 3 | READY | AppWorld ready; AgentBench Docker blocked | partial |
| Interactive baselines | READY | 0/12 cells | pending |

## Adaptive Attack coverage

| Model | Cached | Expected |
|---|---:|---:|
| deepseek-v3.2 | 0 | 211 |
| deepseek-v4-flash | 28 | 211 |
| glm-5.1 | 0 | 211 |
| glm-5.1-fp8 | 0 | 211 |
| qwen3.5-27b | 28 | 211 |

## Interpretation

- `PASS` means the code and tracked data satisfy the local reproducibility contract.
- `partial` means some paper rows require regenerating per-victim Adaptive artifacts.
- `blocked` means an external environment (usually Docker for AgentBench) is unavailable.
- No model API is called by this audit.

