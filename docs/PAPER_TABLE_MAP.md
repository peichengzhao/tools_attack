# Paper Table Map

This document maps each paper table to code, inputs, tracked artifacts, and
runtime dependencies.

## Shared Inputs

| Input | Path |
|---|---|
| Train split | `InjecAgent/data/splits/train_cases_632.json` |
| Validation split | `InjecAgent/data/splits/validation_cases_211.json` |
| Test split | `InjecAgent/data/splits/test_cases_211.json` |
| Split manifest | `experiments/aaai27_supplement/data/split_manifest.csv` |
| Frozen attack payloads | `experiments/aaai27_supplement/artifacts/frozen_test_attack_211.jsonl` |
| Selected harness | `experiments/aaai27_supplement/artifacts/selected_harness.json` |

## Table 1: Attack Comparison

| Item | Value |
|---|---|
| Purpose | Compare attack artifacts without an added defense |
| Runner | `InjecAgent/src/evaluate_cross_eval.py` |
| Defense mode | `none` |
| Attacks | Naive, InjecAgent-Default, InjecAgent-Enhanced, TopicAttack, ChatInject, Adaptive Attack, Combined, Important Instructions, Tool Knowledge, Ours-Attack |
| Test cases | 211 |
| Tracked output | No result files are distributed; regenerate with a model endpoint |

```bash
cd InjecAgent
python src/evaluate_cross_eval.py \
  --model deepseek-v4-flash \
  --defense none \
  --attack all \
  --test_data data/splits/test_cases_211.json \
  --ours_attacks ../experiments/aaai27_supplement/artifacts/frozen_test_attack_211.jsonl \
  --setting base \
  --prompt_type InjecAgent
```

## Table 2: Attack × Defense × Model

| Item | Value |
|---|---|
| Purpose | Cross-evaluate attack rows against static and evolved defenses |
| Runner | `InjecAgent/src/evaluate_cross_eval.py` |
| Defenses | Sandwich, Spotlighting, StruQ-Prompt, SecAlign-Prompt, Ours |
| Attacks | Nine Table 2 rows including Ours-Attack |
| Grid | `5 × 5 × 9 = 225` cells in the checked manifest |
| Runtime dependency | OpenAI-compatible victim model; Adaptive artifacts for the Adaptive row |

Static defense example:

```bash
cd InjecAgent
python src/evaluate_cross_eval.py \
  --model deepseek-v4-flash \
  --defense sandwich \
  --attack all \
  --test_data data/splits/test_cases_211.json \
  --ours_attacks ../experiments/aaai27_supplement/artifacts/frozen_test_attack_211.jsonl \
  --setting base \
  --prompt_type InjecAgent
```

Ours defense:

```bash
python src/evaluate_cross_eval.py \
  --model deepseek-v4-flash \
  --defense ours \
  --attack all \
  --test_data data/splits/test_cases_211.json \
  --harness_file ../experiments/aaai27_supplement/artifacts/selected_harness.json \
  --ours_attacks ../experiments/aaai27_supplement/artifacts/frozen_test_attack_211.jsonl \
  --setting base \
  --prompt_type InjecAgent
```

## Table 3(a): Poison-Free Clean Utility

| Suite | Runner | Dependency |
|---|---|---|
| AgentDojo | `exp7/runners/agentdojo.py` | AgentDojo environment |
| InjecAgent user cases | `InjecAgent/src/evaluate_injecagent_utility.py` | Model endpoint |
| BFCL | `exp7/runners/bfcl.py` | BFCL CLI |
| AgentBench | `exp7/runners/agentbench.py` | Python 3.9 + Docker workers |
| AppWorld | `exp7/runners/appworld.py` | Python 3.11 + AppWorld data |

```bash
cd experiments/aaai27_supplement
python scripts/run_clean_utility.py \
  --selected-harness artifacts/selected_harness.json \
  --models gpt-5.5,deepseek-v4-flash,glm-5.1 \
  --benchmarks agentdojo,injecagent,bfcl,agentbench,appworld
```

## Table 3(b): Artifact Combinations

| Combination | Meaning |
|---|---|
| OAtk + OHar | Original test seed poison + original harness |
| EAtk + OHar | Frozen evolved poison + original harness |
| OAtk + EHar | Original test seed poison + selected harness |
| EAtk + EHar | Frozen evolved poison + selected harness |

```bash
cd experiments/aaai27_supplement
python scripts/evaluate_cross_combinations.py \
  --test-data ../../InjecAgent/data/splits/test_cases_211.json \
  --evolved-attacks artifacts/frozen_test_attack_211.jsonl \
  --selected-harness artifacts/selected_harness.json \
  --output-dir ../../results/aaai27_supplement/table3_cross
```

## Interactive Defense Baselines

| Defense | Runner | Paper relation |
|---|---|---|
| Task Shield | `scripts/run_task_shield_agentdojo.py` | Third-party reproduction; separate table |
| MELON | `scripts/run_melon_agentdojo.py` | Official detector; separate table |
| CaMeL | Upstream `main.py` through `uv` | Official research artifact; separate table |

```bash
cd experiments/aaai27_supplement
python scripts/run_interactive_baselines.py
python scripts/build_interactive_table.py
```

## External Baseline: PIArena

PIArena is a platform. This repository fixes `pisanitizer` as its representative
defense for comparison. Rationale is documented in
`experiments/aaai27_supplement/PIARENA_DEFENSE_CHOICE.md`.
