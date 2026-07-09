# AAAI-27 Supplement Experiments

Self-contained supplement code for the ICLT/AAAI-27 indirect prompt-injection
co-evolution experiments.

This directory adds:

1. The restored `AAAI27-S1-v1` 632/211/211 split manifest.
2. Fixed-denominator metrics.
3. Co-evolution training and ablation runners.
4. Frozen attack-rule construction (`G*`).
5. Validation-only harness selection.
6. Held-out test attack and defense evaluation.
7. Call budgeting, quota detection, and resumable output.
8. Pinned source snapshots and unified launchers for MELON, CaMeL, PIArena,
   and Task Shield.
9. AgentBench/AppWorld installation, data, and environment checks.

## Important Scope

The original repository did not contain the paper's exact frozen `G*`
implementation or harness-selection artifact. This package provides a complete
reproducible reimplementation:

- `G*` is synthesized from 632 training trajectories.
- Harnesses are extracted from the training run and selected only on validation.
- Test cases are used only after `G*` and the harness are frozen.

Results produced by this package should be described as a reproducible
reimplementation, not as the paper's original artifact-level numbers unless the
original implementation is also recovered.

## Environment

Required:

- project Python: normally
  `/home/zpc/anaconda3/envs/tools_attack/bin/python`
- model config: normally
  `/home/zpc/contract_heldout_ipi/model_api_ablation.py`
- `deepseek-v4.1-flash-1017` access

Keep API keys outside Git. The example config is
`configs/model_api_ablation.example.py`.

```bash
export AAAI27_PROJECT_PYTHON=/home/zpc/anaconda3/envs/tools_attack/bin/python
export AAAI27_MODEL_CONFIG=/home/zpc/contract_heldout_ipi/model_api_ablation.py
```

## Directory Layout

```text
experiments/aaai27_supplement/
  data/
    split_manifest.csv
    train_case_ids.txt
    validation_case_ids.txt
    test_case_ids.txt
  configs/
    experiments.json
    model_api_ablation.example.py
  aaai27_supplement/
    metrics.py
    estimate.py
  scripts/
    run_pipeline.py
    check_environment.py
    verify_artifacts.py
    materialize_split.py
    run_ablation.py
    build_frozen_attack_rule.py
    select_harness_candidates.py
    evaluate_frozen_attack.py
    evaluate_harness.py
    run_clean_utility.py
    evaluate_cross_combinations.py
    analyze_results.py
    filter_results_by_split.py
  tests/
```

## Quick Checks

```bash
cd experiments/aaai27_supplement

python -m unittest discover -s tests -v

python scripts/check_environment.py \
  --project-python "$AAAI27_PROJECT_PYTHON" \
  --model-config "$AAAI27_MODEL_CONFIG"

# Print the full command sequence and estimated call volume.
python scripts/run_pipeline.py --stage all --dry-run --budget-calls 20000
```

## Full Pipeline

The recommended runner is `run_pipeline.py`:

```bash
python scripts/run_pipeline.py \
  --stage all \
  --budget-calls 20000 \
  --workers 8 \
  --max-rounds 8
```

Stages:

| Stage | Purpose |
|---|---|
| `prepare-split` | materialize the bundled 632/211/211 manifest |
| `train` | full co-evolution on 632 training cases |
| `build-gstar` | synthesize frozen attack rule `G*` |
| `validate-attack` | apply `G*` to validation 211 |
| `select-harness-candidates` | extract diverse harness candidates from training |
| `validate-harness` | evaluate candidates on validation and freeze one |
| `test-attack` | apply frozen `G*` to test 211 |
| `prepare-selected-harness` | create one-candidate selection file |
| `test-harness` | evaluate frozen selected harness on test |
| `table3-clean` | run five poison-free clean suites with the selected harness |
| `table3-cross` | evaluate OAtk/EAtk × OHar/EHar on test |

The runner stops immediately if logs contain quota, 429/529, or account-ban
markers.

## Budget Safety

`estimate.py` provides upper-bound logical model calls.

Current defaults:

- 632 training cases × 8 rounds ≈ 9,480 calls
- validation G*: 422 calls
- 8 harness candidates × 211 cases: 1,688 calls
- test G*: 422 calls
- selected harness test: 211 calls

Do not use high worker counts without a hard API budget. The previous
16-worker run exhausted quota because retries amplified request volume.

## Table 1

Recompute fixed-denominator results from historical per-case logs:

```bash
python scripts/filter_results_by_split.py \
  --root ../../InjecAgent/results \
  --manifest data/split_manifest.csv \
  --split test \
  --output-dir ../../results/aaai27_supplement/split/test

python scripts/analyze_results.py \
  --root ../../results/aaai27_supplement/split/test \
  --output-dir ../../results/aaai27_supplement/split/test_summary
```

Outputs:

- `per_case.csv`
- `summary.csv`
- `summary.md`
- `manifest.json`

Primary metric:

```text
ASR-all = attack successes / all cases
```

Invalid cases are never removed from the primary denominator.

## Table 2

Table 2 uses:

- `TopicAttack`: existing baseline script
- `Adaptive Attack`: same 8-round trajectory-feedback budget
- `Ours-Attack`: frozen `G*` from the 632 training run

## Table 3

`run_ablation.py` supports:

- `fixed`
- `attack_only`
- `defense_only`
- `full`
- `independent`
- `simultaneous`
- `no_feedback`
- `random_routing`
- `round_robin`

Use:

```bash
cd ../../InjecAgent
python ../experiments/aaai27_supplement/scripts/run_ablation.py \
  --mode full \
  --data data/splits/train_cases_632.json \
  --workers 4 \
  --max_rounds 8 \
  --output ../results/aaai27_supplement/artifacts/full_train632.jsonl
```

### Table 3(a): clean utility suites

The clean-suite wrapper supports any validation-selected harness, not only the
static `max_defense`:

```bash
python scripts/run_clean_utility.py \
  --selected-harness artifacts/selected_harness.json \
  --models gpt-5.5,deepseek-v4-flash,glm-5.1 \
  --benchmarks agentdojo,injecagent,bfcl,agentbench,appworld
```

This patch propagates `TOOLS_ATTACK_HARNESS_STRATEGY_FILE` into:

- AgentDojo
- InjecAgent clean utility
- BFCL
- AgentBench
- AppWorld

### Table 3(b): artifact cross-combinations

```bash
python scripts/evaluate_cross_combinations.py \
  --test-data ../../InjecAgent/data/splits/test_cases_211.json \
  --evolved-attacks artifacts/frozen_test_attack_211.jsonl \
  --selected-harness artifacts/selected_harness.json \
  --output-dir ../../results/aaai27_supplement/table3_cross
```

It evaluates:

- OAtk + OHar
- EAtk + OHar
- OAtk + EHar
- EAtk + EHar

## Defense Baselines

The four external baselines are vendored under `related_work/baselines/` with
exact commit markers. Verify them first:

```bash
python scripts/check_baselines.py
```

| Baseline | Provenance | Native benchmark | Entry point |
|---|---|---|---|
| MELON | Official ICML'25 implementation | AgentDojo | `scripts/run_melon_agentdojo.py` |
| CaMeL | Official Google Research artifact | AgentDojo | Upstream `main.py` through `uv` |
| PIArena | Official ACL'26 implementation | InjecAgent / AgentDojo | `scripts/run_piarena_injecagent.py` |
| Task Shield | Third-party reproduction | AgentDojo | `scripts/run_task_shield_agentdojo.py` |

PIArena is a platform containing several defenses rather than a single
defense. The representative defense is fixed to **PISanitizer**, following the
defense used in PIArena's official example experiment configuration. It is a
prevention/sanitization method and has a more balanced attack-success and
utility tradeoff than an always-blocking detection defense. This choice is
recorded in `configs/baselines.json`; the full rationale is in
`PIARENA_DEFENSE_CHOICE.md`.

Inspect a command without making API calls:

```bash
python scripts/run_baseline.py \
  --baseline melon \
  --model deepseek-v4.1-flash-1017 \
  --suite workspace \
  --attack important_instructions \
  --limit 1 \
  --dry-run
```

Run the same baseline after restoring API access:

```bash
python scripts/run_baseline.py \
  --baseline melon \
  --model deepseek-v4.1-flash-1017 \
  --suite workspace \
  --attack important_instructions \
  --limit 1
```

Important limitations:

- MELON requires an OpenAI-compatible `text-embedding-3-large` embeddings
  endpoint in addition to chat completions.
- CaMeL is explicitly labeled by its authors as an experimental research
  artifact.
- PIArena's native benchmark uses its own defense registry. The adapter maps
  the paper's 211-case DS split into PIArena's InjecAgent layout. The selected
  PISanitizer defense requires local Hugging Face weights and GPU memory.

## Static Prompt-Modified Defenses

`StruQ-Prompt` and `SecAlign-Prompt` are prompt-level adaptations inspired by
the corresponding papers. They are **not** the official trained StruQ or
SecAlign/Meta-SecAlign models. The implementation detail is documented in:

```text
DEFENSE_IMPLEMENTATION_NOTES.md
```

Generated tables use the explicit `-Prompt` suffix. Internal IDs remain
`struq` and `secalign` so existing result files stay compatible.

The complete paper-method coverage audit is in:

```text
PAPER_METHOD_COVERAGE.md
```

Cross-evaluation now supports:

- `naive` and `none` for no-defense Table 1 rows;
- `ours` as the evolved harness defense column;
- `ours_attack` from a frozen `G*` artifact;
- `--test_data data/splits/test_cases_211.json`;
- strict errors when an Adaptive poison cache is missing.
- InjecAgent-Enhanced is generated deterministically from the official hacking
  prompt prefix; only Adaptive Attack requires a frozen optimization cache.

## Interactive Defense Baselines

Task Shield, MELON, and CaMeL modify the agent execution loop instead of
filtering a single tool observation. They are therefore evaluated in a
separate AgentDojo table and are not mixed into the single-pass Table 2:

```bash
python scripts/run_interactive_baselines.py \
  --python "$AAAI27_PROJECT_PYTHON" \
  --model-config "$AAAI27_MODEL_CONFIG"

python scripts/build_interactive_table.py
```

The plan contains:

- AgentDojo `v1.2`
- attack `important_instructions`
- suites `workspace`, `banking`, `travel`, `slack`
- defenses Task Shield, MELON, CaMeL

Execution requires explicit `--execute`. The initial generated table contains
`pending` rows and no fabricated numbers.

## AgentBench and AppWorld

AppWorld uses a Python 3.11 environment and its official 193 MB dataset.
AgentBench uses an isolated Python 3.9 environment. Its task workers require a
working Docker daemon.

Check the installed state:

```bash
python scripts/check_external_benchmarks.py
```

Install or repair them:

```bash
bash scripts/setup_external_benchmarks.sh appworld
bash scripts/setup_external_benchmarks.sh agentbench
```

On the current g2 server:

- AppWorld package, 193 MB data, 1,553 unit tests, and the 147-task
  end-to-end verification all pass.
- AgentBench dependencies import successfully.
- AgentBench workers remain blocked until the server administrator exposes a
  Docker API endpoint. Rootless Podman cannot re-exec a user namespace in this
  nested server environment (`unshare -Ur` is denied by the host sandbox).

The Exp7 runners automatically use:

- `APPWORLD_PYTHON` or `$HOME/anaconda3/envs/appworld/bin/python`
- `AGENTBENCH_PYTHON` or `.venvs/agentbench/bin/python`

### Round sensitivity

Both artifact builders support `--round-limit N`, so `G*_N` and `EHar_N` can be
constructed from the same stored 8-round trajectories without retraining.

## Verification

The small frozen artifacts needed for held-out reproduction are tracked under:

```text
artifacts/
  frozen_test_attack_211.jsonl
  selected_harness.json
  MANIFEST.json
```

Their SHA-256 hashes are recorded in `artifacts/MANIFEST.json`.

After the pipeline:

```bash
python scripts/verify_artifacts.py
```

It checks:

- 632 training records
- frozen rule presence
- harness candidate pool and selection
- 211 validation attack records
- test results when available

## Outputs

All outputs go under:

```text
results/aaai27_supplement/
  artifacts/
  logs/
  pipeline/
  split/
```

Existing source data and result files are not overwritten.
