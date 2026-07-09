# Reproducibility

## 1. Environment

Main experiments use Python 3.10.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.example.com/v1"
export VICTIM_MODEL="deepseek-v4-flash"
export OPTIMIZER_MODEL="deepseek-v4-flash"
export PROTECT_MODEL="deepseek-v4-flash"
export AAAI27_PROJECT_PYTHON="$(command -v python3)"
export AAAI27_MODEL_CONFIG="/absolute/path/to/model_api_ablation.py"
```

Optional benchmark environments:

```bash
bash exp7/setup_benchmarks.sh appworld
bash exp7/setup_benchmarks.sh agentbench
```

AgentBench worker startup additionally requires Docker:

```bash
bash exp7/build_agentbench_images.sh
bash exp7/start_agentbench_workers.sh
```

## 2. Repository and Split Checks

```bash
make check-repository
python experiments/aaai27_supplement/scripts/check_environment.py
python experiments/aaai27_supplement/scripts/check_baselines.py
python experiments/aaai27_supplement/scripts/check_external_benchmarks.py
```

The restored split is:

```text
train      632
validation 211
test       211
```

## 3. Dry Runs

No API calls:

```bash
make dry-run
```

Direct pipeline command:

```bash
cd experiments/aaai27_supplement
python scripts/run_pipeline.py \
  --stage all \
  --dry-run \
  --budget-calls 20000
```

## 4. Table 1 — Attack Comparison

The no-defense grid uses:

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

Attack identifiers:

```text
naive
injecagent_default
injecagent_enhanced
topicattack
chatinject
adaptive_attack
combined
important_instructions
tool_knowledge
ours_attack
```

## 5. Table 2 — Attack × Defense

Static defense:

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

## 6. Table 3 — Clean Utility and Artifact Combinations

Clean suites:

```bash
cd experiments/aaai27_supplement
python scripts/run_clean_utility.py \
  --selected-harness artifacts/selected_harness.json \
  --models gpt-5.5,deepseek-v4-flash,glm-5.1 \
  --benchmarks agentdojo,injecagent,bfcl,agentbench,appworld
```

Artifact combinations:

```bash
python scripts/evaluate_cross_combinations.py \
  --test-data ../../InjecAgent/data/splits/test_cases_211.json \
  --evolved-attacks artifacts/frozen_test_attack_211.jsonl \
  --selected-harness artifacts/selected_harness.json \
  --output-dir ../../results/aaai27_supplement/table3_cross
```

## 7. Interactive Defense Baselines

```bash
cd experiments/aaai27_supplement
python scripts/run_interactive_baselines.py
python scripts/build_interactive_table.py
```

Execution is intentionally opt-in:

```bash
python scripts/run_interactive_baselines.py --execute
```

## 8. Artifact Verification

```bash
python experiments/aaai27_supplement/scripts/verify_artifacts.py
```

## 9. Expected Warnings

A correct setup may still report:

- incomplete Adaptive Attack caches;
- AgentBench Docker unavailable;
- AppWorld or AgentBench environment paths missing before setup;
- PIArena defense model weights not downloaded.

These are explicit blockers, not silent fallbacks.
