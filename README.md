# Co-Evolving Poison Text and Defense Harnesses

Research code for evaluating indirect prompt injection (IPI) in tool-integrated
LLM agents under bilateral attack-defense co-evolution.

The project studies two artifacts produced from the same trajectory-driven
training loop:

- an evolved poison-text artifact (`Ours-Attack`);
- a validation-selected defense harness (`Ours-Defense`).

The repository contains the InjecAgent evaluation pipeline, held-out split
manifests, cross-evaluation scripts, benchmark adapters, and pinned snapshots
of external defense baselines.

> This code is intended for authorized security research and sandboxed
> evaluation. Do not use the attack artifacts against systems without
> permission.

## Repository Layout

```text
InjecAgent/                 InjecAgent evaluation and cross-evaluation code
exp7/                       AgentDojo/BFCL/AgentBench/AppWorld adapters
experiments/
  aaai27_supplement/        Reproducibility package and paper-table scripts
related_work/
  baselines/                Pinned MELON, CaMeL, PIArena, and Task Shield snapshots
model_api.py                Environment-based model configuration
paths.py                    Repository path resolution
scripts/check_repository.py Tracked-file, JSON, and secret hygiene checks
docs/                       Architecture and reproducibility notes
```

## Research Scope

The primary evaluation environment is InjecAgent DS-S1, with a restored split:

| Split | DS | DH | Total |
|---|---:|---:|---:|
| Train | 326 | 306 | 632 |
| Validation | 109 | 102 | 211 |
| Test | 109 | 102 | 211 |

The current paper-facing methods are documented in:

- [`experiments/aaai27_supplement/PAPER_METHOD_COVERAGE.md`](experiments/aaai27_supplement/PAPER_METHOD_COVERAGE.md)
- [`experiments/aaai27_supplement/DEFENSE_IMPLEMENTATION_NOTES.md`](experiments/aaai27_supplement/DEFENSE_IMPLEMENTATION_NOTES.md)
- [`experiments/aaai27_supplement/PIARENA_DEFENSE_CHOICE.md`](experiments/aaai27_supplement/PIARENA_DEFENSE_CHOICE.md)

`StruQ-Prompt` and `SecAlign-Prompt` are prompt-level adaptations inspired by
the corresponding papers. They are not the official trained StruQ or
SecAlign/Meta-SecAlign models.

## Requirements

- Python 3.10 for the main InjecAgent environment
- Python 3.9 for AgentBench
- Python 3.11 for AppWorld
- An OpenAI-compatible chat-completions endpoint for model-backed runs
- Docker for AgentBench task workers

Model configuration is read from environment variables. Never commit API keys:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.example.com/v1"
export VICTIM_MODEL="deepseek-v4-flash"
export OPTIMIZER_MODEL="deepseek-v4-flash"
export PROTECT_MODEL="deepseek-v4-flash"
```

## Quick Start

```bash
git clone git@github.com:peichengzhao/tools_attack.git
cd tools_attack

python3 -m pip install -r InjecAgent/requirements.txt

make check-repository
make test
make compile
make dry-run
```

The dry-run commands do not call a model API.

## Reproducibility

Detailed commands are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Typical validation:

```bash
cd experiments/aaai27_supplement
python scripts/check_environment.py
python scripts/check_baselines.py
python scripts/check_external_benchmarks.py
```

Cross-evaluation setup check:

```bash
cd InjecAgent
python src/evaluate_cross_eval.py \
  --setup_check \
  --setting base \
  --prompt_type InjecAgent \
  --test_data data/splits/test_cases_211.json \
  --ours_attacks ../results/aaai27_supplement/artifacts/test_gstar_deepseek-v4.1-flash-1017.jsonl
```

This validates the Table 2 grid:

```text
5 defense columns × 9 attack rows × 5 configured models = 225 cells
```

## External Baselines

The baseline source snapshots record their upstream URL, commit, and provenance:

- MELON: official implementation
- CaMeL: official Google Research artifact
- PIArena: official implementation, with PISanitizer as the fixed representative defense
- Task Shield: third-party reproduction, not an authors' official repository

See [`related_work/baselines/README.md`](related_work/baselines/README.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Task Shield, MELON, and CaMeL modify the agent execution loop. They are evaluated
in a separate AgentDojo interactive-defense table rather than mixed into the
single-pass Table 2.

## Environment Checks

```bash
# Pinned baseline sources and entry points
python experiments/aaai27_supplement/scripts/check_baselines.py

# AgentBench/AppWorld Python environments, data, and container runtime
python experiments/aaai27_supplement/scripts/check_external_benchmarks.py
```

AgentBench workers require a reachable Docker API. AppWorld uses a separate
Python 3.11 environment and downloaded task data.

## Known Constraints

- Adaptive Attack requires frozen per-victim poison artifacts. The upstream
  AdaptiveAttackAgent repository provides optimization code but no precomputed
  artifacts for this victim set.
- InjecAgent-Enhanced is generated deterministically from the official hacking
  prompt prefix and does not require a cache.
- The paper's Table 2 requires one validation-selected harness per victim model.
- AgentBench cannot run on hosts that do not expose a Docker API endpoint.
- External methods must be described according to their exact provenance. In
  particular, Task Shield is a third-party reproduction and StruQ/SecAlign are
  prompt-level adaptations here.

## Security

- API keys are never committed.
- Runtime outputs, offline poison caches, logs, and virtual environments are
  ignored.
- `scripts/check_repository.py` scans tracked files for secret-like values,
  generated files, and disallowed large artifacts.

See [`SECURITY.md`](SECURITY.md).

## Third-Party Code and Licensing

Upstream components retain their original licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

No repository-wide license has been declared yet. Contact the maintainers
before redistributing project code outside the upstream license terms.

## Contact

Use GitHub Issues for reproducible bugs and documentation problems.
