# FAQ

## Is this repository the official implementation of all listed methods?

No. Each method retains its documented provenance. MELON, CaMeL, and PIArena
are pinned official snapshots. Task Shield is a third-party reproduction.
StruQ-Prompt and SecAlign-Prompt are prompt-level adaptations, not the official
trained models.

## Why are some benchmark repositories not vendored?

AgentBench, AppWorld, and their datasets are installed by setup scripts because
they are large and have separate Python/runtime requirements.

## Why are result JSONL files not tracked?

Victim outputs depend on the selected API endpoint and provider behavior. The
repository tracks the frozen input artifacts and commands needed to regenerate
outputs.

## How do I reproduce the held-out attack without retraining?

Use:

```text
experiments/aaai27_supplement/artifacts/frozen_test_attack_211.jsonl
```

as `--ours_attacks`.

## How do I reproduce the selected defense harness?

Use:

```text
experiments/aaai27_supplement/artifacts/selected_harness.json
```

as `--harness_file`.

## Why does Adaptive Attack require a generation step?

AdaptiveAttackAgent optimizes against a specific victim. The upstream project
provides optimization code but does not distribute a universal precomputed
artifact for every victim model.

## Why does AgentBench require Docker?

AgentBench task workers execute isolated environments and services in
containers. The Python environment alone is insufficient for benchmark runs.

## Where are secrets stored?

Use environment variables or a local ignored configuration file. Never commit
API keys or bearer tokens.
