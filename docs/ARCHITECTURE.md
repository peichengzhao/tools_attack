# Architecture

## Data Flow

```text
InjecAgent case split (632 / 211 / 211)
        |
        +--> co-evolution training (632)
        |       |
        |       +--> training trajectories
        |               |
        |               +--> frozen attack rule G*
        |               +--> harness candidate pool
        |
        +--> validation (211)
        |       |
        |       +--> frozen G* attack evaluation
        |       +--> harness candidate selection
        |
        +--> held-out test (211)
                |
                +--> frozen attack artifact
                +--> frozen defense harness
                +--> cross-evaluation and clean utility
```

## Components

### `InjecAgent/`

The main evaluation environment. It contains:

- victim prompting and output parsing;
- attack-payload providers;
- static prompt defenses;
- cross-evaluation and scoring;
- utility evaluation;
- split-aware result filtering.

### `experiments/aaai27_supplement/`

Reproducibility layer for the paper:

- restored split manifests;
- fixed-denominator metrics;
- co-evolution and ablation runners;
- frozen attack-rule construction;
- validation-only harness selection;
- held-out evaluation;
- interactive external-defense baselines.

### `exp7/`

Adapters for clean-utility and benchmark evaluations:

- AgentDojo;
- BFCL;
- AgentBench;
- AppWorld;
- InjecAgent.

Track-specific environments are isolated:

```text
main experiments: Python 3.10
AgentBench:       Python 3.9
AppWorld:         Python 3.11
```

## Evaluation Invariants

The paper-facing pipeline enforces several invariants:

1. Test cases never participate in candidate generation or harness selection.
2. The frozen attack rule is invoked once per held-out case.
3. Invalid outputs remain in the fixed denominator.
4. Adaptive and enhanced attacks cannot silently fall back to the seed attack.
5. Defense results retain their implementation provenance.

## Static and Interactive Defenses

Single-pass static defenses modify prompts or tool-output presentation:

- Sandwich;
- Spotlighting;
- StruQ-Prompt;
- SecAlign-Prompt;
- Ours harness.

Interactive defenses alter the agent execution loop:

- Task Shield;
- MELON;
- CaMeL.

These are reported in separate tables because their inference protocols differ.
