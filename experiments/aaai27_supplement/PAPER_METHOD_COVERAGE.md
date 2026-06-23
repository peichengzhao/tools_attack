# Paper Method Coverage Audit

Audit target: the current AAAI paper version whose Table 1 contains Naive,
Combined, Important Instructions, Tool Knowledge, InjecAgent-Default,
InjecAgent-Enhanced, TopicAttack, ChatInject, Adaptive Attack, and Ours-Attack.

## Attack methods

| Paper row | Code identifier | Implementation status |
|---|---|---|
| Naive | `naive` | Implemented; can now run through cross-eval with `--defense none` |
| Combined | `combined` | Implemented |
| Important Instructions | `important_instructions` | Implemented |
| Tool Knowledge | `tool_knowledge` | Implemented |
| InjecAgent-Default | `injecagent_default` | Implemented as the raw seed attacker instruction |
| InjecAgent-Enhanced | `injecagent_enhanced` | Implemented deterministically by prepending the official hacking prompt; no cache needed |
| TopicAttack | `topicattack` | Implemented as a topic-transition payload in the same InjecAgent slot |
| ChatInject | `chatinject` | Implemented using the ChatInject multiturn data mapping |
| Adaptive Attack | `adaptive_attack` | Implemented as feedback-driven poison optimization; caches are incomplete |
| Ours-Attack | `ours_attack` | Implemented from the frozen `G*` JSONL artifact |

The default cross-evaluation grid now contains the nine Table 2 rows:

```text
injecagent_default
chatinject
topicattack
adaptive_attack
injecagent_enhanced
combined
important_instructions
tool_knowledge
ours_attack
```

Naive is an additional no-defense row and is enabled explicitly through
`--attack naive --defense none`.

### Important attack correctness constraint

Adaptive and InjecAgent-Enhanced must not silently fall back to the seed
attack. The provider now raises an error when a frozen poison is missing.
`--allow_offline_fallback` exists only for smoke tests and must not be used for
paper results.

## Defense methods in the paper grid

| Paper column | Code identifier | Status |
|---|---|---|
| Sandwich | `sandwich` | Implemented prompt baseline |
| Spotlighting | `spotlight` | Implemented delimiting baseline |
| StruQ | `struq` / displayed as `StruQ-Prompt` | Prompt-modified approximation, not the official trained StruQ model |
| SecAlign | `secalign` / displayed as `SecAlign-Prompt` | Prompt-modified approximation, not the official SecAlign/Meta-SecAlign model |
| Ours | `ours` | Implemented in cross-eval through `--harness_file` |

Cross-eval now produces a 225-cell manifest:

```text
5 models × 5 defenses × 9 attacks
```

For the paper's three-model Table 2, select the corresponding three models.

## Methods outside the paper's experimental tables

| Method | Repository status | Paper relationship |
|---|---|---|
| Task Shield | Integrated as a separate AgentDojo interactive baseline (third-party reproduction) | Not in Tables 1-3 |
| MELON | Integrated as a separate AgentDojo interactive baseline | Not in Tables 1-3 |
| CaMeL | Integrated as a separate AgentDojo interactive baseline | Cited as related work, not an experimental row/column |
| PIArena / PISanitizer | Integrated as an optional InjecAgent baseline | Not in Tables 1-3 |

These must not be described as paper Table 2 cells unless the paper is updated.

## Remaining data gaps

1. Adaptive Attack does not have complete frozen poison caches for all victim
   models. InjecAgent-Enhanced no longer needs a cache because it is a
   deterministic prefix transform.
2. The existing Ours-Attack test artifact has complete payload coverage for the
   211 test cases, but its stored victim outputs were produced while the API
   account was banned. Victim evaluation must be rerun.
3. Table 2 requires a per-model frozen harness. The code accepts one harness
   file per run; each model must be run with its own selected harness.
4. AgentBench clean-utility runs remain blocked until Docker is available.

## Representative commands

Table 2 Ours-defense column:

```bash
cd /home/zpc/model_attack/new_try/tools_attack/InjecAgent

/home/zpc/anaconda3/envs/tools_attack/bin/python src/evaluate_cross_eval.py \
  --model deepseek-v4-flash \
  --defense ours \
  --attack all \
  --test_data data/splits/test_cases_211.json \
  --harness_file ../results/aaai27_supplement/artifacts/selected_harness.json \
  --ours_attacks ../results/aaai27_supplement/artifacts/test_gstar_deepseek-v4.1-flash-1017.jsonl \
  --setting base \
  --prompt_type InjecAgent
```

Table 1 no-defense grid:

```bash
/home/zpc/anaconda3/envs/tools_attack/bin/python src/evaluate_cross_eval.py \
  --model deepseek-v4-flash \
  --defense none \
  --attack all \
  --test_data data/splits/test_cases_211.json \
  --ours_attacks ../results/aaai27_supplement/artifacts/test_gstar_deepseek-v4.1-flash-1017.jsonl \
  --setting base \
  --prompt_type InjecAgent
```
