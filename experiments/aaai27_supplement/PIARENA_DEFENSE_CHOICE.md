# PIArena Representative Defense

PIArena is an evaluation platform containing multiple defenses, not a single
defense method. Comparing “PIArena” as one Table 2 column would therefore be
ambiguous.

## Selected defense

`pisanitizer`

## Why PISanitizer

1. PIArena's official example configuration uses `defense: pisanitizer`:
   `configs/experiments/my_experiment.yaml`.
2. PISanitizer is a prevention/sanitization defense, matching the evaluation
   protocol used by our defense: poisoned context is cleaned and then passed to
   the victim model.
3. PIArena's official website result data shows a balanced tradeoff on its four
   agent-benchmark entries: mean measured ASR is `13.75%` and mean utility is
   `55.67%`. More aggressive detectors reach lower ASR by rejecting most
   attacked requests, which makes their utility incomparable or near zero.

Source snapshot:

`related_work/baselines/PIArena/website/data/results.json`

This is a reproducible comparison choice, not a claim that PIArena declares a
single universally best defense. If the paper requires a different PIArena
defense, change the fixed default consistently in:

- `configs/baselines.json`
- `scripts/run_baseline.py`
- `scripts/run_piarena_injecagent.py`

The current adapter maps the paper's 211-case InjecAgent DS split into
PIArena's native InjecAgent layout before running PISanitizer.
