# Third-Party Notices

This repository includes or integrates the following third-party projects.
Upstream licenses and notices remain in effect.

## Vendored Source Snapshots

| Component | Source | Pinned commit | Provenance |
|---|---|---|---|
| MELON | https://github.com/kaijiezhu11/MELON | `4d3cc9c0175cc26332aac696ac5556c3f85a5e8e` | Official implementation |
| CaMeL | https://github.com/google-research/camel-prompt-injection | `f083b6b396399d3b3c7f2ddaf613a5945eaf32d8` | Official research artifact |
| PIArena | https://github.com/sleeepeer/PIArena | `8bd7a89e4a88d64b8354d9aa80a684923be6b795` | Official implementation |
| Task Shield | https://github.com/cxkc19519-bot/task_shield | `aacac09bd9c652b3bd707df2e156cb684ae3f0a8` | Third-party reproduction |

License files are preserved where provided by the upstream snapshot.

- PIArena: See `related_work/baselines/PIArena/LICENSE`.
- CaMeL: See `related_work/baselines/camel-prompt-injection/LICENSE`.
- MELON and Task Shield snapshots do not include an explicit root license file
  at the pinned commits. Consult the upstream repositories before redistribution.

## Integrated Benchmarks and Projects

The code also integrates with or adapts materials from:

- InjecAgent;
- AgentDojo;
- BFCL;
- AgentBench;
- AppWorld;
- ChatInject;
- TopicAttack;
- AdaptiveAttackAgent.

Benchmark repositories and datasets are cloned or downloaded separately by the
setup scripts and are not relicensed by this repository.

## Research Artifacts

Paper-specific split manifests, frozen attack payloads, and defense harnesses
are research artifacts. They are provided for reproducibility and must not be
used against unauthorized systems.
