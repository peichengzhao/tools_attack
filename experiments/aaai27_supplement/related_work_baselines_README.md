# Vendored Defense Baselines

These directories are pinned source snapshots, not live Git submodules.
Each snapshot contains:

- `VENDORED_COMMIT.txt`: exact upstream commit.
- `VENDORED_SOURCE.json`: source URL, provenance, and integration note.
- Upstream source and license files, unmodified except for the two metadata files above.

| Baseline | Upstream | Commit | Provenance | Native benchmark |
|---|---|---|---|---|
| MELON | https://github.com/kaijiezhu11/MELON | `4d3cc9c0175cc26332aac696ac5556c3f85a5e8e` | official | AgentDojo |
| CaMeL | https://github.com/google-research/camel-prompt-injection | `f083b6b396399d3b3c7f2ddaf613a5945eaf32d8` | official research artifact | AgentDojo |
| PIArena | https://github.com/sleeepeer/PIArena | `8bd7a89e4a88d64b8354d9aa80a684923be6b795` | official | InjecAgent / AgentDojo |
| Task Shield | https://github.com/cxkc19519-bot/task_shield | `aacac09bd9c652b3bd707df2e156cb684ae3f0a8` | third-party reproduction | AgentDojo |

Task Shield must be described as a **third-party reproduction**. No official
authors' Task Shield implementation was located when this snapshot was added.

PIArena is a platform, not one defense. The reproducible comparison point for
this project is **PISanitizer**, chosen from the defense used by PIArena's
official example experiment configuration. Do not label the result as the
entire PIArena platform's best defense.

Validate this directory from the supplement root:

```bash
python scripts/check_baselines.py
```
