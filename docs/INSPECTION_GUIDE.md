# Repository Inspection Guide

This guide is for reviewers checking whether the repository is prepared for
paper reproduction.

## 10-Minute Inspection

### 1. Repository hygiene

```bash
make check-repository
```

Expected:

```text
repository checks passed
```

### 2. Unit tests and syntax

```bash
make test
make compile
```

Expected:

```text
12 tests passed
```

### 3. Reproduction contract

```bash
python experiments/aaai27_supplement/scripts/reproducibility_audit.py
```

This checks split hashes, tracked artifacts, the 225-cell cross-evaluation
manifest, and pinned baseline commits. It does not call a model API.

### 4. Artifact integrity

```bash
cat experiments/aaai27_supplement/artifacts/MANIFEST.json
```

### 5. Dry runs

```bash
make dry-run
```

## Immediately Checkable

- Split restoration and integrity.
- Frozen Ours-Attack payload loading for all 211 cases.
- Selected-harness loading and prompt composition.
- Table 1/2 command generation.
- Table 3 artifact-combination command generation.
- Unit tests and repository hygiene.

## Runtime Prerequisites

| Requirement | Used by |
|---|---|
| OpenAI-compatible endpoint | All model-backed runs |
| Per-victim Adaptive artifacts | Adaptive Attack row |
| Per-victim harnesses | Ours column |
| Docker workers | AgentBench clean utility |
| AppWorld Python 3.11 environment | AppWorld clean utility |
| BFCL CLI | BFCL clean utility |
| PIArena model weights | PISanitizer baseline |

## Key Files

| Purpose | File |
|---|---|
| Repository overview | `README.md` |
| Table mapping | `docs/PAPER_TABLE_MAP.md` |
| Detailed commands | `docs/REPRODUCIBILITY.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Method audit | `experiments/aaai27_supplement/PAPER_METHOD_COVERAGE.md` |
| Prompt caveats | `experiments/aaai27_supplement/DEFENSE_IMPLEMENTATION_NOTES.md` |
| PIArena choice | `experiments/aaai27_supplement/PIARENA_DEFENSE_CHOICE.md` |
| Third-party provenance | `THIRD_PARTY_NOTICES.md` |
