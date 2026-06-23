# Frozen AAAI27-S1-v1 Split

This directory contains the restored split manifest used by the paper.

| File | Purpose |
|---|---|
| `split_manifest.csv` | 1,054 case-level split assignments |
| `train_case_ids.txt` | 632 training case IDs |
| `validation_case_ids.txt` | 211 validation case IDs |
| `test_case_ids.txt` | 211 test case IDs |
| `split_manifest_provenance.txt` | source hashes and generation procedure |

Manifest SHA-256:

```text
695ccb17ad3ea55fc6d9e51016605912e91511a9e4187f3497cfa9e3477f051d
```

The generator is `../scripts/generate_split_manifest.mjs`.

Do not regenerate or modify this manifest unless intentionally starting a new
experiment version.
