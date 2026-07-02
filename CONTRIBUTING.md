# Contributing

## Development Checks

Before submitting a change:

```bash
make check-repository
make test
make compile
```

## Data and Artifact Policy

Do not commit:

- API keys or local model configuration;
- virtual environments;
- downloaded benchmark datasets;
- raw result JSONL files;
- offline poison caches;
- generated benchmark configurations;
- logs or caches.

Small split manifests and experiment manifests are intentionally tracked so the
evaluation protocol is reproducible.

## Method Provenance

Every external method must declare:

- upstream repository URL;
- pinned commit;
- official or third-party status;
- benchmark compatibility;
- required model weights or services.

Prompt-level adaptations must be labeled as such. Do not describe
`StruQ-Prompt` or `SecAlign-Prompt` as official trained models.

## API Usage

Large experiments must use:

- a hard call budget;
- bounded worker counts;
- resumable checkpoints;
- quota and ban detection.

Do not run unrestricted retry loops against shared API endpoints.

## Pull Requests

Include:

1. a concise behavioral description;
2. commands used for validation;
3. the affected result table or artifact;
4. any unresolved data or environment dependency.
