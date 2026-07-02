# Security Policy

## Authorized Use

This repository contains attack and defense research for indirect prompt
injection. Use it only in sandboxed environments or on systems for which you
have explicit authorization.

## Secrets

Never commit API keys, bearer tokens, private endpoints, or local model
configuration containing credentials.

The repository reads secrets from environment variables. A local
`model_api_secret.py` file is ignored by Git and may be used only for local
development.

If a credential is committed:

1. revoke or rotate it immediately;
2. remove it from the current tree;
3. assess whether Git history must be rewritten;
4. audit API usage and billing.

## Reporting

Report accidental credential exposure or unsafe default behavior privately to
the repository maintainers rather than opening a public issue containing the
secret.
