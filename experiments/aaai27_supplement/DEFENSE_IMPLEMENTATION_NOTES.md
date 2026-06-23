# Static Defense Implementation Notes

This project intentionally distinguishes between:

1. Reimplementations inspired by a paper's prompt design.
2. Full official model or system implementations.

The following Table 2 columns are **prompt-modified baselines**:

| Internal ID | Display label | What is implemented | What is not implemented |
|---|---|---|---|
| `struq` | `StruQ-Prompt` | A fixed system prompt that treats tool observations as data and reserves executable instructions for the user task block | The official StruQ model, structured-query training, and fine-tuned model weights |
| `secalign` | `SecAlign-Prompt` | A fixed system-prompt alignment policy that prioritizes the authenticated user request and rejects conflicting tool instructions | The official SecAlign preference optimization, Meta-SecAlign model, and model weights |

For this reason, results must not be described as “the official StruQ model” or
“the official SecAlign model.” The accurate wording is:

> StruQ-Prompt and SecAlign-Prompt denote prompt-level adaptations inspired by
> the corresponding defense papers. They are not the official trained models.

The exact prompt text is implemented in:

`/home/zpc/model_attack/new_try/tools_attack/InjecAgent/src/prompts/static_defense_baselines.py`

The internal IDs remain `struq` and `secalign` for compatibility with existing
result files, but generated table headers use the explicit `-Prompt` suffix.
