# SGLang issue 35122 follow-up: coherent generation validation

## Status

**DRAFT — generation validation explicitly PENDING.**

This branch is cut from `amdpilot-org/sglang` `main` and merges the exact PR57 head
`484c2286c993d36e862343c390a77439a003d244`. The original base is
`db272201a2dbd72e5699e443240a851f1313ad45`, and the previously used source commit is
`082ad8ce15176ac80fa0afd4daa3a3ef71bbb126` (the same converter/source diff as PR56 at
`710dc165936d617826c492016ed9189875376bdc`). The exact source files are preserved rather
than replaced from today's mirror `main`.

The prior TP8 run established original-checkpoint loading and full decode graph capture,
but only short raw `/generate` completions. It did **not** establish coherent longer
generation. This follow-up will test the checkpoint's official chat framing and tokenizer
configuration before making any evidence-driven production change.

## Planned validation

- Verify the mounted original checkpoint index and metadata without modifying weights.
- Record GPU, Torch/HIP, AITER, Transformers, and active attention backend identities.
- Launch TP8 on eight assigned MI300X/gfx942 GPUs with the existing Triton FlashMLA
  backend selection and fresh startup/capture evidence.
- Use the official tokenizer/chat template and EOS/stop configuration.
- Run fixed greedy prompts at temperature 0, top_p 1, a fixed supported seed, and at
  least 128 output tokens, preferably 256, twice per prompt.
- Preserve complete raw requests and responses, timing, token counts, finish reason,
  repeat consistency, and bounded code execution results.

No semantic correctness is claimed from HTTP 200, startup, graph capture, or one/two-token
prefixes. Results will be added to this report as they become available.
