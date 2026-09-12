# Ollama endpoint regression investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/37711

Mirror issue: https://github.com/amdpilot-org/sglang/issues/922

The prepared base reproduced both defects in the actual Ollama handler. The
focused regression failed because `/api/chat` passed a mapping-like tokenizer
result to `GenerateReqInput`, and because the default completion length remained
2048 despite only six tokens being available in the test context.

The correction asks the tokenizer for a flat token list and bounds only the
implicit Ollama default by `context_len - prompt_length - reserved_tokens`.
Explicit `num_predict` values retain their existing behavior.

## Evidence

- Before: the new focused test had 3 failures and 2 passes. The chat failure
  was `ValueError: input_ids should be a list of lists for batch processing`;
  the small-context assertion observed 2048 instead of 6. Raw output:
  `/tmp/amdpilot-repo-j-cc8f3488602a/baseline-test.log`.
- After: the same test had 5 passes. Raw output:
  `/tmp/amdpilot-repo-j-cc8f3488602a/fixed-test.log`.
- GPU HTTP check: a deterministic two-layer random Llama ran on the assigned
  single gfx950 GPU. `/api/chat` returned HTTP 200 with four completion tokens;
  `/api/generate` without `num_predict` returned HTTP 200 with 125 completion
  tokens under a 128-token context. Raw requests, responses, server log, and
  cleanup metadata are under
  `/tmp/amdpilot-repo-j-cc8f3488602a/ollama-http/`.
- Fixture weights were generated outside the worktree at
  `/tmp/amdpilot-repo-j-cc8f3488602a/tiny-random-llama/`, seed `20260912`,
  SHA256 `6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`.

## Limitations

The GPU check qualifies the HTTP transport and engine execution for a synthetic
Llama fixture only. Qwen/Qwen3-0.6B weights were not available, so this does not
claim Qwen architecture behavior or semantic accuracy. No multi-node behavior
was tested. This Python-only change did not require a native rebuild.
