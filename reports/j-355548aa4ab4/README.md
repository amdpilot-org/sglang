# Mixed-batch logprob normalization investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34719

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1519

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding and correction

The prepared source still contained unconditional `.tolist()` calls for
`next_token_top_logprobs_val`, `next_token_top_logprobs_idx`, and
`next_token_token_ids_logprobs_val` in both
`SchedulerBatchResultProcessor.move_logprobs_to_cpu` and
`SchedulerBatchResultProcessor._normalize_decode_outputs`.

Using the real methods with mixed tensor/list entries reproduced
`AttributeError: 'list' object has no attribute 'tolist'` for token-ID and top
logprobs at both sites. The correction only converts tensor entries, matching
the existing type handling in `GenerationBatchResult.copy_to_cpu`. Host-side
list placeholders are preserved.

The new regression covers both consumers and all three fields. Independent
boundaries cover already-host-side lists and homogeneous tensor entries,
including empty tensors.

## Validation

- Before the change, `raw/failing_before.txt` records all four direct cases
  failing on the assigned AMD Instinct MI350X (gfx950).
- After the change, `raw/regression_after.txt` records the four focused tests
  passing.
- `raw/unit_tests_final.txt` records 11 tests and 2 subtests passing across the
  new regression and the neighboring batch-result hidden-state tests.
- `raw/pre_commit_final.txt` records all applicable pre-commit hooks passing.
- `raw/gpu_conversion_reference.txt` records exact gfx950 output matching an
  independently specified Python-list reference.
- A deterministic two-layer random Llama server from the qualified fixture ran
  on the assigned GPU. Eight rounds of concurrent scored/plain requests each
  returned HTTP 200 with eight generated tokens, the server log recorded a
  decode batch with two running requests, and `/health` remained HTTP 200.
  Raw requests/responses, server log, fixture manifest, and lifecycle metadata
  are retained under `raw/`.

## Limitations

The server fixture validates HTTP transport and actual engine/scheduler
execution only. It does not validate semantic accuracy, the reported
Qwen3.5-9B-derived checkpoint, NVIDIA B200/CUDA behavior, or a distributed
workload. No production model weights were available. The reported
prefill-result consumer was reproduced directly with its real implementation;
the HTTP fixture demonstrated a co-batched decode path, not a distinct
prefill-only crash. No native library rebuild was needed because this is a
Python-only change.
