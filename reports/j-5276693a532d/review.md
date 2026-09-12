# Independent review of SageAttention candidate

- Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3281
- Exact commit: `7c1ffa54523e3c943470f2407757fce78fbe3bb9`
- Upstream issue: https://github.com/sgl-project/sglang/issues/1763
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/3285
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a partial experimental integration, not a full verification of the original feature request, and an independently exercised decode configuration produces incorrect attention semantics.

## Failing before

At the recorded base, `sage` is absent from both `ATTENTION_BACKEND_CHOICES` and the attention registry. The reproduction assertion exited 1, establishing that the requested backend did not exist.

## Candidate regression

At the exact candidate, the focused Sage tests passed (`10 passed`). Imports resolved to `/job/repo/python`, not an installed SGLang copy. However, the tests inject `torch.nn.functional.scaled_dot_product_attention` as `backend.sageattn`. They do not import SageAttention or execute its INT8 Q/K implementation.

## Independent decode counterexample

The candidate permits `sage` as a combined backend and as a decode backend. Its inherited decode path invokes `_attention` with `is_causal=True`. For ordinary autoregressive decode this supplies one query token and a longer KV cache.

A deterministic case captured the call as:

```text
q_len=1
kv_len=3
is_causal=True
candidate output first values: [1.0, 0.0]
correct last-token full-cache output first values: [1.6895, 0.3108]
```

The pinned SageAttention docstring says causal mode is only applicable when query and KV lengths are equal. Under SDPA causal semantics, a length-one query is position zero and attends only the first key, rather than representing the final token at cache position two. The candidate should either reject Sage decode configurations or implement correct offset-aware decode semantics and test them against the real kernel.

## Pinned dependency and SM120

At SageAttention commit `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`, the missing comma is real:

```text
SUPPORTED_ARCHS = ['10.012.0', '8.0', '8.6', '8.9', '9.0']
```

But an AST/source audit shows `SUPPORTED_ARCHS` is assigned and never loaded. The active setup path separately handles `12.0`, and the dispatcher has an `sm120` branch. Thus the claim that this malformed set itself rejects an SM120 source install was not reproduced. SM120 still remains unverified because no NVIDIA hardware or CUDA compiler was available.

## Architecture and evidence limits

The assigned GPU was AMD Instinct MI350X (`gfx950`) with Torch `2.11.0+rocm7.2`. `nvcc` and `nvidia-smi` were unavailable. SageAttention is NVIDIA CUDA-only, so no native extension rebuild/import, INT8 Q/K execution, compiler/ISA inspection, quantized numerical comparison, or NVIDIA architecture qualification was possible.

No model-quality, long-context, serving, throughput, latency, or memory benchmark supports the issue's same-accuracy or approximately 2x speed motivation. Dense paged-KV gathering and materialized GQA/MQA expansion also remain unmeasured.

Raw command outputs were preserved outside the candidate checkout under `/tmp/amdpilot-repo-j-5276693a532d/review-evidence/` while switching revisions.
