# Independent review of amdpilot-org/sglang PR 2055

Upstream issue: https://github.com/sgl-project/sglang/issues/32553

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2033

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2090

Candidate commit: `9daf99ce5d11f577eb16b755c606466742e9fb80`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept as test-only hardening of a runtime fix already present in the recorded
base. The candidate changes no runtime or native source. Its focused regression
passes and accurately protects the canonical DeepSeek-V4 prefill-CP argument
state from the removed legacy-field conflict.

This is not independently qualified as a full resolution of the original
eight-H20 DeepSeek-V4-Flash-FP8 serving workload. The available host has one AMD
Instinct MI350X (`gfx950`) on ROCm 7.2, current source intentionally rejects
prefill CP on HIP, and the reported model weights were unavailable.

## Evidence

The v0.5.16 tag (`d21f3c3a10606ba3c7bf43f981496da0a7d620cd`, checkout
at annotated-tag target `fdebc938f`) reproduced the exact exception using its
actual `ServerArgs._handle_legacy_cp_arguments` and
`ServerArgs._handle_context_parallelism` methods. The test constructed the
dataclass from its declared defaults without running unrelated model loading,
then applied the exact four assignments made by the v0.5.16 DeepSeek-V4 hook.
The first legacy projection set the generic prefill flag, the DeepSeek-V4 hook
set the DSA flag, and the final handler raised the reported mutual-exclusion
`ValueError`.

At both the recorded base and candidate, the full public parser plus
`prepare_server_args`, the actual current DeepSeek-V4 validator, and the general
context-parallel handler accepted:

`--model dummy --tp 8 --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave`

The resolved state was `enable_prefill_cp=True`, `cp_strategy=interleave`,
`attn_cp_size=8`, `enable_dp_attention=True`, and `moe_dense_tp_size=1`.
Neither removed legacy field exists in the current `ServerArgs` schema.

Independent adversarial cases established that missing and `zigzag` strategies
are rejected for DeepSeek-V4. A TP=7/attention-CP=4 input is normalized by the
DeepSeek-V4 validator to attention-CP=7 before the general divisibility check;
this is current intended behavior rather than a remaining counterexample.

The candidate's three selected regression tests passed. Candidate versus base
contains only test/report files, so no native rebuild was applicable. Imports
resolved to `/job/repo/python/sglang` and the prepared interpreter used
PyTorch `2.11.0+rocm7.2` with HIP `7.2.26015`.

## Classification and limitations

The original v0.5.16 validation failure is reproduced, and its specific
canonical/legacy flag conflict is absent from the recorded base. PR 2055 adds
test-only hardening; it does not itself implement the runtime correction.

Remaining unverified portions are full model configuration/loading for
DeepSeek-V4-Flash-FP8, eight-rank H20/CUDA execution, distributed collectives,
GPU kernels, semantic output, and performance. No GPU computation was used as
proof because the issue occurs during pre-execution argument validation and the
assigned HIP architecture is intentionally unsupported for current prefill CP.

