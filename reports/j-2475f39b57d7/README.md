# gfx942 expert-route permutation investigation

## Result

- The current mirror (`0084030179bfba86bfeb6d43f7997d4076329d2c`) passed all 48 supported, finite, tie-free expert-label permutation cases on one assigned MI300X (`gfx942`).
- Covered layouts were `[1, 4, 16] × 896`, top-k 16, ungrouped, row-contiguous scores, contiguous bias, bf16 and fp32, with and without renormalization.
- Four bounded permutations were tested: identity, reversal, and two deterministic random permutations. Expert IDs transformed exactly with the inverse label map; weights stayed within the unchanged `rtol=1e-5`, `atol=1e-6` gate. The maximum observed weight delta was `2.9802322387695312e-08`.
- The kernel matched AIter for every one-token case and an independently derived CPU oracle for every case.
- A finite adversarial exact-tie boundary was tested separately. Arbitrary expert-label permutation is not invariant there: the wave64 tie traversal is part of the operation contract and depends on expert position. The kernel nevertheless matched both AIter and the CPU oracle on the permuted tie input.
- No mismatch was demonstrated, so no production or test code was changed.

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-2475f39b57d7/jit
export TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-2475f39b57d7/tvm-ffi
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-2475f39b57d7/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-2475f39b57d7/torch-inductor
AMD_SERIALIZE_KERNEL=3 /opt/venv/bin/python \
  /job/sglang/reports/j-2475f39b57d7/reproduce.py
``+

The bounded run took 7.431 seconds in the recorded execution. It synchronizes after each kernel call and contains no warmup, occupancy, sleep, or unbounded loop.

Raw machine-readable results are in `gpu-evidence.json`. The required installed-source baseline is recorded separately in the job workspace as `/job/baseline-first.json`: 47 existing radix-4 router tests passed in 38.839 seconds (36.61 seconds reported by pytest), including Torch-oracle and AIter comparisons. That baseline is labeled installed-source evidence and is not proof for this checkout.

## Context

- Read-only upstream context: sgl-project/sglang issue 32312 requests low-latency small-batch MoE kernels.
- Upstream PR 34490 introduced the radix-4 router and its AIter tie contract; this investigation tests an additional property rather than repeating that implementation.
- amdpilot-org/sglang issue 229 and PR 330 covered low-token MoE alignment and unsupported tiny-batch candidates. This scope is distinct: expert-label permutation behavior of the supported radix-4 route kernel.
- No upstream issue, PR, or comment was posted or changed.

## Boundaries

- Exact ties are not expert-label permutation invariant. This is expected because `kAiterTieLaneRank` and `tie_priority` define selection by wave64 traversal position; changing expert labels changes that position. The tie-boundary result is therefore an unsupported boundary, not a kernel failure.
- Grouped routing, top-k other than 16, more than 1024 tokens, non-row-contiguous scores, noncontiguous bias, and dtypes outside bf16/fp32 are outside `covered()` and were not claimed.
- The timing characterizes the bounded route/reference calls only; it is not full MoE FFN latency or end-to-end model latency.
- No full model weights, framework replacement, node-wide state change, or artificial GPU burn was used.
- An early temporary probe using an experimental reference path hit an asynchronous HIP unspecified launch failure before producing a complete result. It was discarded and is not used as evidence; the final reproduction uses normal finite bf16 values, a CPU oracle, AIter, and explicit synchronization.
