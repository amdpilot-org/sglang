# Independent review of PR 834 at `bd2ec8e34d5dfbca94ed31ed264610ac60c1b5bf`

Upstream issue: https://github.com/sgl-project/sglang/issues/38207

Mirror issue: https://github.com/amdpilot-org/sglang/issues/875

Candidate: https://github.com/amdpilot-org/sglang/pull/834

## Recommendation

Accept the candidate as a narrow correction to the DCP-widened, no-RoPE MLA
KV-scatter contract. Do not describe the review as an end-to-end reproduction
or full resolution of the original GLM-5.3 report.

At the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
candidate's three focused tests fail on the assigned gfx950 GPU: raw widened
locations are used against a per-rank buffer and the reserved index is written.
At the exact candidate commit, all three pass. An independent PyTorch-reference
matrix also passes 130 nonempty cases plus an empty batch, covering DCP sizes
2, 3, and 8; every rank; int32 and int64 locations; an arbitrary reserved
index; and row widths 1, 127, 128, 129, and 513.

Source inspection connects the changed dispatcher to
`MLATokenToKVPool._scatter_mla_rows`: unresolved DCP write locations use
`set_mla_kv_buffer_dcp_sharded_triton`. The candidate makes the no-RoPE branch
match the existing RoPE branch by owner-filtering `loc`, dividing owned virtual
locations by the DCP width, and masking the reserved index.

## Scope and limitations

This review ran on one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) with ROCm
7.2 and PyTorch 2.11.0. The report requires eight NVIDIA B200 GPUs, CUDA 13,
FlashInfer/TRT-LLM MLA, GLM-5.3 weights, TP8/DCP8 communication, HiCache, and a
long-context serving request. Those prerequisites were unavailable. Therefore
the original illegal-memory-access symptom and B200-specific behavior were not
reproduced, and `fully_resolves_original` is false.

No native C++ source changed, so no native rebuild was applicable. Both base
and candidate runs imported SGLang from
`/job/repo/python/sglang/kernels/ops/kvcache/mla_buffer.py`; Triton came from
`/opt/venv/lib/python3.12/site-packages/triton/__init__.py`. GPU execution was
real, not mocked; only the process-global DCP rank/size accessor was patched to
exercise rank-local kernel behavior on the assigned single GPU.

The full test file passed 8 tests and skipped 58 CUDA-TMA-only cases on ROCm.
That suite result is supporting regression evidence, not proof of the original
CUDA serving outcome.
