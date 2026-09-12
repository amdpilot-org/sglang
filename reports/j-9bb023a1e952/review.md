# Independent review of amdpilot-org/sglang PR 803

Reviewed exact candidate commit `120e7d5efcba370a58285d9ad76d0150877e14b1`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the source-level contract
described by the original issue for `MLATokenToKVPool` retraction backup and restore.

## Findings

The base implementation passes DCP-widened logical IDs directly to the rank-local
physical MLA buffer. Independent negative controls demonstrated both failure modes:

- widened IDs outside the physical buffer raise `IndexError` at the same gather
  expression implicated by the reported CUDA device-side assert;
- widened IDs that remain in bounds silently gather the wrong physical rows.

The candidate filters IDs by ownership (`id % dcp_size == dcp_rank`) and converts
owned IDs to local rows (`id // dcp_size`) immediately before both physical backup
and restore. Restore was independently tested with newly allocated widened logical
locations. DCP=1 remained an identity operation. The separate Mamba/KDA state path
in `Req.offload_kv_cache()` and `Req.load_kv_cache()` continues to use
`mamba_pool_idx` and is not transformed by this MLA method.

The candidate's regression was copied outside the checkout and run against both
revisions: it failed 2/3 cases on the base and passed 3/3 on the candidate. An
independent adversarial harness passed on the candidate, including the silent
in-range corruption case. A real GPU numerical check on the assigned AMD Instinct
MI350X (`gfx950:sramecc+:xnack-`, ROCm 7.2) copied the expected owned rows to CPU
and restored them into different new logical locations.

The imported source was confirmed as
`/job/repo/python/sglang/srt/mem_cache/memory_pool.py`. The diff contains only
Python and test/report files, so no native rebuild was applicable.

## Evidence

Raw preserved evidence is under `/job/review-evidence/j-9bb023a1e952/`, including:

- `base-candidate-regression.log`
- `base-independent.log`
- `candidate-regression.log`
- `candidate-independent.log`
- `candidate-gpu.log`
- `candidate.patch`
- issue and PR metadata snapshots

## Limitations

The available machine has one AMD gfx950 GPU, not 8 NVIDIA B300 GPUs. Kimi-K3
weights and a TP8/DCP8 PD-disaggregated deployment were unavailable. Therefore the
full serving workload, capacity-triggered `retract_decode`, DSPARK/ReplaySSM, and
NVIDIA CUDA device-side assert were not reproduced end to end. The GPU validation
qualifies the changed physical gather/restore behavior only; it does not qualify
the full model, multi-rank orchestration, semantic accuracy, or performance.

Upstream issue: https://github.com/sgl-project/sglang/issues/38645

Mirror issue: https://github.com/amdpilot-org/sglang/issues/851

