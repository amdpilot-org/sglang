# Independent review of amdpilot-org/sglang PR 1269

Upstream issue: https://github.com/sgl-project/sglang/issues/36140

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1306

Candidate commit: `4b285c99dbd89dc86294cf0a7f03f5e0f1f2e78f`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the original issue through the report's
explicitly permitted fail-fast alternative. It does not claim or implement
DFLASH PD state transfer. Instead, it rejects DFLASH with either PD role during
normal server-argument resolution, before scheduler or worker startup, and
explains that draft KV/auxiliary hidden state is not transferred.

## Independent findings

On the recorded base, both `prefill` and `decode` DFLASH combinations pass
`handle_pd_disaggregation`. A direct exercise of the reported downstream path
also reproduces the exact failure:

```
AttributeError: 'NoneType' object has no attribute 'prepare_for_decode'
```

Source inspection confirms why: `SpeculativeAlgorithm.build_disagg_draft_input`
has branches for EAGLE and DSPARK but returns `None` for DFLASH, while
`spec_prepare_for_decode` unconditionally calls `batch.spec_info.prepare_for_decode`
for the DFLASH family.

At the exact candidate commit, its regression passes. More importantly, an
independent full `ServerArgs.resolve_once()` check with a non-special model path
rejects DFLASH for both PD roles with the intended actionable `ValueError`.
Independent direct-handler cases also confirm mixed-case DFLASH rejection and
that standalone DFLASH, PD DSPARK, PD EAGLE, and non-speculative PD remain
accepted.

The candidate changes Python only. No C++/CUDA/HIP/native source, build files,
or generated native library changed, so a native rebuild is not applicable.
Imports were measured from `/job/repo/python/sglang`, including
`/job/repo/python/sglang/srt/arg_groups/pd_disaggregation_hook.py`, rather than
from an installed SGLang wheel.

## Boundary caveat

`ServerArgs.resolve_once()` intentionally returns before the PD handler for the
special `model_path` values `dummy` and `none`. Consequently, those synthetic
records still accept DFLASH plus PD at the candidate commit. This is a genuine
validation bypass and the candidate's direct-handler regression does not cover
it. It is not a remaining reproduction of the original serving failure because
that early-return path does not construct the reported Kimi/DFLASH scheduler or
workers. It should nevertheless be considered if the project intends every
synthetic/dummy configuration to obey all compatibility gates.

An algorithm spelling with trailing whitespace also misses this particular
guard, but later speculative-algorithm validation rejects it as an unknown
algorithm; it does not reach the reported crash.

## Environment limitations

The prepared environment exposes one AMD Instinct MI350X (`gfx950`) with
PyTorch `2.11.0+rocm7.2` / HIP `7.2.26015`. The reported 8x RTX 6000D
(`sm_120`) multi-node topology, Kimi-K3 and Kimi-K3-DFlash weights, and
MoonCake RDMA transport were unavailable. No full-model, NVIDIA, multi-node,
RDMA, semantic-accuracy, or throughput claim is made. GPU execution was not
needed for the deterministic Python startup gate, and none is claimed.

Raw command output was preserved outside the revision-switched checkout under
`/job/review-evidence/j-7c98cb369d0f/`.

