# Independent review of PR 1527

Upstream issue: https://github.com/sgl-project/sglang/issues/34857

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1567

Candidate: https://github.com/amdpilot-org/sglang/pull/1527 at
`034ffd7a3995b4cd941f38ee5398250be938b599`.

## Recommendation

Request changes. The candidate changes the returned tensor dtype to fp32, but
does not fully resolve the precision defect on the aiter fallback exercised by
this prepared gfx950 environment. For untuned bf16 shapes, aiter's
`torch_gemm` ignores `otype`, returns bf16 from `F.linear`, and `tgemm.mm`
casts that already-rounded result to fp32. All candidate outputs tested were
exactly representable as bf16. At the production `(8,7168,256)` shape the
candidate result exactly equaled the independent fp32 reference rounded
through bf16.

The new regression checks only dtype, shape, and finiteness, so it passes this
cast-after-rounding behavior. A regression tied to the precision contract
should also distinguish the result from a bf16-rounded reference, or the
implementation should explicitly document that only the container dtype is
intended.

## What was independently verified

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the real imported
  `aiter_dsv3_router_gemm` returned bf16 for production, fallback, and single-token
  shapes on the assigned MI355X/gfx950 GPU.
- At the exact candidate commit, its GPU regression passed (`1 passed`, two
  subtests) and all nine existing GLM correction-bias tests passed.
- The base already contains the GLM-5.2 correction-bias allocation and aiter
  boundary fix. An independent close-score routing case showed fp32 bias chose
  experts `[7,6]`, bf16-collapsed bias chose `[0,1]`, and the actual aiter path
  matched the fp32 result.
- No native source changed in the candidate. The imported SGLang module was
  `/job/repo/python/sglang/srt/layers/rocm_linear_utils.py`; the imported aiter
  binaries/cache were under `/tmp/amdpilot-repo-j-355f1205be42/cache/aiter`.
  A native rebuild was therefore not applicable.

## Limitations

No GLM-5.2 weights were available, so no full-model semantic or accuracy run
was performed. Testing used one assigned AMD Instinct MI355X (`gfx950`) with
Torch 2.11.0+rocm7.2 and HIP 7.2.26015; no multi-GPU or multi-node claim is
made. The installed aiter dispatcher reported no tuned entry for the tested
shapes, so tuned production kernels remain unverified. One exploratory script
ended nonzero after completing its GEMM measurements because its subsequent
first attempt at calling the routing reference used a stale argument; the
corrected routing command passed and is retained separately.

Raw outputs and the reviewed candidate patch are under `raw/`.
