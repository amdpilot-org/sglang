# Independent review of amdpilot-org/sglang PR 1195

Candidate: `76e3c59c26476b8348743389b97bc650984ec87a`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/36598

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1227

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1195

## Recommendation

Accept as test-only hardening. The candidate does not introduce the production
fix: its parent, which is the recorded base, already contains the relevant
`is_deepseek_dsa()` early return in `get_dsa_seed_metadata_dim`. The exact
candidate adds a focused regression and does not change source or native code.

The prepared base therefore could not reproduce the scheduler failure through
the current helper. I did reproduce the original failing operation on the
prepared implementation by constructing the reported promoted GLM configuration
(`index_topk=None`, `index_share_for_mtp_iteration=True`) and invoking the
legacy DSA-only `get_dsa_index_topk` path: it raises the reported
`AssertionError`. Calling the current scheduler helper with the same top-level
and text-level configurations returns `0`, so the non-DSA guard is the behavior
that prevents the failure.

The existing source guard entered through upstream PR 37500, commit
`52fecfdf0908dca24f4c6799ff5967125cc4110e`, while the issue-specific upstream
PR 36645 remains open. This candidate is consequently not a new full fix or a
partial production fix; it is regression coverage for a full fix already in
its base.

## Independent cases

- The actual `Glm5NextConfig` promotion behavior was exercised at both the
  top-level config and `text_config`; both returned seed width `0` after the
  documented override state.
- A QSA-like non-DSA architecture with the sharing flag set returned `0`,
  confirming that the DSA guard does not consume another architecture's flag.
- Enabled DSA with `index_kpool=1` returned `2048`, and with
  `index_kpool=128` returned `2175`, matching the independent formula
  `index_topk + index_kpool - 1`.
- Enabled DSA with invalid `index_kpool=0` retained its invariant and raised
  `AssertionError`; the new guard does not mask errors on the enabled path.
- A missing sharing flag returned `0`.

No behavioral counterexample was found within the configuration-sizing
contract. One documentation defect is outside that contract: the candidate PR
and its committed report cite mirror issue 1140, whereas the reviewed mirror
issue is 1227.

## Environment and limits

Imports were confirmed from the checkout:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/disaggregation/utils.py`
- `/job/repo/python/sglang/srt/configs/model_config.py`

The assigned accelerator was one AMD Instinct MI355X, `gfx950`, under ROCm 7.2.
It is not the reporter's NVIDIA GB10 `sm_121`. GLM-5.3-Flash-NVFP4 weights were
not available, so this review does not claim full-model loading, HTTP serving,
semantic accuracy, TP2/EP2, multi-node, or SM121 kernel validation. The tiny
Llama fixture was not substituted because it cannot qualify the GLM-specific
configuration contract. The reviewed path is CPU configuration sizing and did
not require GPU execution.

The candidate changes no C++, HIP, CUDA, FlyDSL, or other native source, so no
native rebuild was applicable. Torch and ROCm were left intact.

Raw command output is retained in `commands.log`; the larger revision-specific
evidence set was kept outside the checkout at
`/job/review-evidence/j-7dc4bb8bf5d8` while revisions were switched.
