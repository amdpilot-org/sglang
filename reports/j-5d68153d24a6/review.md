# Independent review of fastsafetensors multi-node fallback

Upstream issue: https://github.com/sgl-project/sglang/issues/29272

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2609

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2605

Candidate commit: `1f0aaaf3ea2c3019010f352ee614d71424128900`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate preserves the earlier local-device correction and makes
the GDS fallback decision collective before any `FilesBufferOnDevice` tensor
broadcast. It also propagates non-GDS and fallback-pass failures to otherwise
healthy ranks, preventing them from entering later broadcasts alone.

The original issue's two source-level defects are fully addressed. This is not
a claim that the exact reported 4-node GLM-5.2 deployment was executed here.

## Evidence

On the prepared base, the candidate regression failed in both relevant ways:
the loader received `cuda:11` for global rank 11 instead of local device 0, and
an asymmetric GDS failure was not retried group-wide. See
`raw/base_candidate_regressions.log`.

The prior candidate at `04c853b98a9abea4e7f4ad1186613e59abeb8f0b` was also
checked directly. Its two-rank run recorded only rank 1 taking the nogds retry,
reproducing the review counterexample. See
`raw/pr2488_counterexample_reproduction.log`.

At the exact reviewed commit, all seven focused fastsafetensors tests passed,
including local-device selection, coordinated two-rank fallback, non-GDS error
preservation, cleanup, and cache behavior. See `raw/candidate_regressions.log`.

The independent adversarial runner uses two real Gloo processes. It verified:

- a constructor-time GDS failure on rank 1 makes both ranks retry nogds;
- both ranks subsequently execute a real tensor broadcast and receive 37;
- an asymmetric non-GDS error stops both ranks before tensor broadcast; and
- an asymmetric failure during the nogds retry stops both ranks before tensor
  broadcast.

The run completed with `ADVERSARIAL_ASSERTIONS_PASSED`; its script and output
are retained as `raw/adversarial_fastsafetensors.py` and
`raw/candidate_adversarial.log`.

The prepared interpreter did not include fastsafetensors. For an additional
integration check, the unmodified 0.4.0 wheel was downloaded without
dependencies, unpacked outside the checkout, and placed before the repository
on `PYTHONPATH`. Import inspection showed SGLang loading
`/job/repo/python/sglang/srt/model_loader/weight_utils.py` and the loader from
the unpacked wheel. On the assigned AMD Instinct MI355X (`gfx950`), exact GPU
tensor loads passed with zero error for explicit nogds and for GDS requested.
In the latter case, fastsafetensors itself warned that `/dev/nvidia-fs0` was
absent and selected its nogds copier. See the two `real_fastsafetensors_*` logs.

Inspection of fastsafetensors 0.4.0 confirmed that `copy_files_to_device()`
performs local submit/wait work and constructs `FilesBufferOnDevice`; tensor
broadcast occurs later from `get_tensor()`. The candidate's status all-reduce
therefore occurs before the relevant broadcast sequence.

## Scope and limitations

- The host has one AMD Instinct MI355X GPU (`gfx950`) and one node, not the
  reported NVIDIA multi-node topology. No real NCCL/RCCL multi-node run or GDS
  transfer was possible.
- GLM-5.2 weights were unavailable, so no full model load or semantic test was
  performed. The tiny GPU fixture validates loader data movement only.
- The prepared environment lacked an installed fastsafetensors package; the
  real-package checks used an unpacked 0.4.0 wheel outside the worktree.
- The candidate changes only Python and tests. No native source changed, so no
  native rebuild was applicable.
