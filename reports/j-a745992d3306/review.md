# Independent review of candidate PR 2488

Upstream issue: https://github.com/sgl-project/sglang/issues/29272

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2520

Candidate: https://github.com/amdpilot-org/sglang/pull/2488 at
`04c853b98a9abea4e7f4ad1186613e59abeb8f0b`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a partial fix.

The local-device correction is justified: the recorded base selects `cuda:11`
when the global process-group rank is 11 even when the process-selected local
device is 0, while the candidate selects `cuda:0`. The candidate also recovers
from the reported `is_gds_supported(0) failed` constructor exception in a
single-process test by retrying with `nogds=True`.

The fallback is not coordinated across the process group, however. An
independent adversarial case makes rank 0's GDS copy succeed and rank 1's copy
fail. Rank 0 performs one copy and proceeds, while rank 1 closes its loader and
performs a second non-GDS copy. `FilesBufferOnDevice` documents that all workers
must enter its broadcast/scatter operations in the same order. Consequently a
rank-local failure after another rank has completed `copy_files_to_device()` can
leave the healthy rank entering tensor collectives while the failing rank is
still retrying, which can hang or desynchronize the original multi-node TP
workload. The candidate regression covers only a one-rank group and cannot
detect this counterexample.

A safe resolution needs a group-wide fallback decision before any rank starts
iterating tensors, or must select `nogds=True` up front for the affected
multi-rank configuration. A distributed regression should prove that one
rank's GDS failure causes every rank to take the same retry path.

## Evidence

- `raw/base_original_contract_failures.log`: candidate regressions run against
  the recorded base; both fail as expected (global-rank device selection and no
  GDS retry).
- `raw/candidate_unit.log`: the candidate's complete focused file passes, 25
  tests total.
- `raw/adversarial_uncoordinated_retry.log`: independent two-rank control-flow
  simulation; the healthy rank performs one GDS copy while the failing rank
  performs a GDS copy and a non-GDS retry.
- `raw/import_paths.log`: the tested module came from the candidate checkout at
  `/job/repo/python/sglang/srt/model_loader/weight_utils.py`.
- `raw/gpu_environment.log`: real arithmetic ran on the one assigned AMD
  Instinct MI350X, gfx950, and matched an independent CPU result.

No native source changed, so a native rebuild was not applicable. The prepared
interpreter does not include `fastsafetensors`; loader behavior was tested with
deterministic doubles. Only one GPU and one node were available, and GLM-5.2
weights were unavailable, so the original four-node TP launch and a real
fastsafetensors/model load could not be performed. The GPU check establishes
the available architecture and execution only, not the disputed distributed
loader behavior.

The candidate's `git diff --check` is also nonzero because its committed raw
pytest logs contain trailing whitespace. This is packaging hygiene, not the
basis for the request-changes recommendation.
