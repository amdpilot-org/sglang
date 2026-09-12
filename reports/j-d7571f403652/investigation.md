# Fastsafetensors multi-rank GDS fallback correction

Upstream issue: https://github.com/sgl-project/sglang/issues/29272

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2574

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2488 at `04c853b98a9abea4e7f4ad1186613e59abeb8f0b`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2569

## Finding

The candidate correctly changes the CUDA device from the global process-group
rank to `torch.cuda.current_device()` and adds a GDS-specific non-GDS retry.
Its retry decision is local, however. A deterministic two-rank Gloo regression
ran the actual candidate control flow with a loader double: rank 0 completed
its GDS copy while rank 1 raised `is_gds_supported(0) failed`. The candidate
recorded attempts `[(0, False), (1, False), (1, True)]`, proving only rank 1
retried before the ranks would enter `FilesBufferOnDevice` tensor operations.

The correction reduces a local copy result across the process group before any
buffer tensor is requested. If any rank reports a GDS error, every rank closes
its first loader and repeats the copy with `nogds=True`. Non-GDS errors remain
errors and are also propagated group-wide so a healthy rank cannot continue
into later collectives alone.

## Evidence

- `raw/candidate_failing_before.log`: the two-rank regression fails against the
  candidate behavior because rank 0 does not retry (pytest exit 1).
- `raw/coordinated_passing_after.log`: the same regression passes after the
  group-wide decision (pytest exit 0).
- `raw/final_unit.log`: the complete focused test file passes, 25 tests.
- `raw/gpu_smoke.log`: real arithmetic on the assigned AMD Instinct MI355X
  (`gfx950`) matches an independently computed CPU result.
- `raw/related_history.log`: related path history inspected on the prepared
  base; no existing main-branch correction was present in the prepared clone.

## Limitations

The prepared interpreter has no real `fastsafetensors` installation, the job
has one GPU and one node, and GLM-5.2 weights are unavailable. Therefore this
does not claim a real GDS transfer, multi-node TP launch, full model load, or
GLM-5.2 semantic validation. The distributed regression validates coordinated
control flow using two real Gloo ranks and a deterministic loader double. No
native source changed, so no native rebuild was applicable.
