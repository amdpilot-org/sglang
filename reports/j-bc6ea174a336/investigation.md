# Oversized hybrid-SWA admission correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31205

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3220

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2349 at `1d3dec7eab68bfaf1e28ab1068b2438543d3c406`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2457

## Reproduction before the correction

The candidate's production implementation was exercised directly with
`size_swa=rem_swa=512`, `page_size=512`, `max_new_tokens=1`, and a 2,000-token
prompt. `_swa_req_never_fits` returned true, `_swa_chunk_cap` returned zero,
and three consecutive calls to `add_one_req` returned `NO_TOKEN` without
setting an extend range. A separate 128-token request was admissible. This
confirms the review's permanent FCFS-head deferral counterexample.

## Correction

The candidate's valid safe-chunking fix and issue-shaped tests are preserved.
When an intrinsically oversized request cannot fit even one page-aligned chunk
after the entire SWA pool drains, `PrefillAdder` now returns a terminal reject
result. The scheduler removes that request, returns a 400 response, releases
cache-side state, and continues scanning the waiting queue. If the current SWA
availability is too small but the fully drained pool can fit a chunk, the
request still defers; transient pressure is not misclassified as permanent.

## Verification

- The complete PrefillAdder unit module passes: 34 tests and 12 subtests.
- The original reported 20,992-token pool geometry still admits the safe
  19,968-token chunk.
- Exact 512-token and below-one-page 1,023-token total-pool cases reject.
- A whole-request-oversized case with current capacity zero but a 3,072-token
  drained-pool chunk cap still defers.
- An impossible queue head followed by a runnable tail rejects the head and
  admits the tail in the same adder pass.

## Limitations

DeepSeek-V4-Pro/DSpark weights and the reported 1P1D TP4+TP4, mooncake,
multi-node, 8xB300 deployment were unavailable. This is deterministic
scheduler-admission coverage, not a full serving, model-semantic, transfer, or
distributed reproduction. The tested path launches no GPU kernels. No native
source changed, so no native rebuild was applicable. The prepared environment
does not contain `ruff`; `git diff --check` and Python byte-compilation passed.
