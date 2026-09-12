# Correction generation 2: chunked-prefill cancellation

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/1917 at exact
commit `3a1c517def5a96209f259fabe8df8cb4e8cbc85c`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1978.

Upstream issue: https://github.com/sgl-project/sglang/issues/34112

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2024

## Result

The candidate's source correction is justified and is preserved here. On the
recorded base, its issue-specific regressions failed because a cancelled final
prefill committed token/metadata, the beam case did not consume the pending
abort cleanly, and the request entered optimistic decode. At the exact
candidate commit, the focused suite passed (23 tests and 2 subtests).

The review's coverage claim is also correct: neither candidate regression
records the external probe's `schedule_batch` or `schedule_batch_post_run`
`output_ids`. The downloaded reproducer obtains those records through an
external `backends.sglang.tools.probe_runtime_structures` instrumentation layer
that is not present in this repository. Its attached historical result records
65 negative steps beginning in `schedule_batch_post_run`, with a minimum of
`-82`.

No further source edit is made for that observation. The required
`meta-llama/Llama-3.2-1B-Instruct` weights and RTX 4090/CUDA environment are
unavailable, and the probe instrumentation needed to map the recorded field to
an in-repository tensor is also unavailable. Treating an internal tensor as the
same field and clearing it would be speculative.

## Evidence

- `evidence/failing-before.txt`: three regressions fail on base `358c163...`.
- `evidence/passing-candidate.txt`: exact candidate focused suite passes.
- `evidence/original-probe.py` and `evidence/original-summary.json`: downloaded
  issue artifacts establishing precisely how negative scheduler IDs were
  counted and what the original environment observed.
- `evidence/model-search.txt`: no matching model directory in the prepared
  runtime/workspace.
- `evidence/gpu-probe.txt`: one assigned gfx950 GPU completed a Torch/ROCm
  tensor operation; this is environment evidence only.

No native source changed, so no native rebuild applies.
