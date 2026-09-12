# HiCache mixed-batch round-head correction

Upstream issue: https://github.com/sgl-project/sglang/issues/38448

Mirror issue: https://github.com/amdpilot-org/sglang/issues/949

Candidate PR: https://github.com/amdpilot-org/sglang/pull/827

Independent review PR: https://github.com/amdpilot-org/sglang/pull/915

The candidate was checked out and tested at exact commit
`0b653c280b4e96ffa31571a4cff170c95f6322e8`. Its 21 ordering tests passed, but
source inspection confirmed that `pre_upload_forward_inputs` skipped the
prefill `input_ids` upload whenever `mix_running_indices` was present. The
candidate therefore called `ready_to_load_host_cache` before
`resolve_forward_inputs` issued that H2D copy.

The correction stages the mixed batch's prefill slice before the hand-over.
At forward entry, `resolve_forward_inputs` consumes that staged device tensor
and retains the required late FutureMap gather for decode tokens. An identity
guard falls back to the original upload path if the CPU source was replaced.

The correction regression failed on the candidate-derived tree with two
failures and passes after the source change. The focused suite passes 59 tests.
A real pinned nonblocking H2D check on the assigned AMD Instinct MI355X
validated mixed prefill staging, concatenation with a late decode token, and
one-shot consumption.

The reported NVIDIA H100 TP=8 multi-GiB HiCache workload was unavailable.
Accordingly, this work establishes deterministic source enqueue order and the
staging mechanics; it does not claim reproduction of NVIDIA copy-engine
arbitration, TP all-reduce delay, or an end-to-end speedup.
