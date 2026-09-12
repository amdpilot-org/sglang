# DSpark draft LoRA candidate correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/37772

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2804

Candidate: https://github.com/amdpilot-org/sglang/pull/2710 at
`ad222b0a04e766e3feb92b67bac9c7d98ce943cb`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2773.

The candidate was checked out on the prepared base and its focused tests passed,
establishing the valid narrow behavior of one pinned, server-wide DSpark draft
adapter. A separate two-request regression then supplied distinct draft adapter
identities. The actual draft `ForwardBatch` contained the same fixed adapter ID
for both rows, reproducing the review counterexample.

No correction is proposed because a local routing substitution would be unsafe
and incomplete. There is no request-level draft adapter path/ID, independent
tokenizer registry and reference counting, draft-worker load/unload protocol,
scheduler capacity/admission policy, or mixed-adapter CUDA-graph lifecycle.
Those components must be designed together. Reusing the target `lora_id` would
couple identities for weights belonging to different model architectures.

No compatible DSpark checkpoint and stage-2 LoRA adapters were available, so
real-weight DSpark numerical behavior and CUDA-graph replay could not be
qualified. The tiny Llama transport fixture cannot validate DSpark semantics.
The candidate's partial fixed-adapter source changes are therefore not carried
into this correction PR; this report preserves their demonstrated value and the
precise remaining contract without presenting a partial feature as complete.

Raw outputs:

- `raw/candidate_exact_tests.txt`: candidate focused suite, 2 passed.
- `raw/candidate_adversarial_failure.txt`: distinct task adapters collapse to
  `server-wide-fixed-adapter`, 1 failed as expected.
