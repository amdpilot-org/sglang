# Correction review: Kimi-K3 cross-prompt reasoning leakage

Upstream issue: https://github.com/sgl-project/sglang/issues/34259

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1744

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1645

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1710

Exact candidate reviewed: `bab33e2e1c26450edc7661eb4de484af3d331740`

## Outcome

The candidate is rejected as a resolution of the original issue. Its two added
offset tests are valid regression coverage for the optional returned-hidden-state
response path, so they are retained. They do not reproduce or explain ordinary
generated-text leakage.

No production source correction is justified by the available evidence. The
reported symptom remains unverified because the required Kimi-K3 weights,
eight NVIDIA B300 GPUs, tensor parallelism, and a deterministic reproducer are
not available in this environment.

## Independently reproduced counterexamples

### Ordinary generation bypasses the tested response slicer

In `process_batch_result_prefill`, `_append_prefill_hidden_states` is guarded by
`batch.return_hidden_states`. The sampled `next_token_id` is subsequently
appended to `req.output_ids` independently. A new boundary test supplies a
sentinel hidden-state tensor, sets `return_hidden_states=False`, makes the
response-slicing helper fail if called, and verifies that token 17 is committed.
The helper is not called.

This directly exercises the path omitted by the candidate. It does not prove
that ordinary generation is free of every possible cross-request defect; it
shows that the candidate's tested offset mechanism is outside the reported
default request path.

### The correction predates v0.5.17

`git merge-base --is-ancestor a0b7bcf59289a6cf916fa5ee44e3cfa865a25f3d
refs/tags/v0.5.17` exits 0. Commit `a0b7bcf5` is upstream PR #30177,
`[Feature] Support return_hidden_states="last"`, dated 2026-08-02. The official
`v0.5.17` tag resolves to `29481685`, dated 2026-08-07. Inspection of the tag
also shows the corrected packed offset accounting. Therefore the candidate's
claim that PR #30177 landed after the affected release is false, and restoring
that behavior cannot explain a symptom reported on v0.5.17.

### Candidate scope versus the reported workload

At the exact candidate commit, the focused suite passes: 9 tests, 17 warnings,
and 2 subtests. The candidate changes only tests and reports; it contains no
production source modification. None of its tests launches Kimi-K3, uses HTTP
concurrency, runs tensor parallelism on eight B300 GPUs, or repeats randomized
scheduling. The original report itself says the symptom occurs randomly and
provides no deterministic request sequence.

The qualified tiny-Llama serving fixture from mirror PR #649 was inspected. It
can validate transport and engine execution, but its random two-layer Llama
weights cannot validate Kimi-K3 architecture behavior, reasoning semantics, or
the reported eight-GPU scheduling regime. Running it would therefore not turn
the missing issue-specific reproduction into evidence for a source change.

## Preserved and added coverage

The candidate's mixed-request and cached/zero-forward-row hidden-state tests
are retained. This correction adds an ordinary-generation boundary proving
that requests which do not ask for returned hidden states bypass that response
slicing path while still committing their sampled token.

The consolidated suite passes with 10 tests and 2 subtests. A separate gfx950
execution confirms that the retained packed-offset helper selects rows 5-7 for
the second request and returns offset 8. This GPU check validates only that
helper, not Kimi-K3 generation or semantic isolation.

For failing-before/passing-after evidence, a controlled rollback moved the
offset increment behind the early return. The retained focused regression then
failed with `3 != 8`; after restoring the release/current implementation, the
full consolidated suite passed. This establishes the regression's value for
the helper, but it is not a reproduction of the source issue because the
rollback is older than v0.5.17 and ordinary generation bypasses the helper.

Raw outputs are under `reports/j-0047d505a4c6/evidence/`.
