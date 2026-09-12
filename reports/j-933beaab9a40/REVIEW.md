# Independent review of candidate PR 2267

Upstream issue: https://github.com/sgl-project/sglang/issues/31711

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2216

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2302

Candidate: https://github.com/amdpilot-org/sglang/pull/2267 at exact commit `cedc90e4f8b6847af919ad7d0cc13ebc8136823f`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original implementation-level issue.

The candidate is a direct child of recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Its production change replaces the full-history slice assignment with a guarded in-place tail deletion. This removes retained-history copying for the one-token EAGLE traversal rollback and makes `rollback(0)` a no-op for `accepted_tokens`, while continuing to call the underlying matcher with the requested count.

No counterexample remains within the reported contract. Zero, one, multiple, and full-history rollback cases preserve list identity and yield the expected history. One-token rollback cost was independently flat through a one-million-token retained history.

## Evidence

The prepared checkout initially matched the recorded base exactly. The active interpreter imported SGLang and `xgrammar_backend.py` from `/job/repo/python`, not from an installed SGLang wheel.

On the base, the independent probe observed:

- `rollback(0)` changed `[10, 20, 30, 40]` to `[]`.
- Positive rollback returned correct values but replaced the list object.
- Append plus `rollback(1)` grew from 1.410 us at 1,000 retained tokens to 2,113.055 us at 1,000,000 retained tokens.
- Traced peak allocation rose to 26,000,968 bytes in the one-million-token case.

At the exact candidate commit:

- Candidate regression and existing speculative grammar-tree traversal coverage passed: 7 tests.
- The independent boundary probe preserved list identity and returned expected histories for counts 0, 1, 2, and 4.
- Append plus `rollback(1)` remained between 0.222 and 0.287 us from 1,000 through 1,000,000 retained tokens.
- The exact candidate source was imported from `/job/repo/python/sglang/srt/constrained/xgrammar_backend.py`.

Raw captured values are retained under `reports/j-933beaab9a40/raw/`.

## Scope and limitations

The assigned single GPU was an AMD Instinct MI355X (`gfx950`). GPU execution was not used: the defect and change are Python list operations on the scheduler CPU path, and the candidate contains no native or GPU source changes. Consequently no native rebuild was applicable.

No EAGLE-compatible model weights were supplied, so this review does not claim a full serving reproduction, an end-to-end TPOT measurement, semantic model validation, or a distributed-workload validation. The provided deterministic tiny Llama fixture cannot qualify EAGLE speculative decoding and was not used as an unrelated substitute. These limitations do not leave an implementation-level counterexample to the original rollback contract.
