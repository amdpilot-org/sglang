# Independent review evidence

Reviewed https://github.com/amdpilot-org/sglang/pull/1933 at exact commit `fc74e1f19a9d56bf9261ebf5abb3aad62ab92650` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/34740
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1969

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Imports resolved to `/job/repo/python/sglang`, including `detokenizer_manager.py` and `spec_utils.py`. Torch was `2.11.0+rocm7.2`; the assigned device was AMD Instinct MI350X (`gfx950:sramecc+:xnack-`). The candidate has no native changes, and the prepared environment declares `native: null`, so no native rebuild applied.

## Base reproduction

Running the candidate detokenizer fixture externally against the unchanged base produced `4 failed, 4 passed`. Complete U+FFFD encoded as `EF BF BD` and a literal U+FFFD token both emitted an empty chunk instead of committing immediately. Repeated stray continuation bytes left decoded text empty, including oversized events. A direct fixed-mode simulated-acceptance check produced `predict=[100, 100, 100, 100]`.

## Exact candidate

The candidate's focused regressions passed (`11 passed`). Adding adjacent stop-trimming coverage passed (`19 passed`). The previously reported `E4 | B8 AD` recovery-threshold case and complete-U+FFFD case are directly covered.

Independent seeded adversarial testing generated 335 byte streams with randomized event boundaries at lengths from 1 through 4096. After a clean token flush, candidate output exactly matched Python's independent `bytes(...).decode("utf-8", errors="replace")` result, and the uncommitted decode window stayed below 64 tokens. Separate 2-, 3-, and 4-byte valid scalars crossing the recovery threshold passed.

On gfx950, the synchronized fixed simulated-acceptance path used the supplied safe token id and produced `predict=[64, 64, 64, 64]`, `num_correct_drafts=[2]`.

Raw logs are preserved outside the checkout under `/job/review-evidence-j-8d94586045e5/`.

## Assessment

Recommendation: **accept**. This is a full source-level fix for both original issue mechanisms, not test-only hardening. No remaining counterexample was found. The exact DeepSeek-V4-Pro tokenizer, eight-GPU topology, and reported large serving benchmark were unavailable, so model-specific semantics and production-scale performance recovery remain unverified environment limitations rather than contrary evidence.
