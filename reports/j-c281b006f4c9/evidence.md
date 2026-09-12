# Independent review evidence for PR 1996

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1996 at exact commit `322cfb45762e9f6586a9751c073bb607142b1c41`.

Original issue: https://github.com/sgl-project/sglang/issues/34631

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2003

The image-prepared checkout was clean at the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Before committing this report, the
prepared `amdpilot/j-c281b006f4c9` branch was fast-forwarded to current mirror
`main` at `a207786205bff0919eb2c8c9126c67f302ccff34`, as required for delivery.

## Source and build scope

The candidate changes Python only in the implementation and tests:

- `python/sglang/srt/constrained/reasoner_grammar_backend.py`
- `python/sglang/srt/environ.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `test/registered/unit/constrained/test_reasoner_grammar_backend.py`

No C, C++, HIP, CUDA, Cython, header, or other native source differs from the
recorded base, so a native rebuild was not applicable. All commands used
`PYTHONPATH=/job/repo/python` and
`/tmp/amdpilot-repo-j-c281b006f4c9/venv/bin/python`. Runtime inspection resolved
`sglang`, the reasoner backend, and reasoning parser beneath `/job/repo/python`.
The environment had Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and one visible
AMD Instinct MI350X (gfx950-class) GPU. The tested state machine is CPU-side and
no GPU execution was needed or claimed.

## Reproductions

Recorded base, original issue stream:

```text
masked [False, False, False, False, False, False, True, True, True, True, True, True, True]
first 6 body_start 11
accepted [200022, 140680, 328, 76976, 200023, 2001, 2002]
```

This independently reproduces the issue: all five answer-channel header tokens
are constrained and fed to the inner JSON grammar.

Parent candidate `b52daa1dd1564889b9eb2e59ef8c6bfcd7e32091`:

```text
masked [False, False, False, False, False, False, False, False, False, False, False, True, True]
first 11 body_start 11
accepted [2001, 2002]
rollback0_noop False accepted [] calls [2]
```

This confirms both its valid channel-boundary fix and the prior independent
review's concrete zero-depth rollback regression.

Reviewed candidate `322cfb45762e9f6586a9751c073bb607142b1c41`:

```text
masked [False, False, False, False, False, False, False, False, False, False, False, True, True]
first 11 body_start 11
accepted [2001, 2002]
rollback0_noop True accepted [20, 21] calls []
```

The original boundary is corrected and `rollback(0)` is a complete no-op for
wrapper state, state history, inner accepted tokens, and inner rollback calls.

## Regression and adversarial validation

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-c281b006f4c9/venv/bin/python -m pytest -q test/registered/unit/constrained/test_reasoner_grammar_backend.py test/registered/unit/parser/test_reasoning_parser.py
156 passed, 17 warnings, 64 subtests passed in 12.54s
```

An independent seeded state-machine test compared the mutated object after each
valid accept or bounded rollback (including zero) with an oracle rebuilt by
replaying the surviving prefix. It covered partial multi-token terminators,
answer headers, repeated reasoning headers, fail-safe exits, and full rollback:

```text
replay_equivalence_cases 20397
```

`git diff --check` passed. Invalid negative and oversized rollback counts remain
unchecked, as in adjacent grammar backends; examined production callers use
positive unit rollback and bounded token counts. They are therefore not
counterexamples to the original contract or this correction.

## Limitations

Muse Glimmer 30B NVFP4 weights were unavailable, so the reported production
placeholder frequency, semantic answer quality, and a full HTTP serving run were
not reproduced. GPT-OSS behavior remains unverified. The deterministic scheduler
state-machine mechanism in the original issue and the concrete regression from
PR 1935 were both reproduced and corrected without model weights. The supplied
tiny Llama fixture would only validate transport/engine execution and cannot
qualify Muse channel semantics, so it was not substituted as proof.

Recommendation: **accept**. This is a full source-level fix for the original
issue's demonstrated contract, with no remaining in-contract counterexample.
