# Correction generation 2 evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/34740

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1843

Candidate parent: https://github.com/amdpilot-org/sglang/pull/1732 at exact
commit `084b94f67bf37874a070e7e36199857bde662f25`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/1810

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Candidate reproduction

The two review counterexamples were added as independent assertions and run
against the exact candidate before the correction. The focused command reported
`2 failed, 7 passed`:

- `63 * 0x80 + 0xE4`, followed by `0xB8 0xAD`, produced 64 replacement
  characters instead of 63 replacement characters followed by `中`.
- A complete `0xEF 0xBF 0xBD` sequence emitted an empty string and remained
  uncommitted instead of immediately committing U+FFFD.

This confirms both review claims. The candidate's other seven focused tests
passed, so its tokenizer-derived simulated token and token-bounded recovery were
preserved.

## Correction

The commit gate now examines trailing byte-fallback token metadata (`<0xHH>`)
to identify a genuinely incomplete UTF-8 suffix. Complete U+FFFD output,
including a literal replacement token and the valid UTF-8 bytes `EF BF BD`, is
committed immediately.

At the 64-token recovery bound, only stable decoded text is committed. The
one-to-three byte-token incomplete suffix is retained, and its provisional
replacement character is excluded from surrounding-text length arithmetic so
continuation bytes in the next event can complete the original character.

## Passing-after evidence

```text
/tmp/amdpilot-repo-j-edc1be67fb1d/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_detokenizer_incremental_window.py \
  test/registered/unit/spec/test_simulated_acceptance_token.py
11 passed
```

The suite includes the original candidate regressions, both review cases, a
four-byte character split across recovery, and a literal U+FFFD token.

All pre-commit hooks passed for the corrected source and regression file.
The same focused suite with neighboring stop-trimming coverage reported
`19 passed`.

On the assigned GPU, the synchronized preserved simulated-acceptance tensor
path reported:

```text
device= AMD Instinct MI350X
arch= gfx950:sramecc+:xnack-
predict= [64, 64, 64, 64]
num_correct_drafts= [2]
```

Raw passing logs are retained under
`/tmp/amdpilot-repo-j-edc1be67fb1d/evidence/`.

## Limitations

DeepSeek-V4-Pro weights/tokenizer and the reported eight-GPU TP/DP/EAGLE
topology were unavailable. Model semantics, serving throughput, TTFT, and the
distributed workload were not reproduced or claimed. The deterministic
byte-fallback fixture validates incremental decoding control flow and the
single-GPU check validates only the tensor path. No native source changed, and
the prepared environment declares `native: null`, so no native rebuild applied.
