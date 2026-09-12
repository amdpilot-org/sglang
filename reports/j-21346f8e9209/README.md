# DeepSeek V4 envelope correction review

This correction consolidates the valid parser repair from
https://github.com/amdpilot-org/sglang/pull/879 and independently checks the
counterexamples reported by https://github.com/amdpilot-org/sglang/pull/941 at
exact candidate commit `7d454f646cb8f7d1c0778eec666a62ad1c375121`.

The prepared base retained clean `arguments`/`args` envelopes. The candidate
normalizes clean schema-disambiguated object, JSON-string, alternate-key, and
repeated envelopes in both one-shot and three-character streaming paths. The
candidate's source repair and regression tests are retained here.

No further parser correction is justified by the three remaining examples:

- duplicated or corrupted inner command text has no recoverable source of truth;
- valid outer wrappers can be peeled, but syntactically damaged inner JSON
  cannot be decoded into tool arguments safely;
- without declared top-level schema properties, `arguments` can be a legitimate
  free-form key, so removing it would be ambiguous.

Regression coverage now explicitly records the clean quadruple-envelope repair
and preservation of these irrecoverable or schema-ambiguous boundaries. Raw
base/candidate output and test logs are in `raw/`.

The named `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` weights were unavailable.
Therefore this does not claim model-generation, vision-semantic, retry-loop, or
distributed reproduction. The visible assigned gfx950 GPU was inventoried but
not executed because this deterministic correction is Python parser-only.

Upstream issue: https://github.com/sgl-project/sglang/issues/38013

Mirror issue: https://github.com/amdpilot-org/sglang/issues/977
