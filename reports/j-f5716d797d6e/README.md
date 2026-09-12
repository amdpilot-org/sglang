# Independent review of PR 1122

Reviewed exact candidate commit `1b5d79c0a1c7553c3bcfac31ca07dd614897564c` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a meaningful partial source fix, not test-only hardening: its admission check rejects the original co-located schema and both string-level `allOf` counterexamples from PR 1048. Its regression suite passes. It does not fully satisfy the original issue's enforce-or-reject contract.

The remaining counterexample puts `properties.v.pattern` and `properties.v.minLength` in separate object-level `allOf` branches. Both branches constrain the same instance location under JSON Schema conjunction. The candidate walks each `properties` map separately and does not combine them, so `dispatch_json` reaches `compile_json_schema`. xgrammar 0.2.1 then accepts the invalid short document `{"v": "ab"}`. The equivalent construction through two local `$defs` references also remains admitted.

Evidence:

- `evidence/base-original-assert.log`: original issue reproduction at the exact base, expected assertion failure (recorded exit 1).
- `evidence/base-reproduction.log`: direct compiler behavior for the original and composed schemas at the base.
- `evidence/candidate-unit.log`: candidate suite, 8 passed and 14 subtests passed.
- `evidence/candidate-adversarial.log`: exact-candidate detector and direct xgrammar results, including the two missed cases (exit 1).
- `evidence/candidate-dispatch.log`: both missed schemas proceed to compilation in the actual SGLang dispatch method.
- `evidence/environment.log`: interpreter, source packages, ROCm, and available GPU architecture.
- `review_cases.py`: independent direct-compiler reproduction.

The imported backend source was `/job/repo/python/sglang/srt/constrained/xgrammar_backend.py`; xgrammar came from `/opt/venv/lib/python3.12/site-packages/xgrammar/__init__.py`. No native files changed, so rebuilding native code was not applicable. The available device is one AMD Instinct MI355X (`gfx950`), but this CPU-only admission/compiler behavior did not execute on GPU. Outlines behavior and serving/model execution were not qualified.
