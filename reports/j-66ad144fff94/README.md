# Independent review of PR 1276

Upstream issue: https://github.com/sgl-project/sglang/issues/37707

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1310

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1276

Reviewed exact candidate commit: `79543e9a469bdeb14a910c785aa735da26056056`

Recommendation: **request changes**.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
reproduces the original failure: raw xgrammar 0.2.1 accepts the invalid short
string and SGLang admits the schema. The candidate rejects the original schema
and both previously reported object-level `allOf` counterexamples, and its
focused regression suite passes.

The fix is incomplete for the original contract that a constraint must either
be enforced or rejected. An equivalent composition one object property deeper
is missed. In the following shape, both branches constrain the same `v.w`
instance location, but the candidate's collector combines schemas only for
`v`, then traverses each branch independently and never combines the two `w`
schemas:

```json
{
  "type": "object",
  "allOf": [
    {"properties": {"v": {"type": "object", "properties": {"w": {"pattern": "^[a-z]+$"}}}}},
    {"properties": {"v": {"type": "object", "properties": {"w": {"minLength": 5}}}}}
  ]
}
```

At the exact candidate, raw xgrammar accepts `{"v":{"w":"ab"}}` and
`XGrammarGrammarBackend.dispatch_json` returns `XGrammarGrammar`, rather than
`InvalidGrammarObject`. The same failure occurs when the two object branches
are reached through separate local `$defs` references. See
`evidence/candidate-probe.log`; the executable reproducer is `review_probe.py`.

No native source changed in the candidate, so no native rebuild was applicable.
Imports were confirmed from the checkout's
`python/sglang/srt/constrained/xgrammar_backend.py` and the prepared
`/opt/venv/.../site-packages/xgrammar` installation. The assigned gfx950 AMD
Instinct MI355X was inventoried, but GPU execution is irrelevant to this
deterministic CPU schema compiler/admission defect. No model, serving,
architecture-semantic, distributed, or multi-node claim is made.

Reproduce:

```bash
git checkout 79543e9a469bdeb14a910c785aa735da26056056
PYTHONPATH=python /tmp/amdpilot-repo-j-66ad144fff94/venv/bin/python \
  /job/repo/reports/j-66ad144fff94/review_probe.py
PYTHONPATH=python /tmp/amdpilot-repo-j-66ad144fff94/venv/bin/python -m pytest -q \
  test/registered/unit/constrained/test_base_grammar_backend.py \
  test/registered/unit/constrained/test_xgrammar_pattern_length.py
```
