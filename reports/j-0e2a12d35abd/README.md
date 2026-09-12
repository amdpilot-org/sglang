# Correction generation 2: xgrammar pattern/length admission

Upstream issue: https://github.com/sgl-project/sglang/issues/37707

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1215

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1122

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1182

The two review counterexamples were reproduced at candidate commit
`1b5d79c0a1c7553c3bcfac31ca07dd614897564c`: object-level `allOf` branches
could place `pattern` and `minLength` on the same property, yet admission did
not detect the combination and xgrammar 0.2.1 accepted `{"v": "ab"}`.

This correction preserves the candidate's direct, nested, property-level
`allOf`, and local-reference handling. It additionally collects property
schemas contributed by conjunctive object-level `allOf`/local `$ref` branches
and rejects the lossy combination before xgrammar compilation. It does not
merge alternatives or differently named properties.

Reproduce with:

```bash
PYTHONPATH=python /tmp/amdpilot-repo-j-0e2a12d35abd/venv/bin/python \
  reports/j-0e2a12d35abd/reproduce_review_counterexamples.py
PYTHONPATH=python /tmp/amdpilot-repo-j-0e2a12d35abd/venv/bin/python -m pytest -q \
  test/registered/unit/constrained/test_base_grammar_backend.py \
  test/registered/unit/constrained/test_xgrammar_pattern_length.py
```

No GPU execution or model weights are relevant to this deterministic schema
admission/compiler reproduction. Raw logs are retained under `evidence/`.
