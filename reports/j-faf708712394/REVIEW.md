# Independent review of amdpilot-org/sglang PR 989

Candidate: `a37c518dc725dd1856499e2f8aa035e8fd681bf7`

Upstream issue: https://github.com/sgl-project/sglang/issues/37707

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1024

## Verdict

Request changes. The candidate is a partial admission hardening: it rejects a subschema where `pattern` and `minLength`/`maxLength` are co-located, including the literal reported reproducer. It does not fully resolve the original contract because semantically equivalent constraints split across `allOf` remain admitted and xgrammar silently accepts the violating document.

## Evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, xgrammar accepted `{"v": "ab"}` for the original schema. It also accepted the same invalid document when `pattern` and `minLength` were split into adjacent `allOf` branches, both directly and through local `$ref`s.

At the exact candidate commit, the candidate regression passed (`6 passed`, `8 subtests`). Source imports resolved to `/job/repo/python/sglang/...`; xgrammar resolved to `/opt/venv/lib/python3.12/site-packages/xgrammar`. The literal schema was rejected before compilation, but both independent `allOf` counterexamples were not detected and reached the compiler. The real xgrammar probe confirmed both accept the invalid short string.

No native files changed, so no native rebuild was applicable. The machine exposed one AMD Instinct MI350X (`gfx950`) with Torch `2.11.0+rocm7.2`; GPU execution was unnecessary for this CPU grammar compilation/admission defect. No model-serving, semantic model accuracy, or distributed claim is made.

Raw commands and output are retained outside the switched checkout in `/job/review-evidence-j-faf708712394/`.

## Reproduction

```bash
/tmp/amdpilot-repo-j-faf708712394/venv/bin/python /job/review-evidence-j-faf708712394/probe.py
/tmp/amdpilot-repo-j-faf708712394/venv/bin/python -m pytest -q test/registered/unit/constrained/test_xgrammar_pattern_length.py
/tmp/amdpilot-repo-j-faf708712394/venv/bin/python /job/review-evidence-j-faf708712394/backend_probe.py
```

