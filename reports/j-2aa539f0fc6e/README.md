# Correction report

Candidate PR: https://github.com/amdpilot-org/sglang/pull/989

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1048

Upstream issue: https://github.com/sgl-project/sglang/issues/37707

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1081

The two review counterexamples were independently reproduced at exact candidate
commit `a37c518dc725dd1856499e2f8aa035e8fd681bf7`. In both cases, the candidate
admission check returned false, dispatch reached xgrammar, and xgrammar accepted
the invalid document `{"v": "ab"}`. The before probe therefore exited 1.

The consolidated correction retains the candidate's co-located and nested
subschema detection and additionally follows conjunctive `allOf` paths and
resolvable local JSON Pointer `$ref`s. It does not merge constraints across
alternatives or different property locations. After the correction, both review
schemas are rejected at admission while the same probe still demonstrates the
underlying xgrammar weakness; the probe exits 0 because admission now prevents
the lossy compilation path.

Raw output is retained in `evidence/`. This is a CPU-only JSON Schema admission
and compiler behavior investigation. No GPU execution, native rebuild, model
serving, architecture-semantic, or distributed claim is made.
