# Candidate response-envelope correction

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3195

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3237

Upstream issue: https://github.com/sgl-project/sglang/issues/35331

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3241

At exact candidate commit `c3fe7535cc959175a455ba226c11ace9776c6685`,
the Cosmos3 decoding output contained `candidate_group` and `candidates`, but
`action_generation_response()` rebuilt the default served envelope without
either field. The isolated failing reproduction is retained in
`failing_before.txt`.

The correction carries candidate metadata and optional raw candidates into the
standard envelope. Candidate arrays become lists for JSON responses and remain
NumPy arrays when `preserve_numpy=True` for msgpack. Focused response tests cover
both branches, and the original non-candidate behavior remains covered by the
existing action API suites.

No model weights were available. This result qualifies the deterministic
response-construction path, not Cosmos3 semantic accuracy or model-backed HTTP
execution. See `result.json` for the full limitations.
