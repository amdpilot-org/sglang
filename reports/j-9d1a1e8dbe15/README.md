# Independent review of PR 1162

Reviewed exact candidate commit `3323334eb03732b320e266a8b67cb953a5aaa6a7`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the original
request-broadcast contract exercised by the issue. On the base, the real
four-process Gloo reproduction fails when a TP-group source holds `None`. On
the candidate, the CP rank-zero column is seeded first and then every TP row
fans out from a source that holds the list.

Evidence:

- `base-original-failure.log`: exit 1 on the recorded base, with
  `TypeError: object of type 'NoneType' has no len()` at the real
  `broadcast_pyobj` source path.
- `candidate-regression.log`: exit 0 at the exact candidate; all four ranks
  receive the request payload.
- `candidate-pytest.log`: 9 focused tests pass.
- `candidate-adversarial.log`: real Gloo collectives pass for TP×CP layouts
  2×3 and 3×2 (including an empty list), plus TP-only and CP-only boundaries.
- `environment.log`: the assigned environment exposes one AMD Instinct MI350X
  (`gfx950`) with Torch 2.11.0+rocm7.2.

The imported candidate source was
`/job/repo/python/sglang/srt/distributed/communication_op.py`. The diff is
Python-only, so no native rebuild was applicable. The original eight-GPU
DeepSeek-V4 serving command, DSV4 model execution, and NCCL behavior were not
run because only one GPU and no model weights were available. The real CPU
Gloo tests validate the failing serialization/collective contract but are not
a full-model or eight-GPU serving reproduction.

Upstream issue: https://github.com/sgl-project/sglang/issues/37590

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1198

Candidate: https://github.com/amdpilot-org/sglang/pull/1162
