# DSpark ragged CUDA Graph review correction

Upstream issue: https://github.com/sgl-project/sglang/issues/34384

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1723

Candidate: https://github.com/amdpilot-org/sglang/pull/1634 at
`0a0b787c4dbc878065c48926d18c5b43eafcf19a`

Independent review: https://github.com/amdpilot-org/sglang/pull/1688

## Result

The review counterexample is reproducible. With
`DecodeCudaGraphRunner._stage_ragged_verify_layout` temporarily disabled, the
candidate's new issue-specific helper test still passed, while the new
integration regression failed. The candidate therefore did not protect the
runner integration that refreshes pointer-stable capture buffers.

The prepared base already contains the justified runtime behavior. This
correction preserves the candidate's valid helper cases and adds coverage that:

- invokes the real runner staging method for 32 requests x 6 tokens into the
  192-slot capture;
- checks capture-buffer pointer stability;
- passes the staged capture layout through DSV4 replay layout resolution; and
- independently covers 32 requests x 5 tokens (160 live tokens) rounded to the
  192-token tier, where 32 synthetic request slots receive one token each.

No production source was changed because the available tests and gfx950 device
execution found no source defect after exercising these paths.

## Evidence

Mutation (failing before):

```bash
# Temporarily return at the start of _stage_ragged_verify_layout.
/tmp/amdpilot-repo-j-5ba9c158ba0a/venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_ragged_verify.py::TestPaddedRaggedVerifyGeometry::test_issue_34384_replay_geometry_matches_192_slot_capture
/tmp/amdpilot-repo-j-5ba9c158ba0a/venv/bin/python -m pytest -q \
  test/registered/unit/model_executor/runner/test_ragged_verify_replay_layout.py
```

The candidate test passed (exit 0); the integration suite failed three tests
(exit 1). Raw logs retain the controlled-mutation output.

Passing after restoring production staging:

```bash
/tmp/amdpilot-repo-j-5ba9c158ba0a/venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_ragged_verify.py \
  test/registered/unit/model_executor/runner/test_ragged_verify_replay_layout.py
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-5ba9c158ba0a/venv/bin/python \
  reports/j-5ba9c158ba0a/gpu_replay_geometry.py
```

Results: 15 tests passed. The assigned AMD Instinct MI355X (`gfx950`) executed
the real device tensor staging path. The 32x6 layout produced 160 zero-length
synthetic rows; the independent 32x5 layout produced 32 nonzero synthetic rows,
and both query indptrs ended at token tier 192.

## Limitations

The DeepSeek-V4-Flash DSpark checkpoint was unavailable. The assigned system
has one AMD gfx950 GPU, not four NVIDIA H20 GPUs. Consequently, the exact CUDA
Graph replay, Hopper behavior, TP4 collectives, full DSV4 attention execution,
and serving workload remain unverified. This correction qualifies the source
integration and device layout transformation only. No native source changed,
so no native rebuild was applicable.
