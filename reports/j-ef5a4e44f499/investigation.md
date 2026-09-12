# Independent review of PR 1572

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1572 at exact commit `1551ab33f49374808645b8975343d587f43048bc`

Upstream issue: https://github.com/sgl-project/sglang/issues/36481

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1604

## Verdict

Recommendation: **request changes**. The candidate is a valid, narrow crash-avoidance fallback, but it does not fully resolve the original compact CUDA-graph contract. For a Triton-backed Qwen hybrid, compact target verification is deliberately excluded from CUDA-graph capture and replay and runs eagerly. The original NVIDIA B300/CUDA 13.2 illegal memory access in `extend_attention_fwd` was not reproduced or retested on matching hardware and weights.

This is a partial fix, not test-only hardening: the Python source guard prevents the known unsupported backend from entering capture-time warmup, while the existing replay admission guard keeps target verification eager. The tests accurately prove that control-flow behavior, but do not execute the failing kernel or hybrid model.

## Evidence

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Before switching revisions, the candidate's exact six-test regression was extracted outside the checkout and run against the base. Five tests passed and `test_compact_capture_skips_unsupported_backend_and_replay` failed because base `DecodeCudaGraphRunner.capture()` called the poisoned `warmup()`. This deterministically reproduces the capture-admission defect that permits the reported boot-time failure, but it is not a reproduction of the hardware illegal memory access.

At exact candidate commit `1551ab33f49374808645b8975343d587f43048bc`, the same regression passed all six tests. An independent adversarial script also passed. It checked all four full-attention/linear-attention capability combinations, repeated unsupported compact capture, eager replay admission, supported compact capture, and both capability values in non-compact mode.

Runtime source checks resolved `sglang`, `decode_cuda_graph_runner.py`, `triton_backend.py`, and `hybrid_linear_attn_backend.py` from `/job/repo/python`, not an installed source tree. `TritonAttnBackend.supports_ragged_verify_graph` was `False`. There are no native/C++ changes in the candidate, so a native rebuild was not applicable.

Raw review evidence was preserved outside the checkout at `/job/review-evidence-j-ef5a4e44f499/`, including the candidate diff, issue/PR metadata, base failure, candidate pass, adversarial pass, import/runtime inventory, and SHA-256 manifest.

## Remaining counterexamples and limitations

- Triton-backed Qwen hybrid compact target verification still does not use a CUDA graph. The source returns from `capture()` and `_can_run_ragged_verify_graph()` remains false, so the path is eager.
- The reported `bs=20`, eight-tokens-per-request B300/CUDA 13.2 illegal memory access was not reproduced or retested.
- Neither `extend_attention_fwd` nor a full-attention/GatedDeltaNet model was executed by the regression or independent checks.
- The assigned GPU is AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and Torch `2.11.0+rocm7.2`; CUDA is absent. Required Qwen hybrid and DSpark weights were unavailable. This environment cannot qualify NVIDIA B300 behavior or the original architecture-specific kernel path.

## Exact commands

Base regression (expected failure, exit 1):

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-ef5a4e44f499/venv/bin/python /job/review-evidence-j-ef5a4e44f499/test_candidate.py
```

Candidate regression (exit 0, six tests):

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-ef5a4e44f499/venv/bin/python test/registered/spec/dspark/test_ragged_verify_backend_capability.py
```

Independent adversarial checks (exit 0):

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-ef5a4e44f499/venv/bin/python /job/review-evidence-j-ef5a4e44f499/adversarial_review.py
```

Candidate diff whitespace validation (exit 0):

```bash
git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365 1551ab33f49374808645b8975343d587f43048bc
```
