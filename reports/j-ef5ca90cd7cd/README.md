# gfx942 compact-versus-padded recurrence investigation

## Scope

This investigation checks whether compact and capped-padded representations of the same active tokens preserve the recurrence output and state on one assigned AMD Instinct MI300X (`gfx942`). It follows the read-only context in sgl-project/sglang issue 36481 and does not repeat the already-fulfilled scope in amdpilot-org/sglang issue 223.

The tested operation is the hybrid linear-attention target-verify recurrence:

- compact layout: `cu_seqlens = [0, L0, L0+L1, ...]`
- capped padded layout: the same active lengths followed by zero-length padding rows, with state indices `-1`
- identical active-token tensors and active state slots

No production code was changed because the demonstrated result was a pass, not a mismatch.

## Environment

- GPU: AMD Instinct MI300X, Device ID `0x74a1`, GUID `39656`, Node ID `3`, GFX `gfx942`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Triton: `3.7.0`
- Installed-source baseline commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Mirror checkout base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Paths

- Installed source: `/sgl-workspace/sglang/python/sglang`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Mirror checkout: `/job/sglang`
- Tested kernel: `/job/sglang/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py`
- Independent recurrent reference: `/job/sglang/python/sglang/kernels/ops/attention/fla/fused_recurrent.py`
- Installed baseline record: `/job/baseline-first.json`

## Baseline

The installed-source baseline ran before cloning or editing:

```bash
cd /sgl-workspace/sglang/test
/opt/venv/bin/python -m pytest registered/attention/test_chunk_gated_delta_rule.py::TestChunkGatedDeltaRule::test_padded_state_index_is_skipped -q --disable-warnings
```

Result:

- `1 passed, 3 warnings, 2 subtests passed in 17.01s`
- First GPU execution elapsed time: `18.821s`
- Timing method: shell `date +%s.%N` immediately before and after the single invocation

This baseline only validates the installed source tree and is not proof for later checkout changes.

## Checkout test

The checkout import was forced with:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python
```

The tested function was:

```python
sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update
```

The independent reference was a pure-PyTorch token-by-token recurrence using the native `[N, HV, V, K]` state layout, the same L2-normalized query/key path, the same softplus gate, and the same sigmoid beta. A second cross-check used `fused_recurrent_gated_delta_rule` with independently computed gates.

### Numerical gates

The gates were unchanged:

- output and state `torch.testing.assert_close(..., rtol=1e-2, atol=1e-2)`
- all output and state values finite
- untouched state slots exactly equal

### Finite adversarial matrix

| Case | Active lengths | Padded rows | H | HV | K | V | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `[1, 7, 3]` | 4 | 2 | 4 | 17 | 23 | pass |
| 2 | `[8, 1, 1, 4]` | 8 | 4 | 4 | 32 | 32 | pass |
| 3 | `[5, 2]` | 5 | 1 | 2 | 64 | 16 | pass |
| 4 | `[1, 3, 2, 1]` | 6 | 2 | 8 | 128 | 128 | pass |

Raw maxima:

- Case 1 compact output `0.0`, state `1.1920928955078125e-07`; padded output `0.0`, state `1.1920928955078125e-07`
- Case 2 compact output `0.0`, state `1.1920928955078125e-07`; padded output `0.0`, state `1.1920928955078125e-07`
- Case 3 compact output `0.0`, state `5.960464477539063e-08`; padded output `0.0`, state `5.960464477539063e-08`
- Case 4 compact output `0.0`, state `1.1920928955078125e-07`; padded output `0.0`, state `1.1920928955078125e-07`

The full checkout matrix elapsed `11.430s`. A cache-compliant rerun of case 1 with `TRITON_CACHE_DIR=/tmp/sglang-cache-j-ef5ca90cd7cd/triton` elapsed `8.924s` and passed.

## Honest boundaries

- No full model weights were downloaded or used.
- No full server boot, CUDA-graph capture, or end-to-end speculative-decoding run was performed.
- Issue 36481’s full-model illegal-memory-access reproduction remains unsupported in this environment because the required Qwen3.5 hybrid checkpoint and draft model were not available.
- The kernel-level compact-versus-padded recurrence property passed on `gfx942`; this does not prove the unrelated full-graph capture path from issue 36481 is fixed.
- No upstream issue, pull request, or comment was posted or modified.

## Conclusion

Compact and capped-padded representations with identical active tokens produced identical recurrence output and state within the unchanged numerical gates on one MI300X. No code change was warranted.
