# Mamba/GDN TARGET_VERIFY GPU investigation

## Conclusion

Upstream issue 34786 is already fixed on this mirror's `main`. Commit
`1c4892d7b` (upstream PR 27998) guards an absent
`kv.mamba_next_track_idx` by defaulting it to ping-pong slot 0, and commit
`7c9257529` (upstream PR 30437) adds the lazy speculative track plan consumed by
TARGET_VERIFY. No production change is needed.

This change adds a tiny synthetic GPU regression test for the metadata path. It
checks that TARGET_VERIFY gathers the intended state destination when the
next-track index is absent and when an explicit lazy plan overrides it, and that
stale track mask/seqlen metadata is cleared.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, unique ID `0x586343f382a69f1d`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4` from
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Installed SGLang source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Delivery checkout: `/job/sglang`, branch
  `amdpilot/j-4b63b1982e19`.
- Job-private Triton cache: `/tmp/sglang-cache-j-4b63b1982e19`.
- The environment imports the native AIter module from
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

The installed-source baseline is recorded in `baseline-first.json`. It is only
evidence for the pre-existing installed source, not for later checkout changes.

## Installed-source baseline

The first successful GPU execution was the existing fused GDN TARGET_VERIFY
precision case `test_fused_gdn_mtp_precision[1-1]`. It compares
`fused_sigmoid_gating_delta_rule_update` with the independent
`fused_gdn_gating` plus
`fused_recurrent_gated_delta_rule_update` reference under the unchanged
`torch.testing.assert_close(..., rtol=1e-2, atol=1e-2)` gate. It passed in
8.707857 seconds.

The installed Mamba state-index test could not be collected because the installed
tree lacks `sglang.test.ci`. The concrete error is recorded in
`baseline-first.json`. A supported inline control then ran the installed
`fused_replay_state_indices` kernel against the unfused aten reference for
`total_bs=7`, `valid_bs=5`. State indices, the padded request-index side effect,
padding sentinels, and both guard tails were bit-identical/unchanged. The
synchronized case passed in 1.495804 seconds.

## Mirror GPU validation

All commands used `PYTHONPATH=/job/sglang/python` and
`TRITON_CACHE_DIR=/tmp/sglang-cache-j-4b63b1982e19`.

```bash
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/mamba/test_fused_replay_state_indices.py \
  -q -s -p no:cacheprovider

/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_fused_verify_triton_gdn.py \
  -q -s -p no:cacheprovider

/opt/venv/bin/python -m pytest \
  test/registered/unit/spec/test_mamba_target_verify_track_indices.py \
  -q -s -p no:cacheprovider
```

Raw commands, outputs, and monotonic timings are in `mirror-gpu-results.json`.

- Mamba state indices: 3 passed, 71 subtests passed, 16.752 seconds. The fused
  Triton kernel remained bit-identical to the unfused aten reference.
- GDN TARGET_VERIFY: 13 passed, 13.265 seconds. Output gates were unchanged.
  Single-step state max differences were `1.77e-03` (N=1), `2.19e-03` (N=16),
  and `3.01e-03` (N=128), each with a 0.00% failure rate.
- New TARGET_VERIFY metadata test: 2 passed, 15.720 seconds.

## Limitations

- Results are specific to one MI300X (`gfx942`) with the preserved ROCm/Torch
  stack; they do not establish NVIDIA CUDA behavior.
- The investigation used tiny synthetic states and existing kernel references;
  no full model weights were downloaded and no full-model E2E run was performed.
- The installed source and delivery checkout are different revisions, so the
  installed baseline cannot prove behavior of the checkout changes.
- `ruff` was not installed in the qualified environment. `git diff --check`
  passed, and the focused GPU tests passed.
