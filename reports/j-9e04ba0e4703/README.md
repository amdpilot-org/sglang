# Investigation report: issue 34384

Upstream issue: https://github.com/sgl-project/sglang/issues/34384

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1607

## Result

The prepared source at base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains the production behavior requested by the report. No runtime
correction was added.

Compact ragged graphs remain keyed by the rounded token tier, but replay no
longer presents the backend with the runtime request geometry. The runner:

1. captures the 192-token tier with `min(192, max_bs)` request slots;
2. pads a live 32-request layout to that captured slot count;
3. clamps each padded row to the configured verify width; and
4. copies the padded `verify_lens` and `qo_indptr` into the pointer-stable
   capture layout before replay.

For the reported width-six case, this produces exactly 32 rows of length six
followed by 160 zero-length rows. DSV4 also pads the resolved layout to the
graph batch size before constructing attention metadata. The existing graph
admission checks reject request counts above the captured slot capacity.

Regression coverage added here fixes the reported geometry in place and adds
the independent native-192-request boundary. Existing scheduler tests cover
the accepted maximum and the first rejected request count.

## Evidence

- `test_ragged_verify_before.log`: the pre-change focused suite passed (10
  tests), showing the source already had generic zero-length padding support.
- `test_ragged_verify_after.log`: the suite plus the two issue-specific cases
  passed (12 tests).
- `gpu_issue_geometry.log`: on the assigned AMD Instinct MI350X (`gfx950`),
  the actual tensor implementation produced 192 slots, 32 live slots, the
  expected `[6] * 32 + [0] * 160` lengths, and an independently constructed
  matching query indptr ending at 192.
- `test_scheduler_boundaries.log`: existing admission boundary checks passed
  for the maximum slot count and the first over-capacity count.
- `gpu_probe.log` and `gpu_inventory.log`: device/runtime inventory and a
  simple independent GPU arithmetic check.

Raw command output and exit-code files are retained in `raw/`.

## Limitations

The original DeepSeek-V4-Flash DSpark checkpoint was not available. The
assigned environment has one AMD gfx950 GPU, not four NVIDIA H20 GPUs, so the
reported TP4 CUDA Graph workload and its original illegal-memory-access
failure were not reproduced end to end. The GPU check validates the exact
layout transformation used by replay, not model semantics, CUDA/Hopper
behavior, or distributed execution. No native source changed, so no native
rebuild was applicable.
