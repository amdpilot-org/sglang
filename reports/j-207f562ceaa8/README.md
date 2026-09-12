# Investigation result: issue already fixed on current main

Upstream issue: https://github.com/sgl-project/sglang/issues/34611

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2712

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Outcome: `not_reproduced`

The prepared checkout already contains the issue-specific solution. Upstream PR
https://github.com/sgl-project/sglang/pull/34560 merged as
`41cd5a718942f97bc45b0b5d7fca82992e8ae529` on 2026-08-14. Its motivation
quotes the same `HybridLinearKVPool.layer_num` scheduler-init traceback and
references the MI355X report.

The fix has two relevant parts:

1. Qwen3.5 draft remapping writes `num_nextn_predict_layers = 1` to
   `hf_text_config`, which is the configuration object read by
   `ModelConfig.num_nextn_predict_layers`.
2. `build_full_draft_pools` defensively unwraps a `HybridLinearKVPool` to its
   `full_kv_pool` before reading `layer_num`.

The current tree retains focused regressions for both behaviors. The historical
pre-fix and post-fix function excerpts are saved under `evidence/`, along with
the upstream PR metadata.

## Validation

- Focused unit suite: 14 tests and 4 subtests passed. See
  `evidence/focused_tests.log`.
- Independent boundaries: empty hybrid, non-empty hybrid, and non-empty plain
  pools all passed. See `evidence/draft_pool_boundary_check.log`.
- The assigned device was confirmed as one AMD Instinct MI350X with
  `gfx950:sramecc+:xnack-`. This was inventory only, not issue-valid GPU model
  execution. See `evidence/gpu_inventory.log`.

No source fix was duplicated because the precise correction and regression are
already present.

## Limitations

The Qwen3.5-class weights and original multi-rank topology were unavailable.
Consequently, this investigation does not claim a full-model serving,
performance, semantic-accuracy, or multi-node reproduction. The deterministic
tiny Llama fixture was not used because it cannot exercise the affected hybrid
Qwen3.5/MTP architecture and would only provide an unrelated transport smoke.
