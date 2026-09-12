# Investigation evidence

- Upstream issue inspected on 2026-09-12, including its follow-up reproduction comment.
- Mirror issue inspected: https://github.com/amdpilot-org/sglang/issues/734
- Actual draft checkpoint metadata inspected from `syvai/Qwen3.8-27B-DFlash2-W4A16`.
- Private runtime checkpoint path: `/tmp/amdpilot-repo-j-3fa85b9b5619/models/dflash2-w4a16`.
- Checkpoint size: 1,280,633,960 bytes.
- Architecture: `DFlash2DraftModel`.
- Quantization: compressed-tensors pack-quantized symmetric INT4, group size 128.
- Ignore rules cover `candidate_selector`, convolution `kernel_projection`, and other `hidden_projection` modules.
- Assigned device observed with `rocm-smi`: AMD Instinct MI350X, gfx950, one visible GPU.
- Full test raw result: `server_args.junit.xml` in this directory.

The checked-out implementation accepted a non-null DFLASH draft quantization without warning. It also intentionally inherits target quantization when the draft option is omitted, tracked by `_speculative_draft_quantization_explicitly_set`. The patch warns in both cases and identifies which source selected the quantization. It does not warn after `unquant` has resolved to `None`.
