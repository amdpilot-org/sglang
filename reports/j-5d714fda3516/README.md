# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/35096
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1434
- Related upstream candidate reviewed: https://github.com/sgl-project/sglang/pull/35097
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared base still had the defect: `apply_rotary_emb_flat_kernel` used the
power-of-two `RD` for all loads and stores while masking only rows. The focused
test was added first and run against that source. It failed for every tested
non-power-of-two width in the sentinel test, while the 64- and 128-column
controls passed. After the narrow kernel correction, all 15 cases passed on the
assigned gfx950 GPU.

The numerical oracle uses PyTorch complex multiplication rather than either
sibling Triton kernel. The storage-boundary test passes a narrow view of a wider
allocation and verifies that every column after `rope_dim` retains its sentinel.
Raw outputs are retained in `raw/`.
