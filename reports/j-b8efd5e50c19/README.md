# Independent review of PR 2273

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/2273 at exact commit `f6f603a21a6cd96b3b4713162564516bb56b2638`.

Upstream issue: https://github.com/sgl-project/sglang/issues/31720

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2209

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2307

## Recommendation

Accept the narrow loader correction, but do not describe it as a fully verified resolution of the original serving issue. The recorded base demonstrably leaves AWQ `qweight` unwrapped and uses logical widths where packed checkpoint widths are required. The exact candidate fixes both behaviors. Its regression suite passes, and independent target-dimension GPU checks verify exact splitting and reconstruction for `qweight`, `qzeros`, and group scales.

The original 27B checkpoint was not downloaded or served, and this host is AMD gfx950/ROCm rather than the reporter's NVIDIA sm_120/CUDA system. Therefore the reported degenerate generations and the CUDA AWQ-Marlin execution path remain unverified. This is a verified issue-specific loader fix, not proof of complete end-to-end resolution.

## Evidence

On base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an independently constructed `PackedvLLMParameter` with logical shard widths `[16, 8, 24]` behaved as follows:

```text
binding_calls [((0, 1, 2), (4, 6))]
direct_wrapper_error RuntimeError split_with_sizes expects split_sizes to sum exactly to 6 (input tensor's size at dimension 1), but got split_sizes=[16, 8, 24]
```

At the exact candidate commit, Python imported the source under `/job/repo/python/sglang`, not an installed copy. The candidate regression plus the pre-existing packed-loader suite passed: `10 passed`.

An independent gfx950 check used the public target configuration's exact GDN q/k/v widths (`16*128`, `16*128`, `48*128`) and AWQ pack factor 8:

```text
qweight shards [0, 1, 2] packed_widths [256, 256, 768] exact True
qzeros shards [0, 1, 2] packed_widths [256, 256, 768] exact True
scales widths [2048, 2048, 6144] exact True
noncontiguous shards [0, 2] widths [256, 768] exact True
device AMD Instinct MI350X gfx950:sramecc+:xnack-
```

The public checkpoint config identifies `quant_method: awq`, 4-bit packing, group size 128, and zero points. Its index contains `linear_attn.in_proj_qkv.qweight`, `.qzeros`, and `.scales`, matching the parameter names newly bound by the candidate. The config excludes `linear_attn.in_proj_b` and `linear_attn.in_proj_a` from quantization.

No C++ or other native source changed. A native rebuild was therefore not applicable. `py_compile` and `git diff --check` passed. Ruff was not installed in the prepared interpreter (`No module named ruff`).

## Classification

- Candidate code correction: verified for the isolated AWQ split-loading contract.
- Original issue: not fully resolved by available evidence.
- Regression: fails on the recorded base and passes at the exact candidate.
- Remaining verification: full `QuantTrio/Qwen3.6-27B-AWQ` serving probe and NVIDIA sm_120/CUDA AWQ-Marlin execution.
