# Independent review of amdpilot-org/sglang PR 765

Candidate reviewed: `0b57a72de32988e091215535a12007d56ef48bac`

Upstream issue: https://github.com/sgl-project/sglang/issues/38854

Mirror issue: https://github.com/amdpilot-org/sglang/issues/786

Recommendation: **accept**.

The candidate fully resolves the original reported default loading path. The
checkpoint configuration is automatically converted to `gptq_marlin`; at the
reported TP=1, Qwen3.5's fused BA projection has output width 96, which Marlin
cannot repack. The candidate instead creates an unquantized merged projection,
so the existing stacked loaders can consume the checkpoint's separate BF16
`in_proj_b.weight` and `in_proj_a.weight` tensors.

## Evidence

- On the prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`), the
  candidate regression failed in the two expected cases: widths 96 and 48
  retained GPTQ-Marlin.
- At the exact candidate commit, all 16 focused candidate/existing tests
  passed.
- The public checkpoint metadata was fetched independently. It contains 48
  `linear_attn.in_proj_a.weight` and 48 `linear_attn.in_proj_b.weight` entries,
  no corresponding BA `qweight` entries, and 48 packed `in_proj_qkv.qweight`
  plus 48 packed `in_proj_z.qweight` entries. Its GPTQ settings select
  `gptq_marlin` automatically.
- An actual `MergedColumnParallelLinear` was constructed through the candidate
  method, loaded from separate BF16 B/A tensors, moved to the assigned gfx950
  GPU, and executed. Its `(17, 96)` output exactly matched an independent
  `torch.nn.functional.linear` reference (maximum absolute error 0.0).
- Independent boundary cases covered a compatible 128-wide Marlin projection,
  a dynamic negative exclusion, and every valid TP divisor of the checkpoint's
  48 value heads. The compatible width stayed quantized; exclusions and all
  reported-shape TP partitions were unquantized.

The candidate commit's own `result.json` refers to raw log paths that are not
present in that commit. Those prose claims were not used as proof; this review's
independent raw logs are retained under `reports/j-5a1612f34970/raw/`.

## Limitations

- The assigned device is one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`),
  not the NVIDIA RTX 6000 Ada used in the report. The NVIDIA Marlin kernel and
  its exact repack exception were not executed locally.
- The full 27B weight shards were not downloaded and a full HTTP server was not
  launched. The validation covers the exact checkpoint metadata, selection and
  construction logic, stacked BF16 loading, and numerical execution of the
  affected projection; it does not establish full-model semantic accuracy.
- No native source changed, so no native rebuild was applicable. The imported
  Qwen3.5 source path was `/job/repo/python/sglang/srt/models/qwen3_5.py`.
- Explicitly forcing the removed non-Marlin `gptq` backend fails during layer
  construction in this environment. The original command does not force that
  backend and automatically selects `gptq_marlin`.
