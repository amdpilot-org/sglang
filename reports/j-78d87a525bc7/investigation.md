# Investigation: DeepSeek-V4-Flash-0731 accuracy with DP < TP

Upstream issue: https://github.com/sgl-project/sglang/issues/33360

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1904

## Finding

The prepared base already contains the issue-specific correction. Upstream PR
https://github.com/sgl-project/sglang/pull/31700 identified the failure mode as
using a partial-value DP gather for tensors that had already been reduced and
replicated across the attention-TP group. With `attn_tp_size > 1`, the gather's
reduction summed identical replicas, multiplying the hidden-state magnitude by
`attn_tp_size` at every MoE layer and producing numerically corrupt output.

The corrected source uses `dp_gather_replicate` for the synchronous
post-attention hidden-state gather in
`DeepseekV4DecoderLayer._run_moe_ffn_dp_sync`. It also uses replicate gathers
for main-model and NextN token IDs and clones the ID view because the MAX_LEN
implementation may zero the local gather input on non-leader attention-TP
ranks. The genuinely partial TBO pre-MoE path remains on `dp_gather_partial`.

No production correction was duplicated. A regression test was added to pin
those three corrected boundaries and the intentionally partial TBO boundary.

## Evidence

- `regression-before.log`: the test fails against a temporary fixture restoring
  the historical partial hidden-state gather and non-cloned token-ID inputs.
- `regression-after.log`: the same regression passes on the prepared source.
- `gfx950_gather_reference.log`: single-GPU numerical mechanism check. Summing
  replicated values scales them by 2 and 4 for attention-TP widths 2 and 4;
  leader-only replicate semantics preserve the values. It also demonstrates
  that zeroing an `input_ids[:, None]` view mutates the source while zeroing a
  clone does not.
- `pr31700.json` and `pr31700.patch`: captured upstream related-fix metadata and
  patch are retained in the private runtime evidence directory
  `/tmp/amdpilot-repo-j-78d87a525bc7/`.

## Limitations

The assigned hardware is one AMD Instinct MI350X (`gfx950`), not eight NVIDIA
H800 GPUs. The DeepSeek-V4-Flash-0731 weights are unavailable. Therefore this
investigation did not reproduce the original model's generated text, execute
the NVIDIA Marlin kernel, or exercise distributed TP8/DP4. The gfx950 run only
validates the gather arithmetic and aliasing mechanism. The full-model,
multi-GPU evidence remains the upstream PR's documented H200/H100 reproduction,
not a claim made from this environment.
