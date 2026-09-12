# Investigation of Kimi-K3 corruption under pressure

Upstream issue: https://github.com/sgl-project/sglang/issues/36859

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1060

## Outcome

`candidate_verified`

The recorded base already contains an issue-relevant correction from
https://github.com/sgl-project/sglang/pull/34820, commit
`13469c16d3e9f857c53c3f33da8d33268fd4f570`.

That change fixes KDA/Mamba radix-cache checkpoints for chunk-unaligned
prefixes. Before the change, the checkpoint was taken from a bf16 per-chunk
intermediate even when `--mamba-ssm-dtype` selected an fp32 state pool. A later
prefix-cache hit therefore resumed from a rounded recurrent state, while a cold
run kept fp32 state. This is a plausible mechanism for Kimi-family later-turn
divergence that becomes more visible with cache reuse under load.

The correction snapshots the kernel's fp32 accumulator at the tracked boundary
and casts it once to the configured pool dtype. The base ancestry check returned
zero, confirming the change is already included; no duplicate source edit was
made.

## Local evidence

The combined regression command was:

```bash
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-26a877e4a8a5/venv/bin/python \
  -m pytest -q -rs \
  test/registered/kernel/ops/attention/test_kda_track_state.py \
  test/registered/unit/layers/attention/test_mamba_track_state_dtype.py
```

It completed with `4 passed, 1 skipped, 5 subtests passed`. On the assigned AMD
Instinct MI355X (gfx950), the Triton KDA test compared the snapshot with an
independent run truncated at the 64-token boundary and passed at
`rtol=1e-5, atol=1e-5`. The regression also proves that the old bf16 intermediate
is exactly the rounded form of the fp32 snapshot and that the rounding is lossy.

Independent boundary coverage checks fp32, bf16, and fp16 state pools; an fp16
double-rounding discriminator; destination ordering and untouched rows; and the
aligned/no-unaligned-row case. Helion coverage skipped explicitly because Helion
is not installed.

Raw evidence is retained in `reports/j-26a877e4a8a5/raw/`.

## Scope and limitations

This is not a full reproduction of the reported request corruption. The report
used Kimi-K3 on 16 B200 GPUs or 8 H800 GPUs, while this job has one AMD MI355X
and no Kimi-K3 weights. It also lacks the reporter's prompts, request trace,
sampling seed, and deterministic oracle. The tiny Llama fixture was not used as
a substitute because it cannot exercise Kimi-K3's KDA architecture or validate
semantic multi-turn correctness.

Accordingly, the evidence verifies a source-level candidate already fixed in
the prepared base, but does not establish that it is the only cause of the
original distributed high-pressure symptom.
