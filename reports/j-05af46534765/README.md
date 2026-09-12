# gfx942 decode attention boundary and padding follow-up

## Scope

- Follow-up to amdpilot-org/sglang issue 205, focused on xAI temperature boundary
  values and padded-row independence in the exposed split-KV logits/LSE buffers.
- Read-only upstream context: sgl-project/sglang issue 32942 and pull request 32943.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Existing mirror candidate: `686f106c829d55e1a6c49e728cd79ef48a412c8b`, open as
  amdpilot-org/sglang pull request 244. It already covers the identical non-unit
  unified-extend scaling scope from issue 205, so this follow-up did not repeat or
  reapply that fix.
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`, 206,141,652,992 bytes.
- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`,
  local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python`; Torch `2.9.1+rocm7.2.0.git7e1940d4`;
  HIP `7.2.26015-fc0010cf6a`.

The unified extend kernel does not expose logits or LSE outputs. The relevant exposed
contract is therefore the Triton decode split-KV stage: `attn_logits` contains per-split
normalized P·V partials and `attn_lse` contains the matching per-split log-sum-exp.
The verify split-KV and verify MLA paths explicitly reject `xai_temperature_len > 0`,
so they are not supported controls for this property.

## Installed-source baseline

The installed source was `/sgl-workspace/sglang`, imported through
`/sgl-workspace/sglang/python/sglang/__init__.py`. Its attention implementation was
`/sgl-workspace/sglang/python/sglang/kernels/ops/attention/extend_attention.py`.
The installed `sgl_kernel` package was
`/opt/venv/lib/python3.10/site-packages/sgl_kernel`, version `0.4.6.post1`. Aiter
loaded `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

The first successful GPU baseline was:

```bash
cd /sgl-workspace/sglang
timeout 300 /opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention_unified_vs_regular -s
```

Timing used `date +%s%N` immediately before and after the bounded command. The first
GPU execution took 41.499743 seconds; pytest reported 37.48 seconds. Two subtests
passed and one failed with maximum absolute difference `0.1669921875`. This is an
installed-source baseline only and is not evidence for the mirror checkout.

## Mirror probes

All probes used `/opt/venv/bin/python`, `PYTHONPATH=/job/sglang/python`, one gfx942
GPU, finite seeded bfloat16 random QKV, and float32 Torch references. The references
computed each split's scaled logits, `logsumexp`, and normalized P·V partial directly;
they did not call the kernel under test.

### Grouped decode: boundary and padded rows

Command shape:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python timeout 180 /opt/venv/bin/python \
  reports/j-05af46534765/reproduce.py
```

The case used batch sizes/sequence lengths `[4, 5, 6]`, 18 query heads, 2 KV heads,
head dimension 64, one split, and `xai_temperature_len=4`. Eighteen query heads with
a group size of nine forces a second 16-wide head tile with valid heads 9–17 and
padded lanes 18–24. The LSE tensor was a contiguous view at the front of a larger
sentinel-filled storage buffer, so writes beyond the contracted head count were
observable without relying on an out-of-bounds fault.

Raw results:

- Query positions `[3, 4, 5]`.
- Expected scales `[1.0, 1.0, 1.160964047443681]`.
- Maximum LSE error: `2.384185791015625e-07`.
- Maximum split-output error: `0.0028290487825870514`.
- Maximum final-output error: `0.015625`.
- Guard entries changed: `0`.
- All outputs finite: true.
- One-call elapsed time including Triton JIT: `1.8007478201761842` seconds.

The valid LSE and split partials matched the independent reference, and the padded
lanes did not modify the guard storage. This is a bounded numerical probe, not a
performance benchmark.

The committed `reproduce.py` was rerun once for delivery validation. It exited 0 in
10.165235 seconds and reproduced the same numerical results and zero guard changes.

### Normal decode: position boundaries

Command shape:

The same command covers the normal-path boundary case and the unsupported
`threshold=1` boundary; `reproduce.py` dispatches all three bounded cases.

The case used sequence lengths `[4, 5, 6]`, four MHA heads, head dimension 64, one
split, and `xai_temperature_len=4`.

Raw results:

- Query positions `[3, 4, 5]`.
- Expected scales `[1.0, 1.0, 1.160964047443681]`.
- Maximum LSE error: `0.0061299800872802734`.
- Maximum split-output error: `0.013254553079605103`.
- Split-output `allclose(atol=0.02, rtol=0.02)`: true.
- All outputs finite: true.
- One-call elapsed time including Triton JIT: `0.9731436157599092` seconds.

### Unsupported threshold boundary

`xai_temperature_len=1` is not a supported logarithmic boundary because
`log2(1) == 0`. The independent reference cannot define the intended scale, and the
normal decode kernel produced nonfinite results for query position 2:

- Nonfinite LSE entries: 4 of 4.
- Nonfinite split entries: 256 of 256.
- Reference status: unsupported, `float division by zero`.

No code change was made for this boundary. The supported neighboring control was
`xai_temperature_len=4`, which passed as recorded above.

## Conclusion

No mismatch was demonstrated in the supported grouped or normal decode paths. The
existing masks preserve padded-row independence for both exposed buffers, and the
position boundary at the threshold uses scale 1.0 while the next position uses the
expected logarithmic scale. Because the task requires code changes only after a
demonstrated mismatch, this PR intentionally contains findings only.
