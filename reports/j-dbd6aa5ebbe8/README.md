# DFlash `kv_indices` storage-representation experiment

## Scope

- Upstream context: `sgl-project/sglang` issue `37553`.
- Prior mirror context: `amdpilot-org/sglang` issue `217`.
- This is a bounded follow-up to the earlier custom-mask investigation. It does
  not repeat the original custom-mask growth trigger.
- The uncovered case is the DFlash `kv_indices` output, which has both a fresh
  allocation path and a documented CUDA-graph buffer-reuse path.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one assigned AMD Instinct MI300X, architecture `gfx942`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## First installed-source baseline

`/job/baseline-first.json` records the required installed-source baseline. It is
explicitly not proof for later checkout changes.

- Command: `cd /sgl-workspace/sglang && /opt/venv/bin/python /sgl-workspace/sglang/test/registered/attention/test_triton_attention_kernels.py TestTritonAttention.test_decode_attention -v`
- Result: `1 passed`, `0 failed`
- First GPU execution elapsed time: `26` seconds overall; unittest reported
  `10.682` seconds for the four bounded decode configurations.
- Reference comparison: the existing test compares Triton decode attention
  against an independent float32 stable-softmax Python reference.

## Experiment

`verify_kv_indices_storage.py` compares two execution representations for
`DFlashVerifyInput.generate_attn_arg_prefill`:

- **Fresh output storage**: `kv_indices_buf=None`, so the operation allocates a
  new `int32` output for each call.
- **Documented safe buffer reuse**: a persistent `int32` CUDA-graph buffer is
  passed as `kv_indices_buf`, preserving a stable device address across calls.

The finite batch matrix is `[1, 8, 1, 16, 1, 32, 1]`, with
`draft_token_num=8`, `paged_kernel_len=10`, and three timed repeats per case.
Every case independently compares:

- `kv_indices`
- `cum_kv_seq_len`
- `qo_indptr`

The reuse buffer is sentinel-protected with `-1` outside the current output
prefix. The experiment verifies that only the expected prefix is overwritten and
that the sentinel tail remains intact after each call.

## Native dispatch

The actual native dispatch observed with `torch.profiler` is:

- Kernel: `create_flashinfer_kv_indices_triton`
- Source: `python/sglang/kernels/ops/kvcache/kv_indices.py`

## Unsupported variants

The experiment explicitly rejects the following unsupported buffer variants
before native dispatch:

- Wrong dtype: `torch.int64` instead of `torch.int32`
- Too small: fewer elements than the current output requires
- Aliasing: a buffer that overlaps `req_to_token`

Each rejection is a clear `ValueError`; no unsupported variant is forced through
the native kernel.

The production path itself does not currently guard these variants. If the
experiment’s validation layer is bypassed, exploratory calls showed that a
wrong-dtype buffer is silently accepted and an aliased buffer can overwrite
`req_to_token`. This is recorded as negative evidence; the experiment does not
force either unsupported variant through.

## Raw results

The complete raw JSON, including every timing sample, output address, sentinel
check, and reference comparison, is in:

`reports/j-dbd6aa5ebbe8/kv-indices-storage.json`

All `kv_indices`, `cum_kv_seq_len`, and `qo_indptr` comparisons passed for both
representations across every batch size. Fresh output addresses were not stable,
while the reuse buffer address remained identical across all calls. The
sentinel tail remained intact in every reuse case.

## Reproduction

```bash
export PYTHONPATH="$PWD/python"
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-dbd6aa5ebbe8/triton
/opt/venv/bin/python reports/j-dbd6aa5ebbe8/verify_kv_indices_storage.py \
  --output reports/j-dbd6aa5ebbe8/kv-indices-storage.json
```

## Limitations

- Only one assigned MI300X (`gfx942`) GPU was used.
- No CUDA graph capture or replay was executed; the experiment verifies the
  documented static-address property by checking that the reuse buffer’s device
  address remains stable across distinct batches.
- No full model weights or alternate framework stack were downloaded.
- No node-wide state was modified.
- The experiment does not claim to fix the already-merged upstream custom-mask
  change or the open mirror candidate in PR 367.
