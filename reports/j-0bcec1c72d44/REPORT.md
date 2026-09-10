# KDA gfx942 synthetic metadata investigation

## Result

The bounded synthetic investigation passed on one assigned AMD Instinct MI300X (`gfx942`). There were no child-process timeouts and no observed hang. This result does **not** prove absence of the reported intermittent MI350X TP8 hang.

The metadata-behavior candidate is already present in the tested mirror commit:

- Tested mirror commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Candidate: `KDAAttnBackend.needs_cpu_seq_lens = False`
- Source: `python/sglang/srt/layers/attention/linear/kda_backend.py:400`
- Current value at runtime: `False`

This mirrors the metadata-only one-line behavior confirmed in read-only upstream issue 33846 and included in read-only upstream PR 32219. No duplicate product change was made. Upstream PR 33810 was also reviewed; issue 33846 reports that its GDN `-1` sentinel fix is real but did not resolve the KDA D2H stall.

## Scope and upstream context

- Read-only context: `sgl-project/sglang` issue 33846, PR 32219, and PR 33810.
- The issue reports an intermittent MI350X TP8 KDA prefill hang at `chunk_kda_fwd -> prepare_chunk_indices -> Tensor.tolist() -> hipMemcpy D2H`.
- This investigation exercised the current `prepare_chunk_indices` and a small supported Triton `chunk_kda`/`chunk_kda_fwd` path on gfx942 with synthetic varlen metadata.
- It did not run Kimi-K3, DSPARK, TP8, MI350X, a full model, or node-level changes, and did not download model weights.
- No upstream issue, PR, or comment was posted or modified.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, GUID `61795`, 304 CUs, about 206 GB reported memory.
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Triton: `3.7.0`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Tested SGLang source: `/job/j-0bcec1c72d44/sglang/python/sglang`
- `prepare_chunk_indices`: `/job/j-0bcec1c72d44/sglang/python/sglang/kernels/ops/attention/fla/index.py`
- `chunk_kda`: `/job/j-0bcec1c72d44/sglang/python/sglang/kernels/ops/attention/fla/kda.py`
- KDA backend: `/job/j-0bcec1c72d44/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py`
- Native `sgl_kernel` package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel` (`0.4.6.post1`); the exercised KDA path itself is Triton-based.
- Preinstalled source at `/sgl-workspace/sglang` was environment context only; `PYTHONPATH` selected the tested clone.
- Job-private cache: `/tmp/sglang-cache-j-0bcec1c72d44`

## Method

The probe is `reports/j-0bcec1c72d44/kda_gfx942_probe.py`. It uses:

- Single-request metadata: sequence lengths `[1]`.
- Multi-request metadata: sequence lengths `[1, 65]`.
- Ten child-process trials per case, bounded by a 45-second per-trial timeout.
- Stage markers before and after `prepare_chunk_indices`, `chunk_kda`, and stream synchronization.
- An independent Python index-map reference and an independent sequential PyTorch KDA reference.
- A state-index pool with a preserved `-1` padding sentinel and an untouched sentinel state-pool row.
- `torch.profiler` for one representative D2H observation.

The numerical gates were unchanged throughout:

- Output maximum absolute error must be at most `0.01`.
- Output relative L2 error must be at most `0.02`.
- Final state maximum absolute error must be at most `0.01`.
- Index maps must be exactly equal.
- The `-1` sentinel and its state-pool row must remain unchanged.

## Results

Raw per-trial output, stage markers, timings, return codes, and profiler data are in `reports/j-0bcec1c72d44/kda_gfx942_results.json`.

| Case | Trials | Timeouts | Index map | Sentinel | Numerical gates |
|---|---:|---:|---|---|---|
| single request | 10 | 0 | exact | preserved | passed |
| multi request | 10 | 0 | exact | preserved | passed |

Observed ranges:

- Single-request output maximum absolute error: `9.324576240032911e-06` to `0.0001843869686126709`.
- Single-request output relative L2 error: `0.0027016871608793736` to `0.007730374578386545`.
- Single-request state maximum absolute error: `0.00013900920748710632` to `0.0014264732599258423`.
- Multi-request output maximum absolute error: `0.00014023296535015106` to `0.00028810277581214905`.
- Multi-request output relative L2 error: `0.003505828790368697` to `0.004647239111363888`.
- Multi-request state maximum absolute error: `0.0002323240041732788` to `0.0014264732599258423`.

No trial reached a timeout. Every trial reached `child_complete`; therefore the last recorded point was successful completion, not a hang.

The representative profiler run used the default stream `<torch.cuda.Stream device=cuda:0 cuda_stream=0x0>` and observed:

- `Memcpy DtoH (Device -> Host)`
- `Memcpy HtoD (Host -> Device)`
- `hipMemcpyWithStream`

This confirms that `prepare_chunk_indices` still performs a D2H transfer on the current stream. The already-present backend candidate avoids forcing the separate per-step `seq_lens` CPU mirror D2H described by the issue; it does not remove the D2H inside `prepare_chunk_indices`.

## Reproduction

From the repository root:

```bash
export PYTHONPATH="$PWD/python"
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-0bcec1c72d44/triton
export TORCH_HOME=/tmp/sglang-cache-j-0bcec1c72d44/torch
export XDG_CACHE_HOME=/tmp/sglang-cache-j-0bcec1c72d44/xdg
/opt/venv/bin/python reports/j-0bcec1c72d44/kda_gfx942_probe.py \
  --output reports/j-0bcec1c72d44/kda_gfx942_results.json
```

The command runs 20 bounded child trials plus one profiler child. It exits nonzero if any trial fails, times out, or violates a gate.

## Limitations

- This is a small synthetic gfx942 test, not the reported MI350X gfx950 TP8 workload.
- Ten trials per synthetic case cannot establish the absence of a low-frequency intermittent hang.
- The profiler records the presence of D2H activity, not a reproduction of the reported permanent `hsa` signal wait.
- The independent output reference is sequential PyTorch and uses the same input tensors, but it is not a separate production kernel implementation.
- No full-model, TP8, node-level, or MI350X validation was attempted.
