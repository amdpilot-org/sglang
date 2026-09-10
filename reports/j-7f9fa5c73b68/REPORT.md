# gfx942 non-contiguous packed-versus-unpacked gather report

## Scope and result

This follow-up tests one additional execution representation on one assigned AMD
Instinct MI300X (`gfx942`): non-contiguous, row-strided `torch.index_select`
source and output views, compared against an unpacked contiguous control.

All four bounded cases passed exactly against independent CPU references and
the unpacked control. The source and output sentinel rows remained unchanged,
the preallocated output address and dtype were preserved, and the profiler
recorded the same native `vectorized_gather_kernel` dispatch for both packed and
unpacked representations. Unsupported dtype mismatch and source/output aliasing
variants failed clearly before dispatch.

No mismatch was demonstrated, so no production or test code was changed. This
delivery adds investigation evidence only.

## Context

- Read-only upstream context: sgl-project/sglang issue 37936.
- Prior mirror context: amdpilot-org/sglang issue 219 and PR 351.
- PR 351 already covered contiguous alternating output buffers and explicitly
  left non-contiguous outputs untested.
- This report extends only that uncovered non-contiguous representation case.
- No upstream issue, PR, or comment was posted or changed.

## Environment

- Campaign: `repo-e2e-20260909`
- Job: `j-7f9fa5c73b68`
- Required image:
  `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, unique ID
  `0xb5c590cf4c10631d`, serial `692440004306`, node ID 2.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Persistent checkout: `/job/sglang`
- Tested commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Installed-source baseline

The first GPU execution used the preinstalled interpreter and Torch stack before
checkout modifications:

```bash
JOB_WORKDIR=/job /opt/venv/bin/python /job/baseline_first.py
```

The installed-source baseline is recorded in `/job/baseline-first.json` and is
not proof for later checkout changes. It used contiguous `bfloat16`
`torch.index_select(source, 0, indices, out=output)` with:

- independent CPU `torch.index_select` reference,
- sentinel-protected source and output storage,
- actual native dispatch recorded with `torch.profiler`,
- three warmups and ten CUDA-event timed calls per case,
- six bounded shape cases,
- first GPU execution elapsed time `1.6139473039656878` seconds.

All six baseline cases matched the CPU reference exactly, preserved sentinels,
and dispatched:

```text
void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)
```

## Persistent-checkout experiment

Reproduction:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-7f9fa5c73b68/reproduce.py
```

The experiment uses `bfloat16` source and output views with row stride twice the
column count, placed inside larger sentinel-protected storage. It compares each
packed result with:

1. an independent CPU `torch.index_select` reference, and
2. an unpacked contiguous GPU control using the same source values.

### Numerical and contract gates

All four cases passed:

- exact CPU-reference match,
- exact unpacked-control match,
- source and output sentinel rows unchanged,
- preallocated output address unchanged,
- output dtype unchanged,
- native `vectorized_gather_kernel` observed for packed and unpacked calls.

Unsupported variants were not forced through:

- dtype mismatch raised `ValueError` before dispatch,
- source/output aliasing raised `ValueError` before dispatch.

### Bounded timing matrix

Each case used three warmups and ten CUDA-event timed calls. Median timings:

| rows | cols | packed median (ms) |
|---:|---:|---:|
| 64 | 64 | 0.0453450009226799 |
| 256 | 64 | 0.02561900019645691 |
| 64 | 256 | 0.021208999678492546 |
| 256 | 256 | 0.026861999183893204 |

Raw timings, native event names, strides, sentinel checks, address checks, and
dtype checks are in:

```text
reports/j-7f9fa5c73b68/noncontiguous-index-select-results.json
```

## Boundaries

- No full model weights, SGLang server, EAGLE model, CUDA graph replay, NCCL
  collective, or grammar request were used.
- No unbounded loops, sleep loops, synthetic burn, or repeated work were used.
- No node-wide state was modified.
- Validation covers one MI300X only, not the reported 4x V100 topology.
- This is a native PyTorch indexing operation, not an SGLang Triton kernel.
- No production or test code was changed because no mismatch was demonstrated.
