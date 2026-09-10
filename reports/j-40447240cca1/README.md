# GPU hidden-state handoff validation

## Scope

This validates the overlap relay used between target and draft stages:
`FutureMap.stash` converts and stores `RelayPayload.hidden_states`, and
`gather_spec_extras` gathers the stored rows for the next draft input. It does
not test hidden-state index compaction or graph-tier control.

The new GPU test covers:

- all six conversions among BF16, FP16, and FP32;
- a non-contiguous `[rows, draft_tokens, hidden_dim]` source view;
- the persistent buffer's contiguous `[pool, draft_tokens, hidden_dim]` shape
  and stride contract;
- a row-0 padded batch tail;
- an independent explicit-conversion and advanced-indexing reference;
- exact gathered dtype, shape, strides, and values.

## Environment

- GPU: one AMD Instinct MI300X (`gfx942`, 304 CUs)
- Image: `amdpilotv2/open-job-mi300x:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Installed source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed Python package: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Installed native package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Delivery checkout: `/job/sglang`, branch `amdpilot/j-40447240cca1`, base
  commit `0084030`

## Installed-source baseline

The first GPU execution used the existing independent-reference gather test:

```bash
PYTHONPATH=/sgl-workspace/sglang/python \
  /opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_gather_spec_extras.py
```

Result:

```text
7 passed, 19 subtests passed
pytest duration: 29.71 s
first-GPU-process wall time: 32.667 s
```

Timing is one bounded process wall-clock measurement using UTC epoch
milliseconds before and after pytest. There is no warmup, burn loop, sleep, or
repeated work. The first attempt to use `/usr/bin/time` failed because that
binary is absent; no GPU work ran in that attempt. The complete early record is
in `baseline-first.json` outside this repository checkout.

## Checkout validation

New relay/conversion test:

```bash
PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/unit/managers/test_overlap_hidden_state_handoff.py
```

Result:

```text
1 passed, 9 subtests passed
pytest duration: 12.93 s
process wall time: 15.789 s
```

Existing independent-reference gather test:

```bash
PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/kernels/ops/speculative/test_gather_spec_extras.py
```

Result:

```text
7 passed, 19 subtests passed
pytest duration: 13.08 s
process wall time: 16.320 s
```

Combined checkout run:

```text
8 passed, 28 subtests passed
pytest duration: 12.93 s
process wall time: 16.044 s
```

## Quantization caveat

A bounded probe showed that the low-level Triton gather can copy raw
`torch.float8_e4m3fn` bits exactly on this stack. That is not evidence of a
supported quantized hidden-state handoff: `RelayPayload` has no hidden-state
scale field, and no speculative hidden-state scale is relayed by this path.
Accordingly, the semantic test covers BF16/FP16/FP32 only and does not claim
FP8 support or scale preservation.

## Issue context

Upstream issue `sgl-project/sglang` number 30734 lists S4 hidden/KV fast paths
as not started. Reviewed upstream PRs 31047, 31260, 31457, 32186, and 32374 do
not already provide this focused FutureMap hidden-state conversion contract;
PR 32374 changes relay scheduling but not hidden-state dtype/shape handling.
