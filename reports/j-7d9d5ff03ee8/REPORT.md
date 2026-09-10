# gfx942 alternating-output-buffer gather report

## Scope and result

This follow-up tests one additional property on one assigned AMD Instinct MI300X
(`gfx942`): valid `torch.index_select` row permutations and independently
derived inverse permutations, written through `out=` into two alternating
preallocated output buffers.

All five bounded cases passed exactly against independent CPU references. The
inverse restored the original finite source matrix byte-for-byte, both output
buffers were preserved as the call targets, and no extra CUDA allocation occurred.
No mismatch was demonstrated, so no production or test code was changed. This
delivery adds investigation evidence only.

This scope is distinct from the already fulfilled evidence reviewed below:

- amdpilot-org/sglang issue 219 and PR 286 cover two-stream boundary gathers and
  the ROCm out-of-range abort boundary.
- PR 339 covers in-place KV-row inverse permutations.
- PR 338 covers EP scatter followed by inverse EP gather.
- PR 288 covers bounded in-place KV cycle permutations.

No identical alternating `index_select(out=)` permutation scope was found.

## Environment

- Campaign: `repo-e2e-20260909`
- Job: `j-7d9d5ff03ee8`
- Required image:
  `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- No container image API was available for independent in-container image-ID
  verification.
- GPU: one AMD Instinct MI300X, `gfx942`, unique ID
  `0xa09b4a46354a21d9`, serial `692412003101`, node ID 9.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP
  `7.2.26015-fc0010cf6a`.
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Torch HIP library:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Installed sglang source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Persistent checkout: `/job/sglang`, base commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`

The qualified Torch/ROCm stack was preserved. No model weights, framework stack,
or node-wide state were downloaded or changed.

## Installed-source baseline

The first successful installed-source GPU execution used the existing relevant
small `index_select` test:

```bash
TIMEFORMAT='pytest_elapsed=%3R'
{ time timeout 120 /opt/venv/bin/python -m pytest \
  /sgl-workspace/sglang/test/registered/kernels/ops/diffusion/test_layout.py \
  -k small_c2 --durations=5 -q; } 2>&1 | tee /tmp/baseline-pytest.log
```

Result:

- `4 passed, 4 skipped, 91 deselected in 36.88s`.
- Shell elapsed time: `41.110s`.
- Slowest relevant call: `1.89s`.
- Independent CPU `index_select` comparison: exact, maximum absolute difference
  `0.0`.
- First separately timed GPU `index_select` call: `0.003975427709519863s`.
- Three bounded repeat trials: `0.00020851660519838333s`,
  `0.000060083344587874298s`, and `0.00004845950752496719s`.

The complete record is in `/job/baseline-first.json`. It is installed-source
evidence only and is not proof for the later persistent checkout.

## Property contract

The tested operation contract was:

```text
forward_buffer[i, :] = source[permutation[i], :]
inverse_buffer[i, :] = forward_buffer[inverse[i], :]
inverse[permutation[i]] = i
```

`torch.index_select(source, 0, permutation, out=forward_buffer)` was used for
the forward direction. The inverse was derived independently by scatter assignment
and cross-checked against `torch.argsort(permutation)`.

The finite adversarial `float32` source was `4096 x 4096` and included:

- signed positive zero and negative zero;
- the minimum positive subnormal;
- the minimum positive normal and its negative;
- positive and negative finite maximum values;
- deterministic seeded random finite values;
- a deterministic finite value pattern across the remaining matrix.

The five bounded permutation cases were:

1. identity;
2. reverse;
3. right rotation by one;
4. left rotation by one;
5. seeded random permutation.

Forward and inverse calls alternated between two preallocated output buffers
`a` and `b`. Every case used only valid indices in `[0, 4095]`, including both
boundary values.

## Raw results

The complete raw record is in
`reports/j-7d9d5ff03ee8/alternating-index-select-results.json`.

| Case | Forward buffer | Inverse buffer | Forward event (ms) | Inverse event (ms) | Exact |
|---|---|---|---:|---:|---|
| identity | a | b | 2.604836 | 0.089967 | yes |
| reverse | b | a | 0.145414 | 0.063947 | yes |
| rotate right one | a | b | 0.091931 | 0.051559 | yes |
| rotate left one | b | a | 0.126170 | 0.060379 | yes |
| seeded random | a | b | 0.098146 | 0.048392 | yes |

All cases passed:

- permutation and inverse indices were valid and unique;
- inverse matched the independent `argsort` reference;
- forward output matched the independent CPU `index_select` reference;
- inverse output matched the independent CPU inverse reference;
- inverse output restored the original source exactly;
- source remained finite;
- returned tensor data pointers matched the intended output buffers;
- allocated and peak CUDA memory remained unchanged across each case.

The profiler control observed:

```text
void at::native::vectorized_gather_kernel<16, long>(...)
```

Timing used one CUDA event pair around each `index_select` call and one
wall-clock measurement per case. The complete property run, including the
single profiler control, completed in `5.923s`.

## Reproduction

```bash
timeout 180 env PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-7d9d5ff03ee8/reproduce.py
```

The script performs five cases, two `index_select` calls per case, and one
profiler control. It does not use an unbounded loop, sleep loop, repeated
work to occupy the GPU, or model weights.

## Upstream context

- Upstream context was read only from sgl-project/sglang issue 37936.
- The issue reports a V100 EAGLE/MTP `vectorized_gather_kernel` out-of-range
  assertion during concurrent constrained tool usage.
- At investigation time it had one comment from Hao-tian-Zheng requesting focused
  logs and describing an isolated A100 minimization.
- An exact upstream search for `37936` and `vectorized_gather_kernel` found no
  linked fix PR.
- No upstream issue, pull request, or comment was posted or modified.

## Limitations and unsupported boundaries

- Validation covers one MI300X (`gfx942`) only, not the reported 4x V100 topology.
- No full SGLang server, EAGLE model, CUDA graph replay, NCCL collective, grammar
  request, or model weights were run.
- This property exercises the native contiguous `index_select(out=)` path, not
  SGLang scheduler state or speculative batch mutation.
- Invalid indices such as `-1` and `4096` were intentionally not repeated; they
  are outside the valid-permutation scope and PR 286 already records the ROCm
  fatal abort boundary.
- Index/output aliasing was not tested because the contract uses two distinct
  preallocated buffers.
- Non-contiguous output layouts and dtypes other than contiguous `float32` were
  not tested.
- No unsupported boundary was encountered for the tested finite matrix and valid
  permutations.
