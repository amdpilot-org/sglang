# gfx942 EP scatter inverse-gather investigation: j-d47fe82d8395

## Scope and result

This follow-up tests one local property on one assigned AMD Instinct MI300X
(`gfx942`): after `ep_scatter`, a valid inverse `ep_gather` restores the original
finite source values when expert groups are padded to 128-token blocks.

The property passed exactly on the installed source and on the mirror checkout at
`main`. No kernel mismatch was demonstrated, so no kernel or production-test code
was changed. This delivery adds investigation evidence only.

This is deliberately separate from amdpilot-org/sglang issue 212 and mirror pull
request 241. PR 241 already fixes the `_fwd_kernel_ep_scatter_1` start-offset race
and tests exclusive starts, expert IDs, `-1` padding, and guard storage. That
fulfilled race-boundary scope was not repeated here.

## Upstream and prior evidence

- Upstream context: sgl-project/sglang issue 31929.
- Candidate fix: sgl-project/sglang pull request 31930, commit
  `3612e7a1ea60324e9425e7a3d41ed9f2414cad78`.
- Existing mirror delivery: amdpilot-org/sglang pull request 241, commit
  `f61b86c7ccc92e8facb7823d00a9c1116394b353`, currently open and not merged.
- PR 241 changes `cur_expert_start` to a register reduction over
  `tokens_per_expert` instead of reloading the just-written global value.
- The candidate commit was reviewed but was not retested in this follow-up; PR 241
  already tested that identical race-fix scope on this architecture. Repeating it
  would duplicate fulfilled work.
- No upstream issue, pull request, or comment was posted or modified.

## Environment

- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-provided local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- `/.dockerenv` exists. No `docker`, `crictl`, or `ctr` utility is available in the
  job, so the local image ID could not be independently resolved from container
  metadata.
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 compute units,
  serial `692440003949`, UUID
  `35383633-3433-6633-3832-613639663164`.
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`,
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Torch HIP library:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Triton: `3.7.0`,
  `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Job-private Triton cache: `/tmp/sglang-cache-j-d47fe82d8395`

The qualified Torch/ROCm stack was preserved. No model weights, framework stack,
or node-wide state were downloaded or changed.

## Sources

- Installed-source commit:
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed Python source:
  `/sgl-workspace/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py`
- Imported installed package:
  `/sgl-workspace/sglang/python/sglang/__init__.py`
- Mirror base and tested checkout commit:
  `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Delivery checkout:
  `/job/sglang`
- Delivery Python source:
  `/job/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py`

The installed-source result is evidence only for that installed source. It is not
proof for the later mirror checkout.

## Installed-source first baseline

The first successful installed-source GPU execution used:

- four experts;
- hidden size 128;
- top-1 routing;
- 256 source tokens;
- padded counts `[128, 256, 128, 128]`;
- valid counts `[127, 1, 128, 0]`;
- 640 padded rows total;
- finite adversarial `float16` values, including zero, signed zero, positive and
  negative subnormals, positive and negative finite maximum values, and
  deterministic random values.

The independent reference used:

- exclusive padded starts from `torch.cumsum`;
- independently constructed expert IDs and `-1` padding;
- `output_tensor[output_index]` as a Torch inverse reference;
- exact `torch.equal` comparisons for integer indices and restored `float16`
  values.

All installed-source checks passed:

- post-scatter cursor equals exclusive start plus valid count;
- `m_indices` matches the independent reference;
- `output_index` is assigned, unique, and in bounds;
- `m_indices[output_index]` matches the routed expert;
- scattered source rows are exact;
- padding rows retain their sentinel value;
- the Torch inverse reference is exact;
- `ep_gather` restores every source value exactly.

The first successful GPU execution completed `5.07073454` seconds after process
start. Timing used one CUDA-event pair around 30 scatter-plus-gather launch pairs
after three warmups, giving `0.0849306344985962` ms per launch pair.

The complete installed-source record is in
`reports/j-d47fe82d8395/baseline-first.json`.

## Mirror checkout cases

The mirror checkout was tested at `main` commit
`0084030179bfba86bfeb6d43f7997d4076329d2c`. The input again used finite
adversarial `float16` values and deterministic random seeds.

| Case | Experts | Hidden | Top-k | Padded counts | Valid counts | Mean ms/pair |
|---|---:|---:|---:|---|---|---:|
| `top1_mixed_padded_groups` | 4 | 128 | 1 | `[128, 256, 128, 128]` | `[127, 1, 128, 0]` | `0.08648356596628824` |
| `top2_one_hot_inverse` | 8 | 128 | 2 | `[128] * 8` | `[0, 1, 127, 128, 0, 0, 0, 0]` | `0.07924156983693441` |
| `top2_half_weight_inverse` | 8 | 128 | 2 | `[128] * 8` | `[0, 1, 127, 128, 0, 0, 0, 0]` | `0.07727166811625162` |
| `top1_zero_padded_expert` | 5 | 128 | 1 | `[0, 128, 128, 0, 128]` | `[0, 128, 0, 0, 127]` | `0.07793320020039876` |
| `top1_wide_hidden` | 4 | 1024 | 1 | `[128] * 4` | `[128, 0, 1, 127]` | `0.1486056645711263` |

The top-2 one-hot case uses weights `[1, 0]`. The top-2 half-weight case uses
`[0.5, 0.5]`; its independent reference performs the weighted sum in `float32`
and then casts to `float16`, matching the gather kernel's accumulation contract.

Every case passed all exact gates:

- post-scatter cursor equals exclusive start plus valid count;
- `m_indices` exactly matches the independently constructed expert and padding
  reference;
- `output_index` is fully assigned, unique, and in bounds;
- `m_indices[output_index]` exactly matches the routed expert;
- every scattered source row exactly equals its source;
- every padding row retains its sentinel;
- the independent Torch inverse reference exactly equals the source;
- `ep_gather` exactly restores the source;
- all source and gathered values are finite;
- gather mismatch count is zero;
- maximum absolute error is `0.0`.

The first mirror-checkout GPU execution completed `5.693776553` seconds after
process start. Each case used one CUDA-event pair around 30 scatter-plus-gather
launch pairs after three warmups. No unbounded loop, sleep loop, synthetic burn, or
repeated work solely to occupy the GPU was used.

The complete raw checkout record is in
`reports/j-d47fe82d8395/mirror-roundtrip-gfx942-results.json`.

## Reproduction

From this repository root on one assigned MI300X:

```bash
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-d47fe82d8395
start_ns=$(date +%s%N)
PYTHONPATH="$PWD/python" /opt/venv/bin/python \
  reports/j-d47fe82d8395/reproduce.py \
  --start-ns "$start_ns" \
  --output /tmp/mirror-roundtrip-gfx942-results.json
```

The script uses the checkout under `$PWD/python`, emits the five-case JSON record,
and exits nonzero if any exact gate fails.

## Honest limitations

- This is a local kernel-contract result only. It is not distributed
  expert-parallel, DeepEP transport, or full-model proof.
- The deterministic cases did not directly reproduce the intra-CTA store/load race
  reported in issue 31929. A pass here does not disprove that latent hazard.
- No multi-GPU, long-running stress, race-scheduling adversarial loop, or illegal
  memory access reproduction was attempted.
- The upstream candidate commit was not retested because PR 241 already covered
  that identical race-fix scope.
- No full model weights were downloaded.
- The container lacks Docker/CRI tooling, so the operator-provided image ID could
  not be independently resolved.

## Delivery

- Branch: `amdpilot/j-d47fe82d8395`
- Base: mirror `main` at
  `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Kernel changes: none
- Production-test changes: none
- Findings and raw records: `reports/j-d47fe82d8395/`
