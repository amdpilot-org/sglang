# Reduced DSpark pipeline study on one MI300X

## Scope and boundary

This is a prefill-sized, synthetic reduced-block study. It is distinct from
standalone operator probes and makes no GLM weight, full-model, or distributed
throughput claim.

The reduced pipeline uses these production SGLang primitives:

- target hidden-state projection through `project_through_lm_head`;
- draft hidden-state selection through `select_draft_hidden_without_anchor`;
- fused commit-KV projection through `CommitKvProj.triton`;
- greedy accept through `accept_greedy_triton`;
- accept finalization through `FinalizeAcceptLens.triton`;
- output token commit through `BuildOutTokens.triton`.

The model-specific GLM/DSpark target-hidden projector constructor was not
instantiated. That boundary requires a full model configuration and checkpoint.
The study therefore starts from synthetic target hidden states and labels this
as the model-specific constructor boundary.

## Environment

- GPU: one AMD Instinct MI300X, gfx942 capability `(9, 4)`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- HIP: `7.2.26015-fc0010cf6a`.
- Triton: `3.7.0`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Checkout SGLang: `/job/sglang/python/sglang/__init__.py`.
- Observed native AIter import path: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Installed-source baseline

Before cloning or editing, the preinstalled source was exercised with:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/spec/dspark/test_dspark_kernel_parity.py
```

The installed source path was
`/sgl-workspace/sglang/python/sglang/__init__.py`. The test compares production
Triton kernels with independent Torch references. It passed `1` test and `22`
subtests in `47.26s` as measured by pytest; surrounding wall time was
`51.062322s`. The return code was zero.

This is an installed-source baseline only and is not proof for later checkout
changes. The first attempt used `/usr/bin/time`, which was absent, so no GPU
work launched; timing was then repeated with nanosecond shell timestamps.

## Dimensions and dtypes

All cases used `gamma=5`, `verify_num_draft_tokens=6`, target hidden width
`4096`, draft hidden width `1024`, vocabulary `8192`, three commit-KV stages,
and commit head dimension `576`.

Hidden states, weights, and logits used `torch.bfloat16`. Token IDs used
`torch.int64`; verify lengths used `torch.int32`; prefix lengths used
`torch.int64`.

## Numerical gates

Every case used the same gates:

- target projection maximum absolute error `<= 0.02`;
- commit-KV projection maximum absolute error `<= 0.02`;
- draft hidden selection exactly equal to its independent reference;
- greedy accept outputs exactly equal to the Torch reference;
- finalized accept lengths exactly equal to the Torch reference;
- committed output tokens exactly equal to the Torch reference;
- synthetic weights below 4GB;
- peak live allocations below 48GB.

All six cases passed all gates.

## Results

Timing uses CUDA events around one complete reduced forward. The first cold
event includes Triton compilation; later case-specific cold events are first
calls with already-compiled kernels. Warm throughput is the median of five
bounded real forwards. No artificial burn or unbounded loop was used.

| Prefill tokens | Spatial batch | Target max error | Commit max error | Accepted tokens | Cold event (s) | Warm median (s) | Prefill tok/s | Pipeline tok/s | Peak allocation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 0.00195360 | 0.00000000 | 16 | 1.656625 | 0.000608444 | 6,731,926 | 6,797,668 | 556,856,320 B |
| 8192 | 16 | 0.00195360 | 0.00000000 | 39 | 0.004861 | 0.001009771 | 8,112,731 | 8,191,956 | 664,917,504 B |
| 16384 | 32 | 0.00195432 | 0.00097656 | 66 | 0.002316 | 0.001825333 | 8,975,896 | 9,063,552 | 873,960,448 B |
| 24576 | 64 | 0.00195408 | 0.00097656 | 150 | 0.003315 | 0.002707369 | 9,077,447 | 9,195,643 | 1,188,940,288 B |
| 32768 | 128 | 0.00195402 | 0.00195313 | 311 | 0.004172 | 0.003693846 | 8,870,971 | 9,044,232 | 1,538,884,096 B |
| 32768 | 256 | 0.00195396 | 0.00000000 | 629 | 0.004230 | 0.003879274 | 8,446,942 | 8,776,900 | 1,562,051,072 B |

Per-case synthetic weight allocation was `70,647,808` bytes. The complete
study took `5.177702s` after environment import. Raw values, exact dimensions,
all five warm samples, and resource counters are in `results.json`.

## Reproduction

From the repository root:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-399c740a51f1/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-399c740a51f1/inductor
export HF_HOME=/tmp/sglang-cache-j-399c740a51f1/hf
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
/opt/venv/bin/python reports/j-399c740a51f1/reduced_pipeline.py \
  --warm-forwards 5 \
  --output reports/j-399c740a51f1/results.json
```

The command is bounded to six cases, two cold forwards per case, and five warm
forwards per case. The process-level wall limit for the investigation was 7200
seconds.

## Harness note

An intermediate run failed the commit projection gate after synthetic commit
modules were garbage collected and a later module reused an `id()` key in
`CommitKvProj`'s stacked-weight cache. Production model modules remain alive,
so the final harness retains the synthetic modules for the intended primitive
lifecycle. This observation is recorded for reproducibility and is not claimed
as a production defect.

## Left undone

- No GLM checkpoint or full model constructor was used.
- No multi-GPU, TP, distributed, or end-to-end serving claim is made.
- The reduced pipeline does not include attention, KV-cache pool writes, CUDA/HIP graph replay, or scheduler overlap.
