# Scheduler import-cost investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/10492

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3446

## Result

The reported behavior is reproducible on the prepared ROCm stack, but this
change intentionally does not duplicate fixes already under upstream review.
Every measurement used a new Python process and the prescribed interpreter.

The first completed import against the private, initially empty cache took
28.612 s. It built and loaded AIter's `module_aiter_core.so`; the retained
Ninja log records 10.704 s compiling `aiter_core_pybind.cuda.o` and 0.088 s
linking. A subsequent interrupted sampling attempt also built
`module_rmsnorm_quant.so`, so that attempt is excluded from the completed-run
timings. The generated `build.ninja` names `/opt/rocm-7.2.4/bin/hipcc` and
`--offload-arch=native`; ROCm SMI reported the visible device as an AMD Instinct
MI350X (`gfx950`). These are current AMD-host observations, not a reuse of the
issue's old H100 number.

With those private caches warm, five completed clean-process imports took
13.217, 13.143, 12.987, 12.868, and 13.217 s (mean 13.086 s, range 0.349 s).
"Warm" here means filesystem/compiler caches were already populated, not that
Python modules were reused: each sample was a separate process. The shared
host and concurrent load remain uncontrolled sources of variability.

`python -X importtime` attributed 11.059 s cumulative to Scheduler. The largest
relevant chain was:

`scheduler` -> `ModelConfig` -> `layers.quantization` -> `mxfp4` -> `aiter`

The profile attributed 4.773 s cumulative to `aiter`, including 2.434 s self
time in `aiter.ops.flydsl.gemm_kernels`. It also showed `openai`, `fastapi`, and
`torchvision` loaded by the Scheduler import. A direct module check confirmed
all four (`aiter`, `openai`, `fastapi`, and `torchvision`) are present in
`sys.modules` after importing `Scheduler`.

## Source assessment and related work

The eager quantization registry in
`python/sglang/srt/layers/quantization/__init__.py` imports every quantization
implementation in order to construct class-valued dictionaries. On this AMD
stack that imports MXFP4 and AIter even though no model or quantization method
has been selected, and it can cause native compilation in an otherwise simple
Scheduler import.

The exact targeted correction is already implemented in upstream PR #37602,
which changes this registry to string specifications and lazily resolves only
the selected class. Upstream PR #38005 adds import-surface regression coverage
for the public quantization exports. Independently, upstream PR #38177 defers
the OpenAI SDK, FastAPI/Uvicorn, and torchvision branches measured here. All
three PRs were open when this investigation was performed. Reimplementing any
of them in the mirror would duplicate current related work, so this PR records
the measurements and evidence only.

No required initialization was removed. No server, model fixture, numerical
GPU kernel, native project rebuild, or serving-path test was run because the
scope was import behavior and no source candidate was introduced here.

## Reproduction

Use `/tmp/amdpilot-repo-j-a97c2728157d/venv/bin/python` and keep caches outside
the checkout:

```bash
env XDG_CACHE_HOME=/tmp/amdpilot-repo-j-a97c2728157d/cache \
    SGLANG_CACHE_DIR=/tmp/amdpilot-repo-j-a97c2728157d/cache \
    /tmp/amdpilot-repo-j-a97c2728157d/venv/bin/python \
    -c 'from sglang.srt.managers.scheduler import Scheduler'

env XDG_CACHE_HOME=/tmp/amdpilot-repo-j-a97c2728157d/cache \
    SGLANG_CACHE_DIR=/tmp/amdpilot-repo-j-a97c2728157d/cache \
    /tmp/amdpilot-repo-j-a97c2728157d/venv/bin/python -X importtime \
    -c 'from sglang.srt.managers.scheduler import Scheduler'
```

Raw `-X importtime` output and AIter Ninja timing logs are retained in `raw/`.
