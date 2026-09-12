# MI355X graph-replay profiling investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/31545

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2654

## Outcome

The attribution defect is reproduced on the assigned MI355X with the prepared
checkout at `358c163250ad3b1f62939b01ce1314a0a31a0365`. No SGLang production
change is proposed because the missing association is below SGLang's
`record_function` marker layer and the available SGLang-side alternative was
not qualified end to end on the unavailable DeepSeek TP4 workload.

This image differs from the issue's v0.5.14 observation. With Torch
2.11.0+rocm7.2 / HIP 7.2.26015, the graph-on torch trace has nine CPU decode
markers and nine `hipGraphLaunch` calls, but zero GPU kernels overlap any decode
marker. Its 171 GPU kernels are prefill/uncaptured housekeeping; the internal
decode graph dispatches are not unfolded. The eager trace has 79 kernels inside
each of the same nine decode markers. Thus disabling graphs remains a
diagnostic comparison only, not a fix.

The serving test uses the deterministic tiny random Llama fixture from
amdpilot-org/sglang PR 649 at exact commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Its generated weights stayed in
`/tmp/amdpilot-repo-j-60f61744b18d/models`, outside the checkout. This qualifies
the HTTP transport, engine execution, graph replay, and tracing mechanism only.
It does not qualify DeepSeek-V4-Pro, TP4, DP attention, MoE overlap, semantic
accuracy, or distributed timing. The eager and graph runs nevertheless emitted
the same 48 deterministic output token IDs.

## Versions and backend assessment

- GPU: AMD Instinct MI355X, `gfx950`
- Python: 3.12.3
- PyTorch: 2.11.0+rocm7.2, commit
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`
- PyTorch HIP runtime: 7.2.26015
- System ROCm tools: 7.2.4
- rocprofv3 / rocprofiler-sdk tool: 1.1.0, revision
  `97f5574fe2fdc7bef44fb01545347912ee9f1779`

PyTorch reports `USE_KINETO` and `LIBKINETO_NOCUPTI`. The wheel ships
`libroctracer64.so`, and both server captures emit Kineto's runtime warning
`ROCTracer produced duplicate flow start`; this is direct evidence that the
actual torch.profiler GPU activity backend in this image is ROCTracer. The wheel
also ships `librocprofiler-register.so`, but that registration library alone
does not establish that Kineto is collecting through rocprofiler-sdk. A current
SGLang test comment calls the attachment rocprofiler-sdk, but it conflicts with
the observed runtime backend and should not override it.

The separately installed `rocprofv3` is a viable rocprofiler-sdk dispatch-level backend. A controlled HIP
graph probe run with `--kernel-trace --marker-trace --hip-runtime-trace` exposed
four dispatches inside every one of ten replay markers, while matching an
independent CPU result exactly. However, SGLang's `step[DECODE ...]` spans are
currently torch `record_function` annotations, and rocprofv3 does not consume
those Kineto annotations. It needs ROCTx/NVTX markers emitted in the profiled
process. The probe confirmed that `torch.cuda.nvtx.range_push/pop` is captured
on this ROCm stack; the separately installed Python `nvtx` and `roctx` modules
are absent. A supported SGLang integration therefore needs both:

1. opt-in step markers using the Torch NVTX API (which maps to ROCm marker
   records here), and
2. an external `rocprofv3` launch/attach lifecycle and output handling.

That path cannot be folded into an already-running torch-profiler session: the
rocprofiler-sdk tool is injected at process launch or attach, and its timestamps
are in a separate trace domain. Implementing marker emission alone would not
correct `/start_profile`; implementing a launcher without full SGLang and
DeepSeek TP4 validation would overstate the result. No node-wide tooling was
changed.

## Reproduction

Use the prepared interpreter for every Python command:

```bash
export PYTHONPATH=/job/repo/python
export PY=/tmp/amdpilot-repo-j-60f61744b18d/venv/bin/python
export FIXTURE=/tmp/amdpilot-repo-j-60f61744b18d/models/tiny-random-llama

$PY reports/j-60f61744b18d/run_profile_probe.py \
  --fixture "$FIXTURE" --output reports/j-60f61744b18d/evidence/eager
$PY reports/j-60f61744b18d/run_profile_probe.py --graph \
  --fixture "$FIXTURE" --output reports/j-60f61744b18d/evidence/graph
$PY reports/j-60f61744b18d/analyze_traces.py \
  reports/j-60f61744b18d/evidence/eager/traces/eager-TP-0.trace.json.gz \
  reports/j-60f61744b18d/evidence/graph/traces/graph-TP-0.trace.json.gz

rocprofv3 --kernel-trace --marker-trace --hip-runtime-trace \
  --output-format json --output-directory /tmp/rocprof-output \
  --output-file graph_probe -- \
  $PY reports/j-60f61744b18d/rocprof_graph_probe.py
```

The raw requests, responses, server logs, traces, version output, linkage
evidence, and rocprofiler-sdk JSON are retained under `evidence/`.

## Remaining limitations

- DeepSeek-V4-Pro weights and four assigned GPUs were unavailable, so the exact
  TP4 + DP-attention + MoE workload remains unverified.
- This Torch/ROCm build produces opaque graph replay in torch.profiler rather
  than the post-marker unfolded-kernel burst in the reported v0.5.14 image.
- The rocprofiler-sdk dispatch result is a controlled graph probe, not an
  end-to-end SGLang server capture. It establishes backend capability and the
  missing marker dependency, not a completed `/start_profile` fix.
- No native source was changed or rebuilt.
