# Independent review of candidate PR 1287

Upstream issue: https://github.com/sgl-project/sglang/issues/36105

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1322

Candidate: https://github.com/amdpilot-org/sglang/pull/1287 at exact commit
`9038ebdca6547819590298996112fa2f59309bab`.

## Verdict

Recommendation: **accept**. The candidate fully resolves the original reported
contract: `sglang serve --config FILE` no longer rejects a YAML-provided
`model-path` before the normal server argument parser reads the file.

This is a source fix with regression hardening, not a test-only change. It
updates `python/sglang/cli/utils.py` so backend detection uses the existing
`ConfigArgumentMerger` when no explicit CLI model path is present.

## Evidence

The image-prepared checkout was already at the required recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference to
reconcile. Using the prepared interpreter and `PYTHONPATH=/job/repo/python`, a
direct invocation of the real console entry function with argv equivalent to
`sglang serve --config config.yaml` failed on the base with:

```text
Exception: Error: --model-path is required. Please provide the path to the model.
```

The candidate was then checked out detached at the exact requested commit.
Imports resolved to `/job/repo/python/sglang/...`, proving the tested Python
source came from that checkout. The candidate changes no C++, HIP, CUDA, Rust,
or other native source, and the prepared environment declares no native build;
therefore a native rebuild was not applicable.

The candidate's focused suite passed 17 tests. An independent script exercised
the real `sglang.cli.main.main` and `serve` dispatch, replacing only the final
`run_server` boundary to avoid requiring model weights. A YAML-only local model
path and `host` both reached `ServerArgs`. Independent boundary cases confirmed
that an explicit CLI model path overrides YAML during detection, the `model`
alias is accepted from YAML, and missing or malformed YAML model values do not
invent a model path.

Raw outputs and exit statuses are retained under `raw/`.

## Limitations

The prepared host is ROCm 7.2 with a visible gfx950 GPU, while the report came
from a CUDA 13.0 container. This parser/dispatch path is architecture-neutral,
and no GPU execution was necessary or performed. No model was loaded, no HTTP
request or inference was run, and semantic model accuracy or distributed
serving was not evaluated. The prepared virtual environment did not contain an
installed `sglang` console script, so the equivalent registered entry function
(`sglang.cli.main:main`) was invoked with the exact console argv instead.
