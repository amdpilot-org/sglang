# Nsight process-tree startup investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33283

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1919

## Outcome

The issue-specific reproduction is environment-blocked. The report requires
NVIDIA Nsight Systems profiling of a CUDA process tree, but this job has one AMD
Instinct MI355X (`gfx950`) and no `nsys` executable. No source change is
justified without observing the profiler-specific failure or a deterministic
equivalent in the checked-out implementation.

The current source was inspected before testing. Its normal HTTP launch path
sets Python multiprocessing to `spawn` in
`python/sglang/srt/entrypoints/engine.py` before scheduler and detokenizer
workers are created. Searches of the source issue, its comments, and current
upstream issue/PR metadata found no issue-linked correction to duplicate.

## What was verified

The qualified deterministic tiny Llama scripts from amdpilot-org/sglang PR 649
at exact commit `f1d603677ca76a9ea21124a544e405c5b0cbd315` were inspected before
use. The model was generated under `/tmp/j-921905cd9aaf/runtime`, outside the
checkout. The runner was adapted only in that private runtime directory to
enter through `sglang.cli.main`'s `serve` command with the required prepared
interpreter.

On the assigned gfx950 GPU, the actual checkout's CLI server became ready on a
loopback port after 25.040 seconds. Four probes returned HTTP 200: `/generate`,
`/v1/completions`, a two-prompt `/generate` batch, and a streaming completion
with eight SSE events. The server log records model loading, Triton execution,
prefill/decode work, and the requests. This validates the unprofiled transport
and engine path only; random tiny Llama output has no semantic-quality meaning.

## Evidence

- `evidence/environment.txt`: prepared interpreter, source import, Torch/HIP,
  assigned GPU and absent `nsys` result.
- `evidence/cli-direct/run-metadata.json`: exact launch command, readiness,
  probe status, cleanup, and exit information.
- `evidence/cli-direct/server.log`: complete server output.
- `evidence/cli-direct/*.json`: requests and raw/parsed responses.
- `evidence/fixture-manifest.json`: deterministic fixture configuration and
  weight digest.
- `evidence/source-issue/`: captured source and mirror issue JSON plus comments.

## Remaining limitation

The reported `nsys profile --trace=cuda,nvtx,osrt --sample=cpu --wait=all ...`
behavior, the RTX 4060 Ti/CUDA 13.2 environment, and Qwen3.5-0.8B were not
available. Therefore this report does not claim the open bug is fixed or not
reproducible on its stated platform.
