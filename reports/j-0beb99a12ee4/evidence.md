# Validation evidence

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Baseline reproduction

At the recorded base, `bench_serving --dataset-name embedding` was rejected by
argparse and `offline_throughput` unconditionally called `backend.generate()`.
The regression tests added by this change exercise both missing paths and fail
against that baseline.

## CPU regression and compatibility suite

Command:

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-0beb99a12ee4/venv/bin/python -m pytest test/registered/bench_fn/test_benchmark_datasets_api.py -q
```

Result: `49 passed, 6 subtests passed` (exit 0). This measured dataset dispatch,
single and array embedding inputs, exact token accounting, payload preservation,
malformed/undersized workload rejection, online backend compatibility, and the
offline `encode()`-rather-than-`generate()` regression.

All changed files also passed the repository pre-commit hooks. The prepared
interpreter did not contain a `ruff` module, so formatting and linting used the
repository's pinned pre-commit environments under
`/tmp/amdpilot-repo-j-0beb99a12ee4/precommit`.

## Real GPU execution

Hardware: one AMD Instinct MI355X (`gfx950`), ROCm 7.2, Torch
`2.11.0+rocm7.2`.

Model source:
`/tmp/amdpilot-repo-j-0beb99a12ee4/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`

The online server was launched from this checkout with `--is-embedding` on port
31000. The new JSONL dataset benchmark sent three records. One ordinary request
succeeded; two requests deliberately supplied `dimensions=256` and reached the
server, which rejected them because this model does not advertise Matryoshka
support. The raw benchmark result, including per-request errors, is retained at
`/tmp/amdpilot-repo-j-0beb99a12ee4/embedding_benchmark_details.jsonl`.

A separate valid request containing two strings returned two 1024-dimensional
embeddings. An independent `transformers.AutoModel` GPU forward pass used
last-token pooling and L2 normalization. Cosine similarities between SGLang and
the independent reference were `0.9997416` and `0.9998755`; maximum absolute
differences were `0.0027683` and `0.0021243`.

The new offline benchmark then loaded the same model in a fresh engine and
encoded two JSONL records. It completed in 3.06 seconds, reported 19 input
tokens, 0.65 requests/s and 6.21 input tokens/s, and omitted generation-only
metrics. Raw JSON is retained at
`/tmp/amdpilot-repo-j-0beb99a12ee4/offline_embedding_result.jsonl`.

## Limitations

- vLLM interoperability was covered by request-path unit tests but a vLLM
  server was not installed or run.
- Offline JSONL records support one string per record and a common optional
  `dimensions` value. Input arrays and other OpenAI-only fields are rejected
  with an instruction to use `bench_serving`, where they are fully supported.
- No native C++ was changed, so a native rebuild was not applicable.
