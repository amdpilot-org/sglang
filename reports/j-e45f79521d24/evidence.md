# Correction evidence

Corrected candidate: https://github.com/amdpilot-org/sglang/pull/2995 at
`7f23d0cd29dafc0637e85ac6c211752948711a29`.

Independent review: https://github.com/amdpilot-org/sglang/pull/3096.

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Reproduction before correction

The candidate commit was cherry-picked unchanged onto the prepared base. A
standalone loader script wrote each reviewed input to a one-record JSONL file
and called `sample_embedding_requests` with the prepared interpreter and
`PYTHONPATH=python`.

The exact candidate rejected `[101, 102, 103]` and `[[101, 102], [103]]` with
`ValueError`, and accepted `"   "` with `prompt_len == 0`. Raw output is retained
at `/job/repro-evidence-j-e45f79521d24/failing-before.txt`.

This agrees with the source contract: `EmbeddingRequest.input` includes
`List[int]` and `List[List[int]]`, while
`OpenAIServingEmbedding._validate_request` rejects empty or whitespace-only
strings.

## Correction

The loader now accepts and counts tokens for all four textual/tokenized OpenAI
input shapes: `str`, `List[str]`, `List[int]`, and `List[List[int]]`. It rejects
whitespace-only strings both individually and within string batches. The
candidate's payload pass-through, online dataset registration, offline
`encode()` path, metrics, and existing tests remain intact. Documentation now
lists the token-ID forms.

## Passing-after validation

All commands used `/tmp/amdpilot-repo-j-e45f79521d24/venv/bin/python`.

```bash
PYTHONPATH=python /tmp/amdpilot-repo-j-e45f79521d24/venv/bin/python \
  -m pytest test/registered/bench_fn/test_benchmark_datasets_api.py -q \
  -k 'embedding_dataset or offline_embedding'
```

Result: 6 passed, 45 deselected, 2 subtests passed. Raw output:
`/job/repro-evidence-j-e45f79521d24/focused-after.txt`.

```bash
PYTHONPATH=python /tmp/amdpilot-repo-j-e45f79521d24/venv/bin/python \
  -m pytest test/registered/bench_fn/test_benchmark_datasets_api.py -q
```

Result: 51 passed, 8 subtests passed. Raw output:
`/job/repro-evidence-j-e45f79521d24/full-module-after.txt`.

The same standalone script now accepts both token-ID examples with a total of
three input tokens each and rejects the whitespace-only example. Raw output:
`/job/repro-evidence-j-e45f79521d24/passing-after.txt`.

`python -m py_compile` passed for all three candidate implementation files, and
`git diff --check` passed.

## Environment and limitations

No model execution is needed to prove these deterministic loader/schema
counterexamples. GPU execution was therefore not performed, and unavailable
model weights were not used to justify a source change. No native source was
changed, so a native rebuild is not applicable. The candidate's previous GPU
numerical evidence is preserved in its original report but was not repeated.
vLLM server interoperability was not executed in this correction.
