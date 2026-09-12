# Independent review evidence

Reviewed candidate: `7f23d0cd29dafc0637e85ac6c211752948711a29`

Recorded base and prepared checkout before review:
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Finding

The candidate is a meaningful partial implementation, but it does not fully
match the embedding API contract it describes as OpenAI-compatible.

SGLang's `EmbeddingRequest` accepts `List[int]` and `List[List[int]]` token-ID
inputs in addition to strings and lists of strings. The candidate dataset
loader rejects both token-ID forms before they can reach `/v1/embeddings`.
Conversely, the loader accepts a whitespace-only string and counts it as zero
tokens, although `OpenAIServingEmbedding._validate_request` explicitly rejects
whitespace-only input. Thus the new loader is neither a complete representation
of the serving schema nor aligned with its string validation.

The candidate's four added focused tests pass and its offline path calls
`encode()` rather than `generate()`. This validates the implemented string-only
subset, not the complete compatibility claim.

## Reproduction and tests

All Python commands used the prepared interpreter
`/tmp/amdpilot-repo-j-af3a4aba4c00/venv/bin/python` with `PYTHONPATH=python`.
Raw logs were retained outside the checkout in
`/job/review-evidence-j-af3a4aba4c00/` while switching revisions.

1. At the recorded base, running `python -m sglang.benchmark.serving --backend
   sglang-embedding --dataset-name embedding --dataset-path
   /tmp/nonexistent.jsonl --model dummy --num-prompts 1` exited 2 because
   argparse did not offer the `embedding` dataset. This reproduces the original
   missing benchmark workflow.
2. At the exact candidate, `python -m pytest
   test/registered/bench_fn/test_benchmark_datasets_api.py -q -k
   'embedding_dataset or offline_embedding'` exited 0: 4 passed and 45 were
   deselected in 12.49 seconds.
3. `python -m py_compile` on all three changed Python implementation files
   exited 0. `git diff --check` against the recorded base also exited 0.
4. An independent loader script supplied one JSONL record at a time with
   `[101, 102, 103]`, `[[101, 102], [103]]`, and `"   "` as `input`. The two
   valid token-ID forms raised `ValueError`; whitespace was accepted with
   `prompt_len == 0`.
5. Imports resolved to the candidate checkout:
   `/job/repo/python/sglang/benchmark/datasets/embedding.py` and
   `/job/repo/python/sglang/benchmark/offline_throughput.py`.

An attempt to run the entire 49-test candidate module began with six passing
tests but was externally terminated before pytest wrote an exit code. The
focused changed-test selection was then run successfully. This review does not
represent the interrupted full-module run as passing.

## Architecture and environment

- Host: Linux x86-64.
- Assigned GPU: AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, about 252 GiB
  VRAM.
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`.
- The candidate's cited MI355X/Qwen runtime directory was not present in this
  review environment. No model weights were prepared for this dataset/API
  contract review, so the candidate's numerical GPU claims were not
  independently repeated. GPU execution is therefore reported as false.
- No C, C++, HIP, CUDA, or other native source changed. The prepared repository
  declares no native rebuild target, so a native rebuild was not applicable.
- A vLLM server was not installed or run; vLLM transport remains unverified.

## Conclusion

Recommendation: request changes. The candidate fixes the original absence for
string embedding inputs and adds useful offline encoding support, but it is a
partial fix. Supporting the serving contract's token-ID input forms and
rejecting whitespace-only strings (with regressions for both) are remaining
counterexamples to the claimed OpenAI-compatible dataset workflow.
