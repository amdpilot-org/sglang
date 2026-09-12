# Independent review of PR 3175

Candidate: https://github.com/amdpilot-org/sglang/pull/3175

Exact commit: `3cd00373248daf94cc7fde4a8009691001148995`

Upstream issue: https://github.com/sgl-project/sglang/issues/9674

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3118

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3176

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's documented embedding benchmark and dataset contract and the three concrete counterexamples inherited from the prior review. This is a source fix with regression hardening, not a test-only change.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduces the original absence: importing `sglang.benchmark.datasets.embedding` raises `ModuleNotFoundError`. At the exact candidate, the source loader imports from `/job/repo/python/sglang/benchmark/datasets/embedding.py` and accepts `[101, 102, 103]` plus `[[101, 102], [103]]` with three tokens each. It rejects a whitespace-only string and a batch containing a whitespace-only member.

## Independent adversarial coverage

The review compared the loader with `EmbeddingRequest` and exercised documented valid forms plus invalid edges. Empty strings/lists, negative token IDs, malformed nested token batches, and whitespace-only inputs were rejected. A real `/v1/embeddings` server accepted text, string-batch, token-ID, and token-ID-batch payloads and rejected whitespace-only text with HTTP 400.

One SGLang extension is intentionally outside the documented JSONL contract: multimodal embedding objects validate under `EmbeddingRequest` but the candidate loader rejects them. This is recorded as a limitation rather than a remaining original-issue counterexample because the candidate describes and implements the OpenAI-compatible text/token forms requested for the benchmark workflow.

## Environment and source paths

- Prepared interpreter: `/tmp/amdpilot-repo-j-c4e6dd5a3e66/venv/bin/python`
- Candidate source imports: `/job/repo/python/sglang/...`
- GPU: AMD Instinct MI350X, `gfx950`
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`
- Real server: candidate source checkout, Triton attention, embedding mode
- Fixture: deterministic two-layer random tiny Llama, SHA256 `6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`
- Native rebuild: not applicable; no native source changed

The tiny fixture establishes transport and engine execution only. It does not establish semantic embedding accuracy or qualify a specialized embedding architecture. No suitable real embedding-model weights were available for an independent numerical reference, and vLLM server interoperability was not executed.

## Raw evidence

- `raw/base-probe.txt`: failing-before source import
- `raw/candidate-probe.txt`: loader/schema matrix and exact source paths
- `raw/candidate-focused-pytest.txt`: focused candidate regressions
- `raw/candidate-full-pytest.txt`: full benchmark dataset API suite
- `raw/candidate-pycompile.txt`: changed Python module compilation
- `raw/http-probe.json`: requests, statuses, responses, launch command, and cleanup
- `raw/http-server.log`: real server/GPU execution log
- `raw/gpu-before.txt`: assigned GPU identity and idle state

Complete preserved evidence also remains outside the checkout at `/job/review-evidence-j-c4e6dd5a3e66/`.
