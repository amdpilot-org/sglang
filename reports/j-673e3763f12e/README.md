# Independent review of amdpilot-org/sglang PR 1847

Recommendation: **accept as a narrow partial fix**. The exact candidate commit
`f7c757917edcdfe4e0bd36a6501abc41d93a386e` fixes a reproducible GLM parser
failure mode, is correctly wired through chat and Responses API paths, and
keeps the tested truncation, user-stop, tool, continuation, and non-GLM cases
unchanged. It does **not** establish that the full original GLM-5.1 and
Kimi-K2.5 LongBench V2 failure is resolved.

Upstream issue: https://github.com/sgl-project/sglang/issues/36734

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1885

Candidate: https://github.com/amdpilot-org/sglang/pull/1847 at
`f7c757917edcdfe4e0bd36a6501abc41d93a386e`

## Finding

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
classifies all text after a template-opened GLM `<think>` as reasoning when the
model emits no `</think>`. Thus a direct final answer terminated by model EOS
has empty OpenAI `content`. With the candidate tests present but base production
code restored, the targeted class reports 27 failures and 2 passes.

The candidate passes the engine finish-reason metadata into the parser. For
`glm45`/`ling3`, an unclosed template-opened reasoning block terminated by
model EOS is returned as content. Length truncation, abort, a matched user stop
string, tool-call interruption, a client-opened continuation, and other parser
families retain their previous classification. Streaming necessarily emits the
text first as reasoning and re-emits it as content at EOS; this is deliberate
and is covered by both candidate and independent tests.

There is no native/C++ change, so no native rebuild applies. Imports were
confirmed from `/job/repo/python/sglang`, using the required interpreter
`/tmp/amdpilot-repo-j-673e3763f12e/venv/bin/python` with Torch
`2.11.0+rocm7.2` and HIP `7.2.26015`.

## Verification

- Baseline targeted regression: 27 failed, 2 passed, exit 1.
- Exact-candidate targeted regression: 26 passed plus 3 subtests, exit 0.
- Exact-candidate parser and chat/Responses API suites: 330 passed plus 146
  subtests, exit 0.
- Independent parser boundary script: model EOS represented by integer or list
  becomes GLM content; user stop and length remain reasoning; opt-out and Qwen
  isolation remain reasoning; streaming re-emission matches the contract.
- Qualified tiny random Llama server fixture ran the candidate checkout on the
  assigned AMD Instinct MI350X (`gfx950`). `/generate`, OpenAI completion,
  batch, and stream returned HTTP 200; the stream contained eight events. This
  proves actual GPU engine and transport execution only. The synthetic Llama
  cannot test GLM/Kimi parser semantics or response accuracy.

Raw evidence was retained during revision switches under
`/job/review-evidence-j-673e3763f12e/`, including the candidate diff, issue
snapshot, pytest output, independent script/output, server requests/responses,
server log, and cleanup metadata.

## Remaining limitations and counterexamples

- GLM-5.1 and Kimi-K2.5 weights and the LongBench V2 client/data were not
  available. The reported stochastic workload was not reproduced end to end.
- The report requires TP8, while this job had one `gfx950`; no multi-GPU or
  CUDA comparison was possible.
- The change is enabled for the GLM parser family, not Kimi-K2.5 generally.
  Any Kimi-specific empty-output cause remains unverified.
- A genuine GLM reasoning trace that terminates with EOS without `</think>` is
  observationally indistinguishable from a skipped-thought direct answer and
  will be reclassified as content. The candidate documents this heuristic and
  provides a per-request opt-out.
- The tiny-Llama smoke does not qualify model architecture, long-context
  behavior, semantic accuracy, concurrency 503, or TP8 execution.
