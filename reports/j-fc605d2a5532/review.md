# Independent review of PR 939

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/939 at `80c019da6db285e88d02ea4c6dbf26e93435bf5a`

Upstream issue: https://github.com/sgl-project/sglang/issues/39096

Mirror issue: https://github.com/amdpilot-org/sglang/issues/979

Earlier candidate: https://github.com/amdpilot-org/sglang/pull/753 at `bcbb219ef7d4c2a9a6d121928faec7f87d7b7ce0`

Earlier independent review: https://github.com/amdpilot-org/sglang/pull/850

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue and the correction-generation-2 explicit-JSON-null counterexample. No remaining behavioral counterexample was found.

## Evidence

The prepared branch was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, both omitted fields and explicit JSON null for `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty` were converted to generation-config values and then overwrote `--preferred-sampling-params`. The base output is retained in `raw/base-reproduction.txt`.

At the exact candidate SHA, the candidate regression suite passed 6 tests. Independent adversarial validation parsed JSON into `ChatCompletionRequest`, converted it to sampling params, carried explicit-key metadata through `GenerateReqInput`, and applied the tokenizer-manager merge. It confirmed:

- all preference-supported sampling keys win when the client omits them;
- explicit JSON null for the five nullable `get_param()` fields behaves as unset;
- explicit client values, including values equal to schema defaults, still win;
- mixed explicit, null, and omitted inputs preserve the intended precedence;
- `max_tokens`/`max_completion_tokens` map correctly to `max_new_tokens`;
- explicit-key metadata survives batch item extraction; and
- native requests without explicit-key metadata retain their previous request-over-preference behavior.

The nearby OpenAI unit suite passed 349 tests and 127 subtests. Raw outputs are under `raw/`.

## Source and environment qualification

Imports resolved to the candidate checkout under `/job/repo/python`, including `protocol.py`, `serving_chat.py`, `io_struct.py`, and `tokenizer_manager.py`. The interpreter was `/tmp/amdpilot-repo-j-fc605d2a5532/venv/bin/python` with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`.

The candidate changes only Python request/manager code and tests; no native source changed, so no native rebuild was applicable. A ROCm GPU was visible, but GPU execution and model weights are irrelevant to this deterministic pre-engine request-path defect and were not used as evidence. No live HTTP/model server was launched; JSON parsing, conversion, request metadata, and the merge contract were exercised directly.

`git diff --check` reports one trailing-whitespace line in a candidate-owned raw report artifact (`reports/j-d347b4015ea2/raw/focused-tests-after.txt`). It does not affect product code or the reviewed contract.
