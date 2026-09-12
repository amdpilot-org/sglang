# Independent review of PR 3127

- Candidate: https://github.com/amdpilot-org/sglang/pull/3127
- Exact candidate commit: `a02a0a49441faa169529abec7f93cfcdba92fe13`
- Upstream issue: https://github.com/sgl-project/sglang/issues/3365
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3054
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3140
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Request changes. The candidate is a real partial implementation for ordinary eager
generation, but it does not safely cover every path selected by its policy and does
not fully resolve the original feature request.

## Reproduction and positive evidence

The prepared checkout was exactly the recorded base. Running the candidate's test
file from outside the checkout against that base produced 9 failures because the
gather policy, output marker, and rank-zero sampling path did not exist. At the exact
candidate commit, the candidate's focused suite plus the adjacent decode-row suite
passed 10 tests.

Imports were confirmed to resolve to the checked-out sources:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/layers/logits_processor.py`
- `/job/repo/python/sglang/srt/model_executor/model_runner.py`

The candidate changes Python only. No C++, HIP, CUDA, FlyDSL, or other native source
changed, so a native rebuild was not applicable.

A one-GPU MI350X calculation independently compared a full FP32 vocabulary
projection with two separately computed vocabulary shards concatenated in TP order.
It produced `max_abs_error=0.0` and identical argmax token IDs. This validates the
local sharding/order arithmetic only; it does not validate an RCCL gather or a
distributed serving run.

## Blocking counterexample: eager TP dLLM

At the candidate commit, `LogitsProcessor.forward` dispatches
`ForwardMode.DLLM_EXTEND` to `_get_dllm_logits`. That method calls `_get_logits` and
places the result in `LogitsProcessorOutput.full_logits`. Every rank then calls
`DllmAlgorithm.step(forward_batch, out.logits_output.full_logits, states)`.

However, `_use_tp_logits_gather` does not exclude `DLLM_EXTEND` or
`return_full_logits`. For an eager TP dLLM batch (`can_run_decode_cuda_graph=False`),
the policy returns true. `_get_logits` consequently gathers only to TP rank 0 and
returns `None` on every non-root rank. Those ranks still execute the dLLM algorithm,
whose implementations consume `full_logits` as a tensor. The independent
adversarial test expected this all-rank consumer to retain all-gather and failed:

```text
assert not _processor()._use_tp_logits_gather(metadata)
E assert not True
```

This path is not speculative decoding (`spec_algorithm` can be `None`) and is not
made safe by the candidate's sampled-token broadcast, because dLLM bypasses
`ModelRunner.sample` and consumes full logits directly.

## Additional counterexample: non-root cleanup

`ModelRunner._preprocess_logits` deliberately clears `sampling_info.grammar_mask`
after applying it to prevent retained GPU masks in overlap scheduling. The candidate
skips `_preprocess_logits` entirely on non-root gather ranks and also skips clearing
`logits_output.auxiliary_device_output`. An adversarial non-root test confirmed both
objects remain set after `ModelRunner.sample`. This is a rank-local memory/state
regression for structured-output batches, even though token IDs are broadcast.

## Environment and architecture limits

Only one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) was visible, with Torch
`2.11.0+rocm7.2` and HIP `7.2.26015`. Therefore no real two-rank RCCL execution,
distributed model run, transport/serving run, or communication-performance
comparison was possible. The installed ProcessGroupNCCL headers declare a gather
override, but source declarations are not runtime RCCL evidence. The candidate's
claimed raw Gloo and GPU logs are referenced in its result JSON but are not present
in commit `a02a0a4`; its prose was not treated as proof.

Raw review evidence is retained outside the revision-switched checkout at
`/job/review-evidence-j-315f98945407/`.
