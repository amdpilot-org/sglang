# Retract cache provenance correction generation 2

Candidate parent: https://github.com/amdpilot-org/sglang/pull/1965 at `723d55b5b5f30fba408b11761e3bd15292d4623b`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/2022

Upstream issue: https://github.com/sgl-project/sglang/issues/33783

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2058

The exact candidate reproduced the review counterexample: with `disaggregation_mode=PREFILL`, an unfinished `chunked_req`, and an independently inserted four-token simulated RadixCache prefix, retract left the prefix matchable 4/4, retained the chunk, and did not clear the token allocator.

The prepared base already has a safe chunked-prefill abort sequence in `process_pending_chunked_abort`: stop pending chunk-send bookkeeping, abort the sender before releasing KV, release its metadata buffer, and clear bootstrap state. The correction uses the same ownership ordering without aborting the user request. It then includes the chunk in normal retract, performs the candidate's full cache/pool reset, clears `chunked_req`, and requeues the request through `_add_request_to_queue`, which creates a fresh sender/bootstrap from retained token IDs.

After correction the same cache probe measured 0/4 tokens matchable, no retained chunk, one sender abort, one allocator clear, and one requeue. Focused pause/radix tests passed 36 tests plus 2 subtests; related PREFILL bootstrap/abort/race tests passed 25 tests.

The assigned AMD Instinct MI355X completed a basic Torch tensor operation. No model weights or multi-node disaggregated fixture were available, so this does not claim the original Qwen output divergence or a live distributed transfer reproduction.
