# Independent review of PR 1783

Upstream issue: https://github.com/sgl-project/sglang/issues/34259

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1828

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1783

Exact candidate: `e977642913f3b015120029ff075e207d84f299a0`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept PR 1783 as test-only hardening and an honest correction of the earlier
investigation's scope. It does **not** fully resolve the original issue. The
original intermittent Kimi-K3 generated-reasoning leakage remains unverified.

The candidate makes no production or native-code change. Its new tests are
useful for optional returned-hidden-state accounting, and its ordinary request
test correctly demonstrates that this response-slicing helper is bypassed when
hidden states are not requested. That evidence rejects the earlier proposed
causal mechanism for ordinary generation; it is not evidence that Kimi-K3
generation is fixed.

## Base and candidate comparison

The prepared checkout was clean and exactly at the recorded base. I exported
the issue, PR metadata, candidate diff, and test file to
`/job/review-evidence-j-5a7213e046b1` before switching revisions.

I ran the candidate's complete test file against the recorded base without
applying the candidate. It passed with `10 passed, 2 subtests`. Therefore PR
1783 has no failing-before/passing-after transition: the production behavior
under test already exists at the prepared base. This is test-only hardening,
not a candidate production fix.

At exact commit `e9776429`, the same suite passed with `10 passed, 2 subtests`.
The imports resolved to:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

No C, C++, CUDA, HIP, or header files differ between the base and candidate,
so no native rebuild applies. The environment imported an existing aiter
binary from
`/tmp/amdpilot-repo-j-5a7213e046b1/cache/aiter/module_aiter_core.so`; it was
not presented as candidate-built native evidence.

## Independent adversarial checks

On the assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), I exercised 100
deterministically randomized packed full-capture layouts containing two to
eight requests, mixed `False`, `True`, and `"last"` response modes, and
zero-length non-requesting rows. All non-empty requesting slices matched an
independent packed-tensor reference and all offsets reached the total forwarded
row count.

A two-request ordinary-generation batch with tokens 17 and 29 also committed
`[[17], [29]]` while a patched hidden-state response helper would have raised
if entered. This strengthens the candidate's one-request boundary, but it
still does not execute a model, HTTP concurrency, or reasoning semantics.

The randomized check found a remaining boundary counterexample in the helper:
a fully cached request with `return_hidden_states="last"`, full capture, and
`extend_input_len=0` raises
`IndexError: index -1 is out of bounds for dimension 0 with size 0`. The
candidate test named `including_zero` uses `return_hidden_states=False` for its
zero-row request and therefore does not cover this combination. This is an
optional hidden-state response defect, not a reproduction of cross-prompt
generated-reasoning leakage, and this review does not implement a speculative
candidate patch.

## Original-issue status and limitations

The original failure could not be reproduced. The issue supplies no
deterministic corpus or request sequence and says the symptom is random. This
environment has one AMD MI355X/gfx950 GPU with ROCm 7.2, while the report uses
Kimi-K3 on eight NVIDIA B300 GPUs, CUDA 13.0, and tensor parallelism. Kimi-K3
weights were unavailable. Consequently this review cannot qualify Kimi-K3
architecture behavior, eight-way scheduling, CUDA/B300 behavior, or semantic
isolation under repeated concurrent HTTP load.

The deterministic tiny-Llama fixture from mirror PR 649 at exact commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315` was inspected. It can validate
transport and engine execution only. Running it cannot substitute for
Kimi-K3, semantic reasoning checks, or the reported distributed topology, so
it was not used as proof of resolution.

The official `v0.5.17` tag contains the hidden-state offset ordering exercised
by the candidate tests. Since the report explicitly observes the symptom on
v0.5.17, a controlled rollback of that ordering demonstrates only test
sensitivity to an older helper defect; it does not reproduce the reported
failure.

Raw command output used for this review was preserved outside the switched
checkout at `/job/review-evidence-j-5a7213e046b1`.
