# Independent review of amdpilot-org/sglang PR 2706

Candidate: `0e4ffed88e34a6c710e94cfb49253c5fbcaa1283`

Upstream issue: https://github.com/sgl-project/sglang/issues/37944

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2674

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2718

## Recommendation

Request changes. The candidate adds a plausible per-forward token threshold and its focused unit tests pass, but it does not consistently apply that threshold to the DSA interleave path.

For a configured interleave strategy with `cp_size=4`, `min_tokens=16`, and an 8-token extend batch, the candidate reports generic CP execution inactive and DSA CP execution inactive, while `can_dsa_prefill_cp_interleave()` remains true. That stale static predicate is used by `dsa_backend.py` to shard `seqlens_expanded`, per-request lengths, batch indices, cache lengths, and page tables. A below-threshold batch can therefore enter CP-specific DSA metadata preparation even though `prepare_cp_forward()` was not called and the model forward is meant to be replicated/non-CP.

This is a remaining counterexample to the candidate's explicit claim that the threshold applies to interleave and that short batches use the replicated non-CP path. It is also absent from the submitted tests, which only exercise the new runtime threshold with zigzag.

## Before and after

The prepared checkout exactly matched the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; the candidate is a direct child of that commit. On the base, initializing a CP strategy with `min_tokens` fails with `TypeError`, preserving a concrete failing-before result. At the candidate, the submitted focused regression passes: 30 tests and 18 subtests.

The independent DSA interleave adversarial test fails with:

```text
execution_cp=False
dsa_execution_cp=False
dsa_interleave_padding_cp=True
AssertionError: below-threshold batch still selects CP-specific DSA interleave padding
```

## Environment and limitations

The prepared interpreter imports SGLang source from `/job/repo/python/sglang` and Torch 2.11.0+rocm7.2 from `/opt/venv`. The assigned GPU is an AMD Instinct MI355X with HIP 7.2. SGLang's `is_dsa_enable_prefill_cp()` deliberately returns false on HIP, so the reported GLM-5.2-FP8 CUDA/DSA CP workload, multi-rank CP numerical equivalence, and TTFT improvement could not be executed. No model weights were available. GPU execution was limited to confirming the device/runtime; no qualifying CP kernel executed.

No C/C++/CUDA/HIP/native source changed in the candidate, so a native rebuild was not applicable. Python source imports were confirmed to come from the exact checked-out candidate while testing.

Accordingly, the candidate is a partial implementation with a demonstrated interleave defect and an unverified original zigzag hardware/model claim, not a full original-issue fix.
