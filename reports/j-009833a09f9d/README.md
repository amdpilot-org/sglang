# Independent review of PR 2560

Candidate: `ab77a09846cc2503a63052c111d085807af255e0`

Upstream issue: https://github.com/sgl-project/sglang/issues/18262

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2493

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2563

## Conclusion

Recommendation: **accept**, with `fully_resolves_original=false` because the exact
MI355X/Qwen serving reproductions (including online MXFP4) could not be run.  The
candidate is a substantive source fix, not test-only hardening: it charges the
legacy AITER workspace against the profiled KV budget and makes the allocator use
the same byte-count helper.

The deterministic issue-shaped check overcommitted the 219,721,119,039-byte
budget by 69,793,139,393 bytes with the candidate reservation disabled.  At the
exact candidate, it selected 1,669,878 KV tokens and 3,261 requests, leaving
79,167 bytes of headroom.  The candidate's focused suite passed (45 tests plus 9
subtests), and an independent allocation on the assigned MI350X gfx950 allocated
the helper's exact 65,495,040 bytes and successfully wrote both boundaries.

## Environment and limitations

The prepared base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`;
the prepared checkout did not differ. Imports resolved to `/job/repo/python`.
PyTorch was `2.11.0+rocm7.2`, HIP was `7.2.26015`, and the assigned GPU was one
AMD Instinct MI350X (`gfx950:sramecc+:xnack-`). The reported 288 GiB MI355X,
Qwen3-30B-A3B weights, online MXFP4 case, and multi-GPU/distributed execution were
unavailable. No native sources changed, so no native rebuild was applicable.

Raw logs and the reviewed code diff are retained in this directory. The original
base checkout could not directly run the candidate-authored reproduction script
because that script patches a helper introduced by the candidate; the preserved
failing-mode run at the candidate disables the new reservation and measures the
pre-fix resolver behavior. This is a limitation of the submitted regression
harness, not evidence of a full serving reproduction.
