# Independent review of PR 971

Upstream issue: https://github.com/sgl-project/sglang/issues/37755

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1005

Candidate: https://github.com/amdpilot-org/sglang/pull/971 at
`2e9ee6b921a55734331216fef05ff3390b9073ca`.

## Verdict

Recommendation: **request changes**. The candidate is useful test-only
hardening for an implementation already present at the recorded base, but it
does not verify or fully resolve the original open issue.

The exact candidate changes no production or native source. Its six new tests
pass both at the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` and at the candidate commit. Thus
there is no failing-before/passing-after reproduction, and the tests cannot
establish that the current implementation corrected the reported ModelSlim
checkpoint failure. The base checkout exactly matched the recorded base.

The source-level behavior is internally consistent for the synthetic layout:
independent cases checked every compatible runtime TP/rank and used
non-uniform block scales so incorrect scale-to-row association would be
observable. The latter passed on CPU and on the assigned AMD Instinct MI355X
(`gfx950`). These checks validate tensor arithmetic under the assumed format;
they do not identify the actual private checkpoint's metadata or validate
Ascend model accuracy.

## Material evidence

- Candidate diff: nine added files, consisting of one test module and report
  artifacts; zero changes under `python/sglang` and no native changes.
- Base run of the candidate test: 6 passed. This is the principal remaining
  counterexample to treating the test as proof of an original-issue fix.
- Exact candidate run: 6 passed.
- Independent all-rank mapping for runtime attention TP 1, 2, 4, and 8: passed.
- Independent non-uniform scale-to-row mapping: passed on CPU and gfx950.
- Imported source was `/job/repo/python/sglang/srt/models/mimo_v2.py` from the
  active checkout. Torch was `2.11.0+rocm7.2`, HIP `7.2.26015`.
- No native source changed, so a native rebuild was neither required nor
  performed.

## Unverified original contract

The reported `/home/weights/MiMo-V2.5-Pro-W8A8` checkpoint was unavailable.
There was no Ascend 910 hardware or CANN 9.1 runtime, and only one AMD GPU was
assigned rather than the reported two-node, 64-device TP32/DP4 deployment.
Consequently the actual ModelSlim tensor names, shapes, quantization metadata,
load path, generated tokens, and semantic accuracy remain unverified. A tiny
Llama serving fixture would not qualify this different architecture or weight
format and was therefore not substituted for the missing reproduction.

Raw command summaries are retained in `raw/`; the complete switching evidence
and command output was also preserved outside the checkout at
`/job/review-evidence-j-4b5ca5633170/`.
