# Independent review of PR 935

Candidate: `4379f80a1126404096aed27bc3fbe37833ff1f04`

Upstream issue: https://github.com/sgl-project/sglang/issues/37848

Mirror issue: https://github.com/amdpilot-org/sglang/issues/970

## Finding

Request changes. The candidate is a verified narrow fix for
`BailingMoELinearForCausalLM.lm_head`, but it does not fully resolve the
original repository-wide issue. The original source audit failed at the
recorded base with 53 omitted-prefix construction sites and still failed at
the exact candidate with 52. Concrete remaining counterexamples include
`bailing_moe_v3.py:1332` (`ParallelLMHead`), `whisper.py:369`
(`ParallelLMHead`), `grok.py:151` (`FusedMoE`), and `exaone.py:152`
(`RadixAttention`).

The candidate's added constructor regression is meaningful: copied unchanged
outside the checkout, it failed on base because `ParallelLMHead` received no
`prefix` keyword, and the same test passed on the candidate for both root and
nested model prefixes. Thus this is a partial source fix with a valid
regression, not merely test hardening. However, the candidate's retained audit
only prints its 52 remaining failures and exits zero, so it cannot establish
the original contract as passing.

## Evidence

- Base `358c163250ad3b1f62939b01ce1314a0a31a0365`: independent AST audit exited
  1 with `missing_prefix_sites=53`.
- Candidate `4379f80a1126404096aed27bc3fbe37833ff1f04`: focused candidate suite passed
  (`10 passed`, `2 subtests passed`).
- The candidate test copied outside the checkout and run on base failed
  (`1 failed`, `9 passed`) with `KeyError: 'prefix'` at the assertion inspecting
  the Bailing LM-head constructor call.
- Candidate: independent AST audit exited 1 with
  `missing_prefix_sites=52`; the candidate's own audit printed the same 52 but
  exited 0.
- Imports resolved to the checkout source:
  `/job/repo/python/sglang/__init__.py` and
  `/job/repo/python/sglang/srt/models/bailing_moe_linear.py`.
- Diff inspection found no native/C++ changes, so native rebuilding was not
  applicable.

Raw command output was retained outside the revision-switching checkout under
`/tmp/amdpilot-repo-j-c541de9eef73/review-evidence/`.

## Environment and limits

The prepared interpreter used Torch `2.11.0+rocm7.2` with HIP `7.2.26015`.
One assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) was visible. No GPU
execution was needed or claimed because the failure is CPU-reproducible source
and constructor prefix routing. No Bailing, Whisper, Grok, or other model
weights were available, so full-model semantic or serving validation was not
performed. The deterministic tiny Llama fixture would not qualify these model
architectures or the repository-wide per-module quantization contract.
