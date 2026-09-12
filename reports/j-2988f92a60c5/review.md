# Independent review of amdpilot-org/sglang#704

Reviewed exact commit `1edb037e7b5a9177d9f059f08cdf1b1f35f2db2f` against upstream issue https://github.com/sgl-project/sglang/issues/38069 and mirror issue https://github.com/amdpilot-org/sglang/issues/711.

Recommendation: **unverified**. The source-level correction and focused CPU tests are convincing, including restoration of the legacy `bootstrap_host == "2.2.2.2"` behavior that candidate parent `45638ad1cf0d18cad80c9b6095bd063e9693e1a3` regressed. The original issue nevertheless requires real device and HiCache behavior. The supplied E2E regression could not start because its model is gated and unavailable in the prepared environment, so this review cannot honestly mark the full issue resolved.

## Evidence

- Base (`358c163250ad3b1f62939b01ce1314a0a31a0365`): normal requests set `skip_radix_cache_insert=False`; the legacy fake bootstrap host sets it to `True`; constructing `GenerateReqInput(skip_cache_insert=True)` raises `TypeError` because the public field is absent.
- Candidate imports resolved to `/job/repo/python/sglang/...`, confirming tests used the checked-out source.
- Candidate focused suite: 213 tests and 78 subtests passed.
- Independent adversarial script passed sentinel precedence, explicit true/false isolation, mixed batch expansion, and `release_kv_cache(... is_insert=False)` cleanup delegation.
- Candidate GPU E2E: five setup errors. Hugging Face returned 401 for `meta-llama/Llama-3.2-1B-Instruct` before inference. A single MI350X (`gfx950`) was visible, but no GPU cache behavior executed.
- No C++ or other native source changed, and the prepared environment records `native: null`; no native rebuild was applicable.

Raw logs and the independent script are retained outside the checkout at `/job/review-evidence-j-2988f92a60c5`.
