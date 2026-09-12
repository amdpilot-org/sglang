# Independent review of PR 3517

Reviewed exact commit `dbd039d251ddead80957e74febf95dc03d05e477` against https://github.com/sgl-project/sglang/issues/37066 from recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a partial fix.

The prepared base reproduces the bad guidance: both JSX calculators use `(slots + drafts)` and Qwen's explicit pin is `ceil(C * slots)`, without the transient usable slot. At the exact candidate, its focused regression passes and its arithmetic maps the supplied cases to safe allocator capacities 31, 321, and 321.

The candidate nevertheless leaves an actionable contradiction in `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`. Lines 93-98 correctly say plain speculative intermediates are separately allocated and the calculator inverts the configurator. Lines 244-251 later say DSPARK has `D` "in the ratio" as `r = (S + D) ...` and that ReplaySSM is the opposite `D = 0` case. Lines 253-256 repeat that DFlash2's `D` is the same ratio term. A reader following that model-specific guidance can still make the original accounting error. The candidate's tests only assert selected corrected strings and do not detect the stale contradictory section.

Source inspection confirms the relevant current paths: `kv_cache_configurator.py` jointly solves the plain-spec memory budget and caps request-indexed scratch by `min(max_running_requests // attn_dp_size, max_mamba_cache_size // S)`; `memory_pool.py` allocates `intermediate_ssm_state_cache` separately; and `mamba_component.py` allocates the donation slot before releasing the prior slot.

No native source changed. The prepared environment is ROCm 7.2 with no native rebuild target, and the original eight-H100 GLM workload and weights were unavailable. No GPU or substitute-model smoke was used as proof.

Raw outputs and source excerpts are retained under `reports/j-0785a13e2afb/raw/`.
