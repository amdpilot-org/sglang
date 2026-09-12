# Independent review of amdpilot-org/sglang PR 2175

Reviewed candidate: `ecec4d4a021843705851bafece12cdb21d78e8fe`

Upstream issue: https://github.com/sgl-project/sglang/issues/32250

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2104

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2210

## Recommendation

Accept. The candidate is useful test-only hardening and its regression is correct. It does not contain the production fix: the recorded prepared base, `358c163250ad3b1f62939b01ce1314a0a31a0365`, already removed bare `marlin` from `QUANTIZATION_CHOICES`. The exact candidate preserves that source behavior and adds coverage for both permitted sides of the contract: bare `marlin` is rejected by the CLI and registry, while `gptq_marlin` and `awq_marlin` remain accepted and resolvable.

The candidate commit as a whole fully resolves the original issue because it includes the already-fixed base. The candidate's own delta should be described specifically as test-only hardening, not as the production correction.

## Independent evidence

- At the issue's recorded historical commit `a2ddf92e616c6b47f8a166aece71d3c0bc5976af`, the prepared interpreter imported both relevant modules from `/job/repo/python`. `marlin` was present in CLI choices, absent from the registry, accepted by argparse, and rejected by `get_quantization_config` with `ValueError`. This reproduces the original failure exactly.
- At the recorded prepared base, bare `marlin` was absent from both CLI choices and the registry. `get_quantization_config("marlin")` raised `ValueError`; `gptq_marlin` and `awq_marlin` resolved to `GPTQMarlinConfig` and `AWQMarlinConfig`.
- At the exact candidate commit, its focused suite passed: 4 tests and 2 subtests. An independent probe confirmed parser choices equal `QUANTIZATION_CHOICES`, rejected `marlin`, `Marlin`, and `marlin `, and accepted both explicit Marlin formats.
- The candidate changes only Python tests and retained reports. It changes no Python production source, C++, HIP, CUDA, FlyDSL, or other native source. No native rebuild was applicable.

Raw command output is retained under `reports/j-9b486682f969/raw/`.

## Scope and limitations

This issue is a deterministic CLI/registry metadata contract and does not require tensor execution. The assigned environment is AMD ROCm 7.2 (`torch 2.11.0+rocm7.2`); the source report used NVIDIA CUDA. No GPU execution, model weights, serving process, Marlin kernel, serialized-Marlin checkpoint, multi-node workload, or semantic-accuracy test was used or claimed. Consequently, this review does not validate the broader legacy serialized-Marlin restoration proposed separately in upstream PR 32644. Those paths are not remaining counterexamples to this issue's stated contract, which explicitly permits removing bare `marlin` from the CLI.
