# Independent review of candidate PR 2955

- Upstream issue: https://github.com/sgl-project/sglang/issues/18891
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2886
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2989
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Candidate reviewed: `9d08a826d6e9149cb8df17ea3c91ece7d550dc6b`
- Recommendation: `request_changes`

## Verdict

The candidate is a meaningful partial implementation. It adds the requested general HTTP protocol and correctly handles ordinary parameters, renamed parameters, and fused QKV parameters in focused tests. An independent GPU check also confirmed exact disk/live parity and detection of a one-value mutation.

It does not fully resolve the original contract because the protocol can return a false successful comparison:

1. `compare_module_weights_with_disk` builds both checksums exclusively from `module.named_parameters()` and silently skips checkpoint buffers (candidate `weights_updater.py:379-387`). A VAE-like checkpoint containing a live-state buffer can therefore differ from the server and still return `match: true`. The independent counterexample used matching `weight` values and differing `running_mean` values; the function logged that `running_mean` was skipped and returned matching checksums. This matters to the original VAE-focused request and its explicit discussion of running state. Runtime-only state absent from disk should be excluded, but checkpoint-present state must not be silently omitted if the endpoint claims disk/server parity.
2. `module_names=[]` passes request validation and produces `success: true`, `modules: {}`, and “All requested modules match the checkpoint” (candidate `weights_api.py:170-178`, `gpu_worker_post_training_mixin.py:180-213`). This is another vacuous-success path. An explicit empty selection should be rejected or have clearly non-success semantics.

The candidate's own regression suite does not cover either case. Accordingly, this is a partial fix with useful test hardening, not a verified full original-issue fix.

## Reproduction and validation

The prepared checkout was clean and exactly at the recorded base. On that base, raw checkpoint Q/K/V hashing differed from the loaded fused-QKV checksum and `/compare_weights_with_disk` was absent, reproducing the reported gap.

At the exact candidate commit, the candidate's focused regression command passed 4 tests. Independent CPU adversarial cases reproduced both false-success paths above. An independent GPU numerical check passed on one AMD Instinct MI355X using Torch 2.11.0+rocm7.2: the loaded GPU tensor equaled a separately constructed CPU reference, checksums matched before mutation, and diverged after one GPU value changed.

No C++ or other native source changed in the candidate, so no native rebuild was applicable. Imports resolved from `/job/repo/python`; the interpreter was `/tmp/amdpilot-repo-j-6995d3041885/venv/bin/python`; AITer loaded from `/tmp/amdpilot-repo-j-6995d3041885/cache/aiter/module_aiter_core.so`.

## Limitations

- The large FLUX.2-klein and Qwen-Image model-pair integration suites were unavailable because their weights were not prepared locally.
- The one-GPU assignment prevented multi-rank TP, SP, and DTensor validation.
- The deterministic tiny Llama fixture is not a diffusion/VAE/text-encoder architecture and therefore was not used as proof of this issue.
- Candidate evidence reported an MI350X, while this independent review ran on the currently assigned MI355X.
