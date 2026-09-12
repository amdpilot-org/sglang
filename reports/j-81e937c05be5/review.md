# Independent review of PR 2802

Candidate: `252657c6ccfb8d8cbb37598a4fe5f4592397c111`

Upstream issue: https://github.com/sgl-project/sglang/issues/31894

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2756

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2835

## Finding

Recommendation: **accept**, as a correct partial fix. The candidate removes the
reported unconditional pinned-memory allocations from every LoRA layer offset
site and from `TorchNativeLoRABackend.prepare_lora_batch`, using the platform's
device-aware `is_pin_memory_available()` policy. On the CPU platform that policy
returns `False`.

This does **not** establish that the complete original deployment works. The
candidate's retained serving evidence reaches model execution but its request
then fails in the CPU RoPE path with
`NameError: apply_rope_with_cos_sin_cache_inplace`. Neither this review nor the
candidate ran Qwen3-VL-8B-Instruct plus the user's adapter on an accelerator-free
Intel Xeon/AMX host. The tiny Llama startup evidence cannot qualify that model,
multimodal correctness, AMX behavior, or successful HTTP generation.

## Independent evidence

The prepared checkout exactly matched the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`; the candidate has that commit as its
sole parent. Imports resolved to `/job/repo/python/sglang/...`, while Torch came
from the pinned `/opt/venv` installation (`2.11.0+rocm7.2`).

On the base, constructing a CPU `TorchNativeLoRABackend` and preparing a decode
batch while making any `Tensor.pin_memory()` call fail reproduced the original
class of failure at `torch_backend.py:225` with `RuntimeError: no pinned memory
allocator is available` (exit 1).

At the exact candidate commit:

- `python -m pytest -q test/registered/unit/lora/test_torch_native_cpu.py`:
  3 passed (exit 0). This includes an adversarial allocator failure, CPU metadata
  checks, layer-offset checks, and an exact independent two-matmul CPU reference.
- `python -m pytest -q test/registered/unit/lora`: 82 passed, 6 skipped, 14
  subtests passed (exit 0).
- `git diff --check BASE..CANDIDATE`: exit 0.

Raw command output was preserved outside revision switches under
`/job/review-evidence/` during the review.

## Scope classification

This is a source fix plus regression hardening, not test-only hardening. It fixes
the concrete reported pin-memory crash paths. It is only a partial validation of
the broader feature request because successful CPU-engine LoRA serving on the
reported architecture/model remains unverified and the available tiny-Llama
request encountered a separate runtime failure.

No native source changed, so no native rebuild was required or performed. The
review host is an AMD ROCm system with an available GPU, not an accelerator-free
Intel Xeon AMX node. No model weights or serving processes were launched during
this review.
