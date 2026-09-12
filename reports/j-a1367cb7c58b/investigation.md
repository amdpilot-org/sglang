# Independent review of PR 972 at `7764c2a5b91022ed2ad96e9f60bad81d6276dfae`

Upstream issue: https://github.com/sgl-project/sglang/issues/38821

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1006

Candidate: https://github.com/amdpilot-org/sglang/pull/972

## Finding

Recommendation: **request changes**. The candidate is a useful partial fix, but it
does not establish that it resolves the original semantic misidentification.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the GLM-5.3
custom processor is unregistered and
`sglang.srt.configs.glm5_next_processing` does not exist. On the exact candidate,
imports resolve from the checkout and all nine committed processor tests pass.
This verifies processor registration, synthetic JPEG decoding/preprocessing, resize
boundaries, and the reviewed single expanded-token-span correction.

That correction is broader than its evidence: it collapses every consecutive run
of image token IDs without using media cardinality or image-span boundaries. An
independent pretokenized two-image boundary case supplies two adjacent placeholders
and two image items; the candidate silently passes only one placeholder to
`load_mm_data`. The committed tests cover two spans only when another token
separates them.

More importantly, neither the candidate nor this environment verifies the original
reported semantic behavior. The report says the deployed model processed an image
and reasoned about a bird. Registration of a missing processor and placeholder
normalization do not, by themselves, explain that already-visual path. The exact
JPEG, private GLM-5.3-Flash weights, eight NVIDIA H20 GPUs, CUDA runtime, and the
reported deployment were unavailable, so Statue of Liberty landmark semantics
remain unverified.

## Evidence

- `raw/base_original_path_failure.txt`: recorded base prints no registered custom
  processor and fails importing the candidate compatibility module.
- `raw/candidate_regression.txt`: exact candidate's committed suite: 9 passed.
- `raw/candidate_adversarial.txt`: exact candidate collapses two adjacent image
  placeholders to one before `load_mm_data`: 1 failed.
- `raw/base_import_paths.txt` and `raw/candidate_import_paths.txt`: SGLang and the
  added processor load from `/job/repo/python`; Transformers 5.12.1, Torch
  2.11.0+rocm7.2, and the installed `sgl_kernel` load from the prepared environment.
- `raw/gpu_inventory.txt`: one AMD Instinct MI350X/gfx950 was visible.

No native source changed, so no native rebuild was applicable. No GPU model
inference was run: the available AMD GPU cannot substitute for the reported 8x H20
architecture, and the private model weights and exact image were unavailable.

