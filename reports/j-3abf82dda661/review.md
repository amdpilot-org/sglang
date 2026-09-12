# Independent review of PR 2541 at `cdc14d94a9d8efbb48b96910176cb11b1f14eb43`

Upstream issue: https://github.com/sgl-project/sglang/issues/26751

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2484

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2551

## Verdict

Recommendation: **accept as a narrow Bug A fix**. The candidate does not fully
resolve the original issue because it intentionally leaves Bug B (scheduler
survivability and per-request isolation) unchanged.

The candidate changes the standard torchvision GPU JPEG call from the default
`ImageReadMode.UNCHANGED` behavior to `ImageReadMode.RGB`. On the recorded base,
a grayscale JPEG decoded through the checked-out `_load_image` control flow as
`(1, 48, 64)`, could not be stacked with an RGB `(3, 48, 64)` tensor, and its
16x16 patches failed an on-device `Linear(768, 1152)` with `12x256 and
768x1152`. At the exact candidate commit, both inputs decoded as three-channel,
stacked, and the same gfx950 projection produced `(12, 1152)`.

This is a source fix plus regression hardening, not merely a test change. It is
also only a partial original-issue fix. The candidate changes neither
`mm_utils.py` nor `scheduler.py`; `Scheduler.run_batch` still has no
request-local catch around `forward_batch_generation`, and an escaping
exception still reaches `run_scheduler_process`, whose broad exception handler
sends `SIGQUIT` to the parent process. No claim of batch or request isolation is
therefore supported by this candidate.

## Independent checks

All commands used the prepared interpreter
`/tmp/amdpilot-repo-j-3abf82dda661/venv/bin/python` and imported SGLang from
`/job/repo/python/sglang`.

* Base reproduction, exact base `358c163250ad3b1f62939b01ce1314a0a31a0365`:
  the substituted decoder boundary recorded calls with only `device='cuda'`,
  grayscale/RGB shapes `(1,48,64)`/`(3,48,64)`, a mixed-stack failure, and the
  gfx950 projection error `mat1 and mat2 shapes cannot be multiplied (12x256
  and 768x1152)`.
* Candidate regression: `python -m pytest -q
  test/registered/unit/multimodal/test_base_processor_image_decode.py` passed
  12 tests.
* Candidate reproduction: the decoder boundary received
  `mode=ImageReadMode.RGB, device='cuda'`; grayscale and RGB both had shape
  `(3,48,64)`, and the on-device projection returned `(12,1152)`.
* Independent boundaries: grayscale, RGB, and CMYK JPEGs all became
  three-channel tensors through candidate `_load_image`; an RGBA PNG still
  became an RGB PIL image through `_load_single_item`.
* The candidate modifies no C++, CUDA/HIP, FlyDSL, or other native source. No
  native rebuild was applicable.

## Environment and architecture limits

The assigned device was reported by PyTorch as `AMD Instinct MI350X`, gfx950,
with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. This torchvision
`0.26.0+rocm7.2` build raises `decode_jpegs_cuda: torchvision not compiled with
nvJPEG support`, so the real standard GPU JPEG decoder could not execute on
this host. Decoder argument propagation and RGB conversion were checked by
substituting torchvision's CPU decoder exactly at the decode boundary; GPU
execution covered the independent Gemma-shaped projection, not nvJPEG.

Gemma-4 weights and the reported 2xH200 TP=2 deployment were unavailable. No
full-model, HTTP-serving, multi-request batch, scheduler subprocess, or
multi-node reproduction is claimed. The tiny Llama fixture is not a substitute
for Gemma-4 vision or a meaningful qualification of the scheduler failure, so
it was not used as proof.

## Remaining counterexamples

1. Any multimodal encoder exception not prevented by RGB JPEG normalization
   can still escape `run_batch` and trigger the existing scheduler-process
   `SIGQUIT` path.
2. A failing item in a mixed request batch is not isolated. The candidate has
   no item-to-request mapping, retry, re-schedule, or selective `FINISH_ABORT`
   behavior.
3. Actual nvJPEG execution on supported NVIDIA hardware, the Gemma-4 model,
   and the reported two-H200 topology remain unverified here.
