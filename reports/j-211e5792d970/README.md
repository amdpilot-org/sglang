# Independent review of diffusion sleep/wake candidate

Reviewed https://github.com/amdpilot-org/sglang/pull/3065 at exact commit
`d2019b162a993952a2c4728ccad0a1b2b940fc4d` against upstream issue
https://github.com/sgl-project/sglang/issues/19090.

Recommendation: **request changes**. The candidate is a useful partial fix: it
adds the missing local `DiffGenerator` lifecycle/refit surface and fixes the
previously reported transport-exception normalization defect. It does not fully
resolve the original issue's diffusion-model qualification and comparative
benchmark requirements.

The recorded base lacks all three local methods. The candidate's focused suite
passed (13 tests), and independent calls confirmed that `ConnectionError` and
`TimeoutError` become lifecycle-specific `RuntimeError` instances with the
original exception retained as `__cause__`. An additional malformed-response
case found that `None` or an object without `output` still leaks `AttributeError`.

The assigned accelerator was an AMD Instinct MI355X under ROCm 7.2. A synthetic
single-module controller round trip passed on that GPU, but this controller code
already exists in the base and does not exercise a running `DiffGenerator`
transport end to end. No diffusion checkpoint/model weights were available, so
generation, refit, memory behavior, distributed/FSDP behavior, CUDA behavior,
and sleep+wake+refit versus kill-and-relaunch timing remain unverified. FSDP
sleep/wake is explicitly rejected by the implementation.

No native source changed, so no native rebuild was applicable. Imports resolved
to the checked-out source under `/job/repo/python/sglang` using the prepared
interpreter `/tmp/amdpilot-repo-j-211e5792d970/venv/bin/python`.

Raw command summaries are in `raw/`; complete revision-switch evidence was also
preserved outside the checkout at
`/tmp/amdpilot-repo-j-211e5792d970/review-evidence/`.
