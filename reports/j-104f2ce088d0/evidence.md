# Independent review evidence

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1203 at exact commit `99a71d67f2b73b752aa095b175c867bd8c50beba`.

Original reports:

- Upstream issue: https://github.com/sgl-project/sglang/issues/36427
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1236

## Revision and source checks

- The image-prepared checkout was clean and exactly at the recorded base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no prepared/base difference.
- The candidate's merge base with the recorded base is that same commit.
- The candidate changes one MDX page, adds a documentation-content test, and adds its prior investigation report. It changes no Python, C/C++, HIP, CUDA, or other native implementation source.
- Under the prepared interpreter, `sglang` imported from `/job/repo/python/sglang/__init__.py` and `model_path_hook` imported from `/job/repo/python/sglang/srt/arg_groups/model_path_hook.py`. Torch imported from `/opt/venv/lib/python3.12/site-packages/torch/__init__.py` as `2.11.0+rocm7.2`; `torch_npu` was unavailable.
- No native rebuild was applicable because the candidate has no native changes.

## Failing-before reproduction

On the prepared base, a direct issue-contract check confirmed that all three requested documentation elements were absent: the TorchNPU release/package mapping explanation, `--disable-cuda-graph` fallback, and `SGLANG_USE_MODELSCOPE=true` launch guidance. The check exited successfully only after asserting all three values were false. Raw output is retained at `/job/base-reproduction.log`.

## Candidate verification

- Candidate regression: 3 passed in 0.43 seconds. This proves the requested strings are present, but by itself is only a literal-content test.
- Existing ModelScope path suite: 12 passed in 25.89 seconds, exercising disabled mode, cache lookup, downloads, tokenizer paths, and draft-model paths against checkout source.
- `git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365..99a71d67f2b73b752aa095b175c867bd8c50beba` passed.
- Independent CLI inspection showed `--disable-cuda-graph` is registered and maps to disabling both graph backends, although it is currently deprecated in favor of the phase-specific graph backend flags.
- Independent source inspection found the Ascend `NPUGraphRunner` and `NPUCudaGraphBackend`, supporting the candidate's qualified statement that graph capture is supported rather than universally disabled.
- `python/pyproject_npu.toml` includes unpinned `modelscope`, and the serving hook invokes `handle_modelscope_paths` when `SGLANG_USE_MODELSCOPE` is true. With ModelScope 1.39.1, the candidate's exact `modelscope download --model Qwen/Qwen2.5-7B-Instruct` syntax was accepted and began resolving/downloading the 15-file model; it was terminated promptly to avoid downloading multi-gigabyte weights. No completion or model execution is claimed.
- The candidate explicitly presents eager execution as a troubleshooting fallback and warns of reduced performance; it does not claim an automatic runtime fix.

Raw candidate outputs are retained outside the checkout at `/job/candidate-tests.log`, `/job/import-paths.log`, `/job/modelscope-exact.stderr`, and `/job/mint-validate.log`.

## Environment and architecture limits

- The assigned accelerator is AMD gfx950, while the issue concerns Ascend NPU. There are no `/dev/davinci*` devices, CANN, or `torch_npu` in the prepared environment.
- The issue supplies no model/operator-specific graph-capture failure, so the original Ascend startup failure could not be reproduced or attributed. This does not prevent verifying the requested qualified documentation fallback, but it prevents claiming an Ascend runtime defect was fixed.
- No GPU execution was relevant to the documentation-only candidate, and none is claimed.
- A Mintlify validation attempt could not start because the fetched `mintlify` executable returned `Permission denied` (exit 127). The page was therefore checked by diff/content inspection rather than a rendered site build.
- A full ModelScope model download and server launch were intentionally not performed; those would not validate Ascend graph behavior on this AMD host.

## Review conclusion

Recommendation: accept. The candidate fully resolves the original issue as posed: it disambiguates both TorchNPU release/package pairs, adds a qualified graph-capture fallback without falsely changing the default, and documents a working ModelScope channel plus local-path use. Runtime Ascend graph compatibility for any particular unspecified model remains outside what this documentation issue and available hardware can establish.
