# Independent review of PR 3219

Reviewed exact candidate commit `ea369671b9998d6caa4e2888ac53ac7a740d3db6` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original common diffusion-test-entry request.

## Recommendation

Request changes. This is a partial fix, not a full original-issue fix and not merely test-only hardening.

The candidate correctly centralizes the three inherited counterexamples (Ideogram 4, LTX-2.5, and FastH3), scans ordinary unit files, recognizes `_get_config_info`, and recognizes named model assignments. Its four regression tests and 94 directly affected unit tests pass.

However, the production guard still accepts equivalent repetitions. The repository already contains `ideogram-ai/ideogram-4-fp8` in both `server/gpu_cases.py` and `unit/test_diffusion_bcg_padding.py`; an exact call to the candidate helper over those files returns `[]`. The unit occurrence is an executable assertion literal, a form the repeated-ID detector ignores. Independent tests also show that repeated `{"model_path": "..."}` fixtures and positional `DiffusionServerArgs("...")` entries evade the detector.

`unit/test_server_args.py` remains blanket-exempt. Some strings there are intentionally parser/config vectors, but the whole-file exemption also hides executable model-selection inputs. A complete policy needs to distinguish those roles more narrowly.

## Environment and evidence

The prepared interpreter imported `sglang` from `/job/repo/python/sglang/__init__.py`. No C++, CUDA, HIP, or header files differ between base and candidate, so no native rebuild was applicable. No inference code changed; GPU numerical execution and model weights would not validate this AST-policy defect. Ascend/MUSA paths were unavailable on the ROCm host, and the two-GPU serving case exceeded the assigned one-GPU architecture.

Raw base/candidate logs, the independent inventory, and adversarial tests are retained outside the checkout at `/job/review_evidence/j-f77698fc2c1a/`.
