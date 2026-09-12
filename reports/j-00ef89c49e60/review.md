# Independent review of amdpilot-org/sglang PR 2036

Candidate reviewed: `aed3cae082a1786ded12e8d4e7d1f35c93e52807`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate is a partial fix, not a full resolution of the original shared-mount isolation bug.

## Findings

1. **Blocking: scoped clear can still delete another deployment's files.** The candidate's `_name_matches_suffix()` accepts the requested suffix anywhere in a filename when the following text is empty or starts with `_`. For a clearing deployment with suffix `_model_0_1`, a different valid deployment named `tenant_model` writes keys ending in `_tenant_model_0_1`. The requested suffix occurs at the end of that other deployment's filename, so the candidate deletes it. The independent adversarial test failed with `AssertionError: clear crossed deployment scope into model tenant_model`. This violates the original contract that clear only removes files belonging to the issuing instance.

2. The candidate does improve behavior for unrelated files, ordinary distinct model names, rank-size boundary cases such as `_model_0_10`, missing scopes, and the degenerate `_` scope. Its own three focused tests pass. Those tests do not cover a suffix that is the tail of another valid model name.

3. The recorded base reproduces the original failure directly through `NixlFileManager` and production bucket routing: the calling instance file, another deployment file, an overlapping-name deployment file, a rank-size boundary file, and an unrelated file were all removed.

## Validation and limitations

- Imports resolved to `/job/repo/python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py` and `hicache_nixl.py` while detached at the exact candidate commit. The observed signature was `clear(self, suffix: Optional[str] = None)`.
- The diff changes Python only; there is no native/C++ change and no native rebuild is applicable.
- Environment: Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015, one AMD Instinct MI350X (gfx950-class, reported capability `(9, 5)`). No GPU execution was used because the defect and tests are CPU filesystem operations.
- No full HTTP server, real NIXL plugin, model weights, multi-rank race, or distributed workload was exercised. The production file manager and bucket routing were exercised directly, which is sufficient to demonstrate both the original deletion behavior and the remaining filename-matching counterexample, but not serving-stack integration.
- Upstream PR 32705 was inspected. It uses the same occurrence-plus-trailing-boundary matching concept and does not remove this counterexample from the candidate under review.

## Evidence

- `raw/original-failure.log`: recorded-base reproduction.
- `raw/candidate-regression.log`: candidate's three tests passing.
- `raw/adversarial-test.log`: independent isolation counterexample failing on the exact candidate.
- `raw/import-paths.log`: checked-out source import paths and method signature.
- `raw/environment.log`: measured interpreter, ROCm, and device information.

