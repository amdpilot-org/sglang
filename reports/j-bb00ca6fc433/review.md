# Independent review of candidate PR 2183

Upstream issue: https://github.com/sgl-project/sglang/issues/32202

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2111

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2225

Candidate commit: `c482d630e9247686887b6683394d999dda60c00b`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the deterministic loader-selection contract in the original issue.

On the exact recorded base, an S3 target and S3 draft with both formats set to `auto` resolved to target `runai_streamer`, draft `auto`, and `DefaultModelLoader`. The assertion that the draft should resolve independently failed. On the exact candidate, the draft resolves to `runai_streamer` and `get_model_loader(LoadConfig(...))` constructs `RunaiModelStreamerLoader`.

The source change is narrowly scoped: `handle_load_format()` now applies the existing object-store draft resolution when the draft format is either unspecified or explicitly `auto`. Explicit non-auto formats remain unchanged. The resolution is stored in the draft-specific field; independent cases confirmed that the target format is not overwritten by draft resolution.

## Independent coverage

The candidate's focused regression suite passed 26 tests. An independent six-case matrix also passed, covering explicit `auto` for `s3://`, `gs://`, and `az://` draft paths; a non-object-store draft; an unspecified draft format; an explicit non-auto draft format; and target/draft format isolation. Loader types were instantiated through the production `LoadConfig` and `get_model_loader` path rather than inferred from prose.

The prepared interpreter imported `sglang` from `/job/repo/python`; the exercised source was therefore the checked-out base or candidate, not an installed SGLang wheel. The candidate changes no C++/HIP/native source, and `repository-environment.json` declares no separate native artifact, so no native rebuild was applicable.

## Limits

This review did not connect to an S3-compatible service or download model weights, so object transfer and a full server launch were not tested. It did not reproduce the reporter's NVIDIA H20, two-node, TP=8, PP=2 deployment. The assigned device is one AMD Instinct MI355X (`gfx950`) with Torch `2.11.0+rocm7.2`; it was detected but not used because the issue is completely determined during pre-hardware argument resolution and loader dispatch. A GPU startup smoke would not add evidence for that contract.

Raw command output was preserved outside the revision-switching checkout at `/job/review-evidence-j-bb00ca6fc433/`.
