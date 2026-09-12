# Independent review of PR 2808

Candidate: https://github.com/amdpilot-org/sglang/pull/2808 at `8c799873170db6688e35734e591e36a5f188eaa5`

Upstream issue: https://github.com/sgl-project/sglang/issues/32485

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2840

Candidate provenance issue: https://github.com/amdpilot-org/sglang/issues/2741

## Verdict

Request changes. The candidate fixes the main reported path, but it does not fully satisfy the safe-fallback contract.

On the recorded base, an independent capture of the argument passed to `runai_safetensors_weights_iterator` showed that both required layouts passed every target and draft shard. At the candidate commit, the same capture showed that both native `mtp.<index>.*` and Hugging Face `model.mtp.layers.<index>.*` layouts passed only the requested layer shard and draft-wide shared shard. The candidate's own focused suite also passed.

The remaining counterexample uses the existing `maybe_add_mtp_safetensors` compatibility path. For a supported GLM4Moe configuration, that function detects an on-disk `mtp.safetensors` missing from the index and deliberately appends it because the checkpoint index is incomplete. If the index also contains a recognized requested-layer entry, the candidate's new selector considers selection safe and removes the just-added supplemental file. The exact list handed to RunAI therefore omits `mtp.safetensors`. This can turn a previously loadable checkpoint into a partial load and conflicts with the issue requirement to fall back safely when index metadata cannot completely describe required draft weights.

## Environment and paths

- Prepared base and prepared branch before and after review: `358c163250ad3b1f62939b01ce1314a0a31a0365`, `amdpilot/j-af0f87f992fe`.
- Candidate was temporarily checked out detached at the exact requested commit.
- Interpreter: `/tmp/amdpilot-repo-j-af0f87f992fe/venv/bin/python`.
- Imported loader source under both revisions: `/job/repo/python/sglang/srt/model_loader/loader.py`.
- Candidate files changed: Python loader, Python unit test, and reports only. There were no native changes, so no native rebuild was applicable.
- The pinned environment reports Torch `2.11.0+rocm7.2` and HIP `7.2`.

## Evidence retained outside the checkout

Raw command output was retained in `/tmp/amdpilot-repo-j-af0f87f992fe/review-evidence/` while switching revisions:

- `base-reproduction.txt`
- `candidate-import-path.txt`
- `candidate-regression.txt`
- `candidate-independent.txt`
- `candidate-adversarial-unindexed-mtp.txt`
- `upstream-issue.json`

The independent review scripts are `/job/base_candidate_review.py` and `/job/candidate_adversarial.py`.

## Architecture and validation limits

This change is Python metadata and file-list selection, not numerical model execution. GPU numerical validation and native compilation do not measure the affected contract. No live object-store credentials or large MTP checkpoint were available, so the review verified exact pre-RunAI path lists, cached-index behavior in the candidate tests, and the real local compatibility call order. Live remote header-read counts, distributed loading, and end-to-end model semantics remain unverified.
