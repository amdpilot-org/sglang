# Independent review of amdpilot-org/sglang PR 1247

Candidate reviewed: `6b37f294aa085347e44d747b235dc4987c747823`

Upstream issue: https://github.com/sgl-project/sglang/issues/36495

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1281

## Finding

Request changes: `validate_ngram_capacity()` uses `assert` for user-input validation. Python removes that statement under `python -O`, and the invalid NGRAM `capacity=4, max_trie_depth=4` configuration is then accepted by the server-argument validation layer. The independent command recorded in `raw/candidate_optimized_validation.log` exits 0 and prints `OPTIMIZED_ACCEPTED`.

The candidate's rebuilt native constructor guard still prevents the original invalid free before it creates the insertion thread, so this is a partial fix rather than a test-only change. However, it does not robustly provide the claimed early serving-argument validation in all supported Python execution modes. Replace the assertion with an explicit exception and test the actual validation entry point under optimized Python.

## Evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a fresh JIT build of the production `NgramCorpus` path constructed the 4/4 corpus, queued one four-token insertion, and aborted with `munmap_chunk(): invalid pointer` (exit 134). The log and built library path are in `raw/capacity_equal.log` and `raw/native_paths.txt`.

At the exact candidate commit, another fresh JIT build rejected capacities 4/4 and 3/4 synchronously with `RuntimeError`; capacities 5/4 and 3/2 both completed an insertion. The full native-backed corpus suite passed 45 tests, and the candidate server-argument test file passed 4 tests. `git diff --check` also reports trailing whitespace in two committed raw test logs; this is non-functional.

## Environment limitation

The prepared host has one AMD Instinct MI350X (`gfx950`), ROCm 7.2, and Torch 2.11.0+rocm7.2. The original report used an NVIDIA L40S, CUDA, and Qwen3-4B. Those weights and architecture were unavailable, and NGRAM serving in this checkout explicitly supports CUDA or CPU rather than ROCm. Therefore no full HTTP/model reproduction or GPU execution is claimed. The defect was reproduced and the candidate checked through the same production CPU-only C++ trie JIT source and newly built shared libraries.
