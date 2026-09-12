# Independent review of candidate PR 881

Upstream issue: https://github.com/sgl-project/sglang/issues/38029

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/818

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3214

Candidate: https://github.com/amdpilot-org/sglang/pull/881 at exact commit `399f520f72ea5ad85c33546a60c317f2c42759f8`

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365` (no difference).

## Recommendation

Accept. The candidate is a narrow source fix, not merely test hardening: `DeepseekSparseAttnBackend.__init__()` now copies the owning runner's `kv_index_translator`. This covers both kinds of DSA backend that EAGLE constructs after `ModelRunner.init_attention_backends()` has completed its generic bind pass: the draft-extend backend and each child of the multi-step draft-decode backend.

The exact candidate regression was independently run against both revisions. On the untouched base, both tests failed because the backend translator was `None`. At the candidate commit, both passed on the assigned gfx950 GPU using the repository's real FP8 DSA token pool, including non-empty static-pool prefix IDs.

An additional adversarial check replaced the runner translator with a non-identity recording implementation. The candidate preserved that exact object across a draft-extend backend and three draft-decode children, and a GPU read-index call returned the translator's deliberately shifted result. This guards against only proving the static identity case.

## Source and native path verification

Imports resolved to `/job/repo/python/sglang` and specifically `/job/repo/python/sglang/srt/layers/attention/dsa_backend.py`, so the checked-out source was tested rather than an installed copy. The candidate changes only Python, tests, and reports; it changes no C++, HIP, FlyDSL, or other native source. A native rebuild was therefore not applicable. The prepared AITER native module loaded from the private runtime cache at `/tmp/amdpilot-repo-j-fc0804744b49/cache/aiter/module_aiter_core.so`.

## Limitations and ancillary observations

The original four-GPU TP/EP GLM-5.3 EAGLE HTTP workload was not runnable: only one AMD Instinct MI355X (`gfx950`) GPU was assigned, and the GLM-5.3 weights/configuration were unavailable. The evidence qualifies the concrete failing lifecycle, translator binding, FP8 DSA pool allocation, and read-index behavior, but not full-model semantic accuracy or distributed execution.

An existing CUDA DSA indexer test was attempted as a compatibility check, but failed before constructing the backend because this environment's `dsa_indexer` module lacks the optional `deep_gemm` attribute that the test patches. This is not evidence of a candidate regression. Candidate-wide `git diff --check` also reports trailing whitespace in the candidate's committed raw pytest logs; the source and test code themselves compile, and this evidence-formatting issue does not affect the fix.

No remaining counterexample tied to the reported contract was found.
