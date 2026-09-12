# Independent review of amdpilot-org/sglang PR 1226

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/1226 at exact commit `38949e27a717f57a3563432f49d7f5886ad6030f`.

Upstream issue: https://github.com/sgl-project/sglang/issues/36500

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1258

## Recommendation

Accept. The candidate fully resolves the original API lifecycle contract by implementing the issue's explicitly permitted rejection behavior: removal of the same corpus ID returns `success=False` while that corpus has an unconsumed pending load. It leaves removal of a different corpus during the load and removal after load completion intact.

This is a source fix with regression coverage, not test-only hardening. The recorded base reproduced the defect deterministically: the candidate regression's same-ID overlap assertion failed because `remove()` returned `success=True`. At the exact candidate commit, its three tests passed. Three independent adversarial cases also passed, including a completed worker thread whose result had not yet been polled/committed, a failed pending load, and different-ID removal during a pending load. The existing 42-test NGRAM corpus suite passed.

The interpreter imported `ExternalCorpusManager` from `/job/repo/python/sglang/srt/speculative/external_corpus_manager.py` while the candidate commit was checked out. No C++, HIP, CUDA, FlyDSL, or other native source changed, so no native rebuild was applicable.

## Limitations and hygiene

The original Qwen3-4B HTTP reproduction was not run because the prepared environment did not provide those weights and differs architecturally from the report: one AMD Instinct MI350X (gfx950 family), ROCm 7.2, and PyTorch 2.11.0+rocm7.2 versus the reporter's NVIDIA L40S/CUDA environment. No GPU execution is needed for this Python scheduler lifecycle transition, and no unrelated GPU smoke is presented as proof. The qualified tiny Llama fixture would validate transport/engine execution but not strengthen the directly synchronized lifecycle evidence.

`git diff --check` over the candidate is nonzero only because its archived report log files contain trailing whitespace. The source and test behavior reviewed here are correct; this is nonfunctional report-artifact hygiene rather than a remaining counterexample to the issue contract.

An open upstream related change, https://github.com/sgl-project/sglang/pull/36517, implements the same rejection strategy. Its CI state was not treated as proof of this candidate.
