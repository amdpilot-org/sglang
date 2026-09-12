# Independent review of candidate 32cfaf2

Upstream issue: https://github.com/sgl-project/sglang/issues/36841

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3396

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3413

Recommendation: **accept**. The candidate fully resolves the reported DOTS-3 deprecated dtype getter on the pinned Transformers 5.12.1 stack.

The prepared checkout was exactly the requested failing-before commit `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image/prepared revision difference. The candidate commit `32cfaf2e45c6fe1b120ecacbd92f254c04f0f9ab` has that commit as its sole parent.

On the base, the candidate's preserved regression failed 4/4 cases because constructing `Dots3MoE` read `config.torch_dtype`, causing Transformers' `warning_once` to receive `` `torch_dtype` is deprecated! Use `dtype` instead! ``. The same execution had already propagated a value identical to `config.dtype`, isolating the defect to the deprecated read.

At the exact candidate, the candidate regression passed 4/4 cases. An independent test passed 8/8 cases using both `dtype=` and compatibility `torch_dtype=` configuration inputs across the default, float16, bfloat16, and float32. It exercised the real `Dots3Config`, `Dots3MoE`, and `MaybeTboDeepEPDispatcher`; only the heavyweight `DeepEPDispatcher` endpoint was replaced with a capture class. Every captured `params_dtype` was identical to `config.dtype`, and no deprecated getter warning occurred.

The imported paths were `/job/repo/python/sglang/srt/configs/dots3.py`, `/job/repo/python/sglang/srt/models/dots3_common/modeling.py`, and `/job/repo/python/sglang/srt/batch_overlap/two_batch_overlap.py`. The interpreter used Transformers 5.12.1 and Torch 2.11.0+rocm7.2. One AMD Instinct MI350X was visible. No native source changed and the prepared environment declares no native build target, so no native rebuild was applicable.

Limitations: no DOTS-3 weights were available, so full model loading and distributed DeepEP transport were not run. This does not leave a counterexample to the narrow original contract, which is the deprecated host-side config getter. Future Transformers removal behavior was not executed or claimed.

Full transient test output and the independent test source were preserved outside the revision-switching checkout at `/job/review-evidence-j-eb126b5f6a93/` during review.
