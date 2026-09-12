# Independent review of PR 1394

Reviewed exact commit `e68b4de088fed8fc8d77882676dac6b8862e4d37` against upstream issue https://github.com/sgl-project/sglang/issues/36010 and mirror issue https://github.com/amdpilot-org/sglang/issues/1430.

Recommendation: **unverified**. The source change is materially aimed at the original contract: it routes NVFP4 speculative verify/draft-extend away from FlashInfer's host-dependent dequant workspace and into TRTLLM MHA's packed FP4 decode path. It also retains an explicit guard if the unsafe FlashInfer route is reached. The candidate's mocked regression passes, while the recorded base cannot collect it because the new resolver is absent.

This is stronger than test-only hardening, but it is not independently proven as a full original-issue fix. On the assigned AMD MI355X/gfx950 ROCm environment, both kernel-level tests skip. The original SM120 model, CUDA graph capture/replay, successful first request, numerical FP4 attention, and external draft-model serving could not be exercised. The candidate changes Python only; no native rebuild was applicable.

Raw logs and the reviewed source diff were preserved outside the checkout at `/tmp/amdpilot-repo-j-a1e41ca4093c/review-evidence/` while revisions were switched.
