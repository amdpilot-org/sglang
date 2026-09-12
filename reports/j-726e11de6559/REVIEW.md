# Independent review of PR 3138

Upstream issue: https://github.com/sgl-project/sglang/issues/32928

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3141

Candidate: https://github.com/amdpilot-org/sglang/pull/3138 at `ac2aa3d000cddf8be855362a8254df957172a3cb`.

Recommendation: `request_changes`. The patch is a partial correction, not a full original-issue verification.

The prepared base has no `sglang.srt.distributed.nccl_ras` module. At the prior candidate `f924d21cd65995706d9180fe289f088ec3ef8e2b`, an independent real-Prometheus sequence reproduced the stale divergence: after publishing `AllReduce=7`, a successful recovered snapshot that omitted `AllReduce` still exported it.

At the reviewed commit, all 36 focused tests plus 2 subtests pass. Checked-out source imports resolve under `/job/repo/python`. Independent cases show that a normal registry removes recovered collective and disappeared communicator labelsets, a failed poll preserves the last snapshot, and a communicator can disappear and reappear with only current labels.

The correction remains partial in Prometheus multiprocess mode. With `PROMETHEUS_MULTIPROC_DIR` set before import, the implementation deliberately sets obsolete series to zero instead of removing them. Thus the stale nonzero alert is fixed, but recovered collective and disappeared communicator series remain exported. This is the prior review's physical-series counterexample, and the candidate's regression checks for those zero-valued ghost series rather than their absence.

No C/C++/CUDA/HIP source changed, so native rebuilding was not applicable. The host exposes one AMD Instinct MI350X through Torch 2.11.0+rocm7.2 and RCCL API compatibility 2.27.7. It cannot qualify live NVIDIA NCCL 2.28.7+ STATUS polling, two-node unresponsive/dead transitions, or TP=2 serving metrics. No unrelated startup smoke was substituted for those paths.

Raw command outputs were retained outside the revision-switched checkout at `/job/review-evidence-j-726e11de6559/raw/` during review.
