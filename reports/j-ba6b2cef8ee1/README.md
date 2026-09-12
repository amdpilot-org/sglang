# DSA top-k v2 tie-overflow investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35257

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1404

The prepared base silently selected the wrong fp32 values when more than 2048
distinct candidates occupied the threshold coarse bin. On the assigned MI350X,
both the register and streaming paths returned 2048 valid, unique indices while
all 2048 selected values differed from `torch.topk`. The non-overflow boundary
and an overflowing exact-tie control were exact, isolating candidate truncation.

This branch carries the progressive exact-key refinement from the already-open
upstream candidate PR https://github.com/sgl-project/sglang/pull/37941. It
re-histograms only overflowing rows in 12/12/8-bit rounds and leaves the common
path and shared-memory size unchanged. The new exactness cases fail before and
pass after; the complete top-k v2 file passes 288 tests on gfx950.

This is a verified candidate, not a complete resolution of the source report.
The CUDA-only cluster implementation still has the original truncation. CUDA
would route the issue's batch-1, 262K B200 fixture through that path, and no CUDA
device or reported model workload was available here.

Raw evidence is retained under
`/tmp/amdpilot-repo-j-ba6b2cef8ee1/logs/`. The loaded post-change JIT library is
`/job/.cache/sglang/jit/gfx950/sgl_kernel_jit_dpsk_v4_topk_v2/build-19843f518534d173/deps-671843af4f1edc96/sgl_kernel_jit_dpsk_v4_topk_v2.so`; its extracted gfx950 code object is
`/tmp/amdpilot-repo-j-ba6b2cef8ee1/topk_v2_after.hsaco`.
