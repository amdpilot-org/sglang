# CPU OffloaderV2 / DeepEP expert-count investigation

The base source at `358c163250ad3b1f62939b01ce1314a0a31a0365` contains the reported defect. `deep_gemm.py` unconditionally calls `copy_list_to_gpu_no_ce` when OffloaderV2 forbids copy-engine use, while `copy.cu` accepts only 32, 64, and 72 elements. A 256-expert model at EP16 supplies 16 local counts, so the checked-in dispatch reaches `TORCH_CHECK(false, "unexpected N")`.

The fix keeps the transfer off the copy engine. It adds the reported 16-element specialization and a bounded by-value fallback for other positive sizes through 512. Existing compact specializations remain in place. Tests cover the reported size, an independently unsupported size, all prior specializations, and both fallback boundaries.

The prepared ROCm wheel does not expose this CUDA-oriented operator. To validate the changed native source rather than that wheel, the source was built as a focused Torch extension with the pinned ROCm toolchain and executed on the assigned AMD Instinct MI355X (`gfx950`). Exact integer copies passed for N=16, 17, 32, 64, 72, and 512; N=0 and N=513 produced the expected errors. Full compiler and runtime output is retained at `/tmp/amdpilot-repo-j-3ab50de5160a/evidence/native-build-and-gpu-test.log`.

The full GLM-5.2, NVIDIA H800, two-node TP16/EP16 DeepEP serving workload was not available and is not claimed as reproduced. See `result.json` for the complete test record and limitations.
