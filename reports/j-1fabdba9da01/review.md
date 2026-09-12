# Independent review of amdpilot-org/sglang PR 1288

Reviewed exact candidate commit `57422568856af1b282c5313f81b35aa5bc4e01a2`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and
upstream issue https://github.com/sgl-project/sglang/issues/36010.

## Finding

Request changes. The candidate is diagnostic hardening, not a fix for the
original issue. It checks the three CPU metadata inputs and changes the
observed `TypeError` into an actionable `RuntimeError`. It does not populate
the missing metadata, move workspace preparation outside graph capture, or
provide a GPU-resident/native FP4 attention path. Consequently both original
contracts remain broken: graph-off speculative verification still terminates
on its first affected request, and graph capture still cannot safely select
request-dependent pages on replay.

The candidate's own test accurately proves the exception translation, but it
does not exercise a speculative batch, FlashInfer, NVFP4 data, graph capture,
or successful attention output. Calling that test a regression for the full
issue would therefore overstate its evidence.

## Evidence

- On the recorded base, the candidate test produces the original three
  unhandled `None` failures, including the reported
  `extend_prefix_lens_cpu[i]` subscript. See
  `/job/review-evidence-j-1fabdba9da01/base-regression.log`.
- On the exact candidate, the same test passes because all three missing
  values are rejected by the newly added guard. See
  `/job/review-evidence-j-1fabdba9da01/candidate-regression.log`.
- Source inspection confirms `ForwardBatch.init_new` intentionally leaves the
  CPU mirrors unset for the GPU-only path, while
  `FlashInferAttnBackend.forward_extend` still passes them into
  `get_flashinfer_dequant_workspace_kv_buffer`. The candidate changes neither
  path.
- The prepared interpreter imported
  `/job/repo/python/sglang/srt/mem_cache/memory_pool.py`, so the checked-out
  candidate source rather than an unrelated installed SGLang copy was tested.
- No C++/CUDA/HIP/native source changed in the candidate, so no native rebuild
  was applicable. The imported AITER extension was the prepared environment's
  `/tmp/amdpilot-repo-j-1fabdba9da01/cache/aiter/module_aiter_core.so`; it was
  not evidence for this CUDA-only FP4 path.

## Environment limitations

The assigned GPU is AMD Instinct MI355X (`gfx950`) with Torch
`2.11.0+rocm7.2`, not the reported RTX 5090 (`SM120`) CUDA system. The
Qwen3.8-27B-NVFP4 weights and NVIDIA FlashInfer/TRTLLM NVFP4 execution path
were not available. Thus no full-model serving, CUDA graph, or numerical NVFP4
GPU claim is made. Open upstream PRs 36038 and 36045 contain substantially
different architecture-specific proposed solutions; neither is part of the
reviewed candidate.
