# NVFP4 speculative decoding correction

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1288 at
`57422568856af1b282c5313f81b35aa5bc4e01a2`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1340.

The candidate's guard is useful diagnostic hardening, but an exact-candidate
call with GPU-only-style missing prefix metadata still exits with its new
`RuntimeError`; it does not serve the request. Its test also invokes the
workspace helper directly rather than constructing speculative attention
metadata. Raw evidence is in `candidate-target-verify-failure.log` and
`candidate-exception-only.log`.

This correction ports the native NVFP4 speculative decode route developed in
https://github.com/sgl-project/sglang/pull/36038 onto the prepared main base.
Target verify and both draft-extend forms are routed through TRTLLM MHA instead
of FlashInfer's host-prepared extend workspace. Packed FP4 KV and block scales
are passed to the native decode API, and fixed/ragged speculative requests get
the required bit-packed XQA causal mask. Configuration resolution requires the
FlashInfer-prefill/TRTLLM-decode pair and selects decode-mode speculative
attention, covering the external draft worker as well as integrated MTP.

The candidate's actionable missing-host-metadata guard is retained as a
backstop if a speculative phase is misrouted into the FlashInfer workspace.

The dedicated regression constructs target-verify, DRAFT_EXTEND_V2, and legacy
draft-extend batches and checks native packed KV/scales, routing, fixed and
ragged masks, and backend configuration. It passes after the correction.

The assigned device is gfx950/ROCm, not SM120/CUDA, and the reported model
weights are unavailable. No local full-model, CUDA graph, FlashInfer NVFP4, or
TRTLLM NVFP4 numerical execution claim is made. The source correction is based
on focused host regressions plus the parent implementation's recorded SM120
end-to-end evidence; that external evidence was inspected, not rerun here.
