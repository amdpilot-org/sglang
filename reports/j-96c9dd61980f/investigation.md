# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/36830

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1078

The prepared base still rejected every CUDA TileLang + FP8 KV combination in
`_check_dsa_backend_constraints`, selected the scaled 528-byte CUDA pool row,
wrote raw rows only on HIP, and dispatched `sparse_mla_fwd_decode_partial_fp8`
only on HIP. Thus the reported model had no usable SM90 FP8 route when KPool
excluded `flashmla_kv`.

Upstream PR https://github.com/sgl-project/sglang/pull/36904 identified that the
generic TileLang FP8 kernel already existed and supplied the missing plumbing.
The issue reporter subsequently recorded successful validation of that route on
the original 8xH20/SM90 class: 4,922,240 FP8 tokens versus 2,742,144 BF16 tokens
in approximately the same 32.5 GB, with identical final answers for three
temperature-zero spot checks. The PR closed without merging, and the prepared
base retained the pre-fix gates.

This change ports that narrow solution to the current refactored implementation.
It intentionally requires both local DSA consumers to be TileLang so every
consumer agrees on the raw layout. It also rejects DCP because the raw fused
writer has no DCP rank filtering, and rejects CUDA architectures below SM89.

The assigned GPU was an AMD Instinct MI355X (`torch.cuda.get_device_capability`
reported `(9, 5)`, HIP `7.2.26015`). A one-hot raw-FP8 TileLang probe reached
compilation, then the prepared TileLang/TVM stack raised `InternalError: Check
failed: pb->value != 0 (0 vs. 0): Divide by zero` in layout inference. No kernel
launched, so this is recorded as a GPU/toolchain limitation rather than a
numerical validation or rejection of the CUDA candidate.
