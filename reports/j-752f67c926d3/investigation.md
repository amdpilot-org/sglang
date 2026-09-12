# DeepGEMM standard pre-permute ownership investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32056

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2156

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) already contains the
reported fix in `pre_permute_standard_to_deep_gemm`: both the masked-layout
and compact-layout branches call `dispose_tensor(hidden_states_ref)` only when
`runner_config.inplace` is true. The upstream issue links open PR
https://github.com/sgl-project/sglang/pull/32057, whose patch introduces the
same guard for the original compact-layout code.

A regression test was added because the linked upstream PR did not include
one. It executes the checked-out pre-permute function while replacing only the
packing kernels, then checks the caller's actual aliased tensor storage. The
four cases independently cover compact and masked layouts with `inplace=False`
and `inplace=True`.

Failing-before evidence was obtained by temporarily restoring unconditional
disposal in both branches. The two borrowed-input cases failed because the
caller's shape became `[0]`; both owned-input cases continued to pass. After
restoring the checked-out guards, all four cases passed. The temporary source
change is not part of the delivered diff.

The test tensors ran on the assigned single AMD Instinct MI350X (`gfx950`) via
PyTorch ROCm. This validates GPU tensor ownership/storage behavior and the
Python pre-permute control flow. The DeepGEMM packing kernels were mocked to
isolate that contract. Qwen3.5-397B-A17B-FP8 weights, eight H800 GPUs, and a
CUDA environment were unavailable, so the reported full-model TP8/EP8 serving
reproduction and numerical comparison remain unverified here.

Raw evidence is retained in `reports/j-752f67c926d3/raw/`.
