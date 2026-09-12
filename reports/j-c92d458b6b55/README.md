# Investigation of HiCache generation-loop report

Upstream issue: https://github.com/sgl-project/sglang/issues/36179

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1241

Outcome: **not reproduced**. No source correction is justified by the available evidence.

The prepared base already includes the recent HiCache correctness changes that
landed around the report, including L3 lifecycle accounting (#37503), queued
Mooncake load preservation (#38195), DeepSeek-V4 unified-KV work (#38269),
load-back quota accounting (#38481), and host-lock boundary fixes (#38138 and
#38835). The source issue also contains an independent H200 negative result,
including a file-backed run where L3 served 98.3% of prefill tokens. It is not
the same single-instance Mooncake workload and therefore does not close the
report.

## Local reproduction

The campaign-qualified deterministic tiny Llama was run on the assigned
gfx950 GPU through the actual HTTP serving implementation. Four natural-EOS,
temperature-zero requests shared a 120-token prefix; three used the same final
token and one used a different final token. This was repeated for a control,
HiCache L2, and file-backed HiCache configuration.

All arms returned 32 completion tokens for every request, and all three
same-input outputs were byte-identical. The HiCache arms reported 120 cached
tokens on repeated prefixes. The file arm created 121 page files in the private
runtime directory. A fresh server using those files did not attribute its first
request to storage, so this evidence does not qualify the L3 read path.

Raw requests, responses, server logs, and run metadata are under `evidence/`.
The weights and compilation caches stayed outside the worktree under
`/tmp/amdpilot-repo-j-c92d458b6b55/`.

## Limitations

DeepSeek-V4-Flash weights, NVIDIA H20/CUDA, TP=4, the reporter's online-coding
dataset, Mooncake master/storage, RDMA, and the multi-terabyte L3 configuration
were unavailable. The tiny random Llama validates transport, GPU engine
execution, and repeated-prefix HiCache behavior only; it cannot validate
DeepSeek-V4 semantics or the reported distributed Mooncake workload.
