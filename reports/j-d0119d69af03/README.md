# Independent review of PR 1062

Upstream issue: https://github.com/sgl-project/sglang/issues/37215

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1098

Candidate: https://github.com/amdpilot-org/sglang/pull/1062 at exact commit
`e5ce94b77e1db3a2c4641abf97ebf391926af8ba`.

## Recommendation

Request changes. The candidate is a useful partial mitigation for automatically
assigned ports, but it does not fully satisfy the original issue's requirement
that all eight DP replicas receive unique, collision-free TCPStore ports.

On the recorded base, all 64 samples from `get_free_port()` fell inside this
host's Linux ephemeral range (`32768-60999`). Reproducing the controller's
reserve/close/start lifecycle, an unrelated client acquired the released port
and a real `torch.distributed.TCPStore` then failed with the reported
`DistNetworkError`, errno 98, `EADDRINUSE`.

At the candidate commit, the focused candidate regression and `TestPortArgs`
suite passed (37 tests). The imported SGLang networking module was
`/job/repo/python/sglang/srt/utils/network.py`, confirming the checked-out
source was tested. Eight candidate-selected ports were outside the ephemeral
range, and eight real TCPStore instances bound them successfully.

However, the candidate retains the close-before-TCPStore-bind window. An
independent listener acquiring a selected non-ephemeral port after the
reservation closed caused the same real TCPStore `EADDRINUSE`. More directly,
the original launch includes `--nccl-port 30101`: the candidate deliberately
leaves explicit ports unchanged. Repeating the DP controller's allocation and
reservation sequence with `ServerArgs(dp_size=8, nccl_port=30101)` returned
30101 for every replica and failed to reserve it by the third iteration on this
host. Thus the candidate does not make that reported configuration allocate a
unique port per DP replica.

## Evidence and limitations

Raw commands and output are retained outside the revision-switching checkout
under `/job/review-evidence/`. No native files changed, so no native rebuild was
applicable. These tests exercised real TCPStore CPU networking; no GPU kernel
was involved. The prepared environment is PyTorch 2.11.0 with ROCm 7.2 and one
assigned AMD MI355X/gfx950-class GPU, not eight NVIDIA H800 GPUs. Qwen3-Embedding
weights were not available. Therefore the full model-serving launch,
CUDA/NCCL behavior, intermittency rate, and eight-GPU execution remain
unverified. The deterministic evidence validates the port lifecycle and
TCPStore bind failure, not the unavailable architecture/model workload.

