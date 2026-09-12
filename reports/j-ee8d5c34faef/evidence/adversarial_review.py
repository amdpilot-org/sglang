import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.bwap.bwap_fused import fused_pruned_mlp, gather_ffn_weights, silu_and_mul
from sglang.srt.bwap.bwap_manager import BWAPManager, build_topk_mask, compute_decode_scores
from sglang.srt.layers.activation import SiluAndMul


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.act_fn = SiluAndMul()


def phase_scoring_counterexample():
    manager = BWAPManager(base_model=TinyModel(), sparsity=0.5)
    key = "mlp.act_fn"
    # Four already-normalized exploration rows. Equation 2 pools the whole phase
    # by column L2/sqrt(T): neuron 0 fires persistently, neuron 1 spikes once.
    rows = torch.tensor(
        [[0.8, 0.0, 0.6], [0.8, 0.0, 0.6], [0.8, 0.0, 0.6], [0.0, 1.0, 0.0]]
    )
    paper_phase_score = rows.norm(dim=0) / rows.shape[0] ** 0.5
    # The candidate updates memory once per decode token using a max reduction,
    # producing max-over-time rather than Equation 2 over T_E tokens.
    for row in rows:
        manager._update_mem(key, compute_decode_scores(row.unsqueeze(0)))
    candidate_score = manager.mem_scores[key]
    paper_top1 = int(torch.topk(paper_phase_score, 1).indices.item())
    candidate_top1 = int(torch.topk(candidate_score, 1).indices.item())
    print("paper_phase_score", paper_phase_score.tolist(), "top1", paper_top1)
    print("candidate_score", candidate_score.tolist(), "top1", candidate_top1)
    assert paper_top1 == 0 and candidate_top1 == 1


def topk_cardinality_counterexample():
    scores = torch.tensor([3.0, 2.0, 1.0])
    actual_k = int(build_topk_mask(scores, sparsity=0.5).sum().item())
    paper_k = int((1.0 - 0.5) * scores.numel())  # floor from Eq. 3 text
    print("d_ff", scores.numel(), "sparsity", 0.5, "paper_k", paper_k, "candidate_k", actual_k)
    assert paper_k == 1 and actual_k == 2


def gpu_reference():
    assert torch.cuda.is_available()
    torch.manual_seed(31987)
    device = "cuda"
    dtype = torch.float16
    tokens, hidden, d_ff, k = 7, 96, 256, 83
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    gu = torch.randn(2 * d_ff, hidden, device=device, dtype=dtype) / hidden**0.5
    down = torch.randn(hidden, d_ff, device=device, dtype=dtype) / d_ff**0.5
    keep = torch.randperm(d_ff, device=device)[:k].sort().values
    mask = torch.zeros(d_ff, device=device, dtype=dtype)
    mask[keep] = 1
    independent = F.linear(silu_and_mul(F.linear(x, gu)) * mask, down)
    gu_k, down_k = gather_ffn_weights(gu, down, keep, d_ff)
    candidate = fused_pruned_mlp(x, gu_k, down_k)
    diff = (candidate - independent).abs()
    print("device", torch.cuda.get_device_name(0))
    print("shape", list(candidate.shape), "k", k)
    print("max_abs", float(diff.max()), "mean_abs", float(diff.mean()))
    torch.testing.assert_close(candidate, independent, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    phase_scoring_counterexample()
    topk_cardinality_counterexample()
    gpu_reference()
