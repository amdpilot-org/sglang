import json
import torch

from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2


def run_case(name, seq, values, k=2048):
    scores = values.reshape(1, seq).cuda()
    lens = torch.tensor([seq], dtype=torch.int32, device="cuda")
    metadata = plan_topk_v2(lens)
    torch.cuda.synchronize()
    out = torch.full((1, k), -1, dtype=torch.int32, device="cuda")
    topk_transform_paged_v2(scores, lens, None, out, 64, metadata)
    torch.cuda.synchronize()
    got_idx = out[0].cpu()
    valid = got_idx[got_idx >= 0]
    got = scores[0, valid.long()].cpu()
    ref = torch.topk(scores[0], k, sorted=False).values.cpu()
    got_sorted = torch.sort(got).values
    ref_sorted = torch.sort(ref).values
    mismatch = int((got_sorted != ref_sorted).sum()) if len(got_sorted) == k else k
    result = {
        "name": name, "seq": seq, "valid": int(valid.numel()),
        "unique": int(valid.unique().numel()), "value_mismatches": mismatch,
        "got_min": float(got.min()) if len(got) else None,
        "ref_min": float(ref.min()),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


torch.manual_seed(20260912)
for seq in (8192, 32768, 262144):
    # 4096 strictly increasing representable fp32 values remain inside the
    # fp16 coarse bucket around 1.0; the rest are far below it.
    vals = torch.full((seq,), -100.0, dtype=torch.float32)
    vals[:4096] = 1.0 + torch.arange(4096, dtype=torch.float32) * torch.finfo(torch.float32).eps
    run_case(f"increasing_overflow_{seq}", seq, vals)

for count in (2048, 2049, 4096):
    seq = 8192
    vals = torch.full((seq,), -100.0, dtype=torch.float32)
    vals[:count] = 1.0 + torch.arange(count, dtype=torch.float32) * torch.finfo(torch.float32).eps
    run_case(f"boundary_{count}", seq, vals)

# Exact ties are allowed to choose any member; the selected value multiset must match.
vals = torch.full((8192,), -100.0, dtype=torch.float32)
vals[:5000] = 1.0
run_case("exact_tie_5000", 8192, vals)

# Negative, fp16-subnormal scores exercise the order-preserving exact key on
# the opposite sign and around signed zero while remaining a single coarse bin.
vals = torch.linspace(-5e-8, 5e-8, 8192, dtype=torch.float32)
run_case("signed_subnormal", 8192, vals)
