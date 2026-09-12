import torch

from sglang.srt.environ import envs
from sglang.srt.utils.async_probe import detect_full_nan_rows, sanitize_nan_logits


device = torch.device("cuda")
logits = torch.full((1, 163840), float("nan"), device=device)
with (
    envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True),
    envs.SGLANG_SANITIZE_NAN_LOGITS.override(False),
):
    mask = detect_full_nan_rows(logits)
    sanitize_nan_logits(logits, "gpu abort-only regression")
    torch.cuda.synchronize()

print(f"device={torch.cuda.get_device_name(0)}")
print(f"gfx={torch.cuda.get_device_properties(0).gcnArchName}")
print(f"mask={mask.cpu().tolist()}")
print(f"nan_after_preprocess={torch.isnan(logits).any().item()}")
print(f"finite_after_preprocess={torch.isfinite(logits).all().item()}")
finite = logits[torch.isfinite(logits)]
print(f"finite_value_count={finite.numel()}")
if finite.numel():
    print(f"finite_min={finite.min().item()}")
    print(f"finite_max={finite.max().item()}")
