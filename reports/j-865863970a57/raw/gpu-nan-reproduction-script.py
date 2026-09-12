import torch

from sglang.srt.environ import envs
from sglang.srt.utils.async_probe import detect_full_nan_rows, sanitize_nan_logits

print("torch", torch.__version__)
print("device_count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0))
logits = torch.full((2, 163840), float("nan"), device="cuda")
logits[1, 42] = 3.0
with envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True):
    mask = detect_full_nan_rows(logits)
with envs.SGLANG_SANITIZE_NAN_LOGITS.override(True):
    sanitize_nan_logits(logits)
probs = torch.softmax(logits, dim=-1)
expected = 1.0 / logits.shape[-1]
print("mask", mask.cpu().tolist())
print("full_row_unique_logits", torch.unique(logits[0]).cpu().tolist())
print(
    "full_row_max_abs_error_from_uniform",
    (probs[0] - expected).abs().max().item(),
)
print("full_row_sum", probs[0].sum().item())
print("partial_row_argmax", probs[1].argmax().item())
print("partial_row_p42", probs[1, 42].item())
