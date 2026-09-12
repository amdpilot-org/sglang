import json
import sys

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import _DRAFT_PROBS
from sglang.srt.utils.invariants import expect


def run(device: torch.device):
    cases = {
        "nan_col0": [float("nan"), 0.5, 0.5],
        "nan_nonzero_col": [0.5, float("nan"), 0.5],
        "pos_inf": [float("inf"), 0.0, 0.0],
        "neg_inf": [0.5, float("-inf"), 0.5],
        "negative": [-0.1, 0.6, 0.5],
        "zero_sum": [0.0, 0.0, 0.0],
        "valid_normalized": [0.1, 0.2, 0.7],
        "valid_unnormalized": [1.0, 2.0, 7.0],
    }
    results = {}
    for name, row in cases.items():
        value = torch.tensor([row], dtype=torch.float32, device=device)
        recovered = expect(_DRAFT_PROBS, value)
        try:
            sample = torch.multinomial(recovered, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            sampling = {"ok": True, "sample": sample.cpu().tolist()}
        except Exception as exc:
            sampling = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        results[name] = {
            "input": value.cpu().tolist(),
            "recovered": recovered.cpu().tolist(),
            "sampling": sampling,
        }
    return results


requested = sys.argv[1] if len(sys.argv) > 1 else "all"
payload = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "python": sys.executable,
    "module": sys.modules[_DRAFT_PROBS.recover.__module__].__file__,
}
if requested in ("all", "cpu"):
    payload["cpu"] = run(torch.device("cpu"))
if requested in ("all", "gpu") and torch.cuda.is_available():
    payload["gpu_name"] = torch.cuda.get_device_name(0)
    payload["gpu"] = run(torch.device("cuda"))
print(json.dumps(payload, allow_nan=True, indent=2))
