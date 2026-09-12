import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.utils.profile_utils import _ProfilerTorch


output_dir = Path("/tmp/amdpilot-repo-j-b0e05ade0242/gpu-profile-v2")
output_dir.mkdir(parents=True, exist_ok=True)
profile = _ProfilerTorch(
    with_stack=False,
    record_shapes=False,
    activities=["CPU", "GPU"],
    output_dir=str(output_dir),
    output_prefix="",
    output_suffix="-DECODE",
    profile_id="correction",
    ps=SimpleNamespace(
        tp_rank=0,
        dp_size=1,
        dp_rank=0,
        pp_size=1,
        pp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    ),
    cpu_group=None,
    first_rank_in_node=True,
)

torch.manual_seed(7)
left_cpu = torch.randn(256, 256)
right_cpu = torch.randn(256, 256)
profile.start()
actual = left_cpu.cuda() @ right_cpu.cuda()
torch.cuda.synchronize()
profile.stop()
profile.export_thread.join(timeout=60)
reference = left_cpu @ right_cpu
trace_path = output_dir / "correction-TP-0-DECODE.trace.json.gz"
print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "max_abs_error": (actual.cpu() - reference).abs().max().item(),
            "trace_path": str(trace_path),
            "trace_bytes": trace_path.stat().st_size,
            "export_thread_alive": profile.export_thread.is_alive(),
        },
        indent=2,
    )
)
