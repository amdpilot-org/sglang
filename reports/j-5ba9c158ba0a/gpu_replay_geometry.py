import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


device = torch.device("cuda")
runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
runner.captured_req_width = 6
captured = RaggedVerifyLayout.from_verify_lens(
    verify_lens_cpu=[1] * 192, device=device, grid=[1, 32, 192]
)
runner._captured_ragged_layouts = {192: captured}

for width, expected in (
    (6, [6] * 32 + [0] * 160),
    (5, [5] * 32 + [1] * 32 + [0] * 128),
):
    live = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[width] * 32, device=device, grid=[1, 32, 192]
    )
    runner._stage_ragged_verify_layout(live, 192)
    torch.cuda.synchronize()
    assert captured.verify_lens.cpu().tolist() == expected
    assert captured.qo_indptr_device[-1].item() == 192
    print(
        f"width={width} slots={captured.bs} total="
        f"{captured.qo_indptr_device[-1].item()} synthetic_nonzero="
        f"{sum(value > 0 for value in expected[32:])}"
    )

print(torch.cuda.get_device_name(0))
