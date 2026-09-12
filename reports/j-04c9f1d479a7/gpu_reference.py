from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import (
    cal_padded_tokens,
    pad_dsa_cache_seqlens,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

assert torch.cuda.is_available()
print("device=", torch.cuda.get_device_name(0))
batch = ForwardBatch(
    forward_mode=ForwardMode.DRAFT_EXTEND_V2,
    batch_size=3,
    input_ids=torch.arange(3, device="cuda"),
    req_pool_indices=torch.arange(3, device="cuda"),
    seq_lens=torch.ones(3, dtype=torch.int32, device="cuda"),
    out_cache_loc=torch.arange(3, device="cuda"),
    seq_lens_sum=3,
)
batch.global_num_tokens_cpu = [3]
batch.dp_padding_mode = DpPaddingMode.SUM_LEN
parallel = SimpleNamespace(
    attn_tp_size=2, attn_cp_size=1, attn_cp_rank=0, attn_dp_rank=0
)
planned = torch.tensor([11, 12, 13], dtype=torch.int32, device="cuda")
with patch("sglang.srt.layers.attention.dsa.utils.get_parallel", return_value=parallel):
    actual_rows = cal_padded_tokens(batch)
    actual = pad_dsa_cache_seqlens(batch, planned)

reference_rows = ((3 + 2 - 1) // 2) * 2
reference = torch.tensor([11, 12, 13, 0], dtype=torch.int32, device="cuda")
torch.testing.assert_close(actual, reference)
assert actual_rows == reference_rows == 4
print("physical_rows=", actual_rows)
print("gpu_metadata=", actual.cpu().tolist())
print("independent_reference=", reference.cpu().tolist())
print("rocm=", torch.version.hip)
