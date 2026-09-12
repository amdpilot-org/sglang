from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import pad_dsa_cache_seqlens
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


device = "cuda"
batch = ForwardBatch(
    forward_mode=ForwardMode.DRAFT_EXTEND_V2,
    batch_size=3,
    input_ids=torch.arange(3, device=device),
    positions=torch.arange(3, device=device),
    req_pool_indices=torch.arange(3, device=device),
    seq_lens=torch.ones(3, dtype=torch.int32, device=device),
    seq_lens_cpu=torch.ones(3, dtype=torch.int32),
    extend_seq_lens_cpu=[1, 1, 1],
    out_cache_loc=torch.arange(3, device=device),
    seq_lens_sum=3,
)
batch.global_num_tokens_cpu = [3]
batch.dp_padding_mode = DpPaddingMode.SUM_LEN
batch.mark_forward_metadata_ready()

parallel = SimpleNamespace(attn_tp_size=2, attn_cp_size=1, attn_cp_rank=0, attn_dp_rank=0)
planned_expanded = torch.ones(3, dtype=torch.int32, device=device)
with patch("sglang.srt.layers.attention.dsa.utils.get_parallel", return_value=parallel):
    padded_dsa = pad_dsa_cache_seqlens(batch, planned_expanded)

backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
backend.device = torch.device(device)
indexer_ranges, token_to_batch = backend._cal_indexer_k_start_end(batch)

# Apply the physical post-plan padding which motivated the issue.
batch.batch_size = 4
batch.input_ids = torch.arange(4, device=device)
batch.positions = torch.arange(4, device=device)
batch.req_pool_indices = torch.arange(4, device=device)
batch.seq_lens = torch.ones(4, dtype=torch.int32, device=device)
batch.out_cache_loc = torch.arange(4, device=device)

backend.forward_metadata = SimpleNamespace(
    dsa_cache_seqlens_int32=padded_dsa,
    dsa_cu_seqlens_k=torch.nn.functional.pad(padded_dsa.cumsum(0), (1, 0)),
    dsa_cu_seqlens_q=torch.arange(5, dtype=torch.int32, device=device),
    dsa_seqlens_expanded=planned_expanded,
    token_to_batch_idx=token_to_batch,
    indexer_k_start_end=indexer_ranges,
    topk_indices_offset=None,
)

print(f"padded_dsa_rows={len(padded_dsa)} expanded_rows={len(planned_expanded)}")
print(f"actual_draft_indexer_ranges={indexer_ranges} actual_token_to_batch={token_to_batch}")
try:
    backend.validate_preplanned_metadata_extent(batch)
except RuntimeError as exc:
    print(f"REJECTED_ACTUAL_SHAPES: {exc}")
else:
    raise AssertionError("candidate unexpectedly accepted actual planner-shaped metadata")
