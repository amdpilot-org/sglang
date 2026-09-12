from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.utils import cal_padded_tokens, pad_dsa_cache_seqlens
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import ceil_align


def batch(counts, mode):
    return SimpleNamespace(
        attn_cp_metadata=None,
        global_num_tokens_cpu=list(counts),
        dp_padding_mode=mode,
        is_extend_in_batch=False,
        forward_mode=ForwardMode.DECODE,
    )


checked = 0
for tp, counts, mode, rank in product(
    (1, 2, 3, 8, 16),
    ((0,), (1,), (0, 1), (1, 0), (7, 8), (8, 9), (11, 1), (17, 33)),
    (DpPaddingMode.MAX_LEN, DpPaddingMode.SUM_LEN),
    (0, 1),
):
    if rank >= len(counts):
        continue
    aligned = [ceil_align(n, tp) for n in counts]
    expected = max(aligned) if mode.is_max_len() else aligned[rank]
    fb = batch(counts, mode)
    with get_parallel().override(attn_tp_size=tp, attn_dp_rank=rank, attn_cp_size=1, attn_cp_rank=0):
        actual = cal_padded_tokens(fb)
        assert actual == expected, (tp, counts, mode, rank, actual, expected)
    assert fb.global_num_tokens_cpu == list(counts), "helper mutated caller counts"
    checked += 1

# Exercise the actual tensor-padding consumer on the assigned GPU, including an
# idle DP rank and a non-power-of-two attention TP size.
device = "cuda"
for tp, counts, mode, rank in (
    (8, (11, 1), DpPaddingMode.MAX_LEN, 1),
    (8, (11, 0), DpPaddingMode.SUM_LEN, 1),
    (3, (4, 7), DpPaddingMode.SUM_LEN, 0),
):
    fb = batch(counts, mode)
    real = counts[rank]
    source = torch.arange(real, device=device, dtype=torch.int32)
    with get_parallel().override(attn_tp_size=tp, attn_dp_rank=rank, attn_cp_size=1, attn_cp_rank=0):
        padded = pad_dsa_cache_seqlens(fb, source)
        expected_rows = max(ceil_align(n, tp) for n in counts) if mode.is_max_len() else ceil_align(real, tp)
    assert padded.shape == (expected_rows,)
    assert torch.equal(padded[:real], source)
    assert torch.count_nonzero(padded[real:]).item() == 0
    print(f"gpu tp={tp} counts={counts} mode={mode.name} rank={rank} rows={padded.shape[0]}")

props = torch.cuda.get_device_properties(0)
print(f"checked={checked} device={torch.cuda.get_device_name(0)} arch={getattr(props, 'gcnArchName', None)}")
