import torch
import torch.nn.functional as F


def reference(x, state, weight, cache_seqlens=None):
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = state.shape[2]
    if cache_seqlens is None:
        joined = torch.cat((state, x), dim=2)
        history_and_x = torch.cat((state[:, :, -(width - 1):], x), dim=2)
        state.copy_(joined[:, :, -state_len:])
    else:
        offsets = torch.arange(-(width - 1), 0, device=x.device)
        history_idx = (cache_seqlens[:, None] + offsets) % state_len
        history_idx = history_idx[:, None, :].expand(batch, dim, width - 1)
        history_and_x = torch.cat((state.gather(2, history_idx), x), dim=2)
        offsets = torch.arange(seqlen, device=x.device)
        write_idx = (cache_seqlens[:, None] + offsets) % state_len
        write_idx = write_idx[:, None, :].expand(batch, dim, seqlen)
        state.scatter_(2, write_idx, x)
    return F.conv1d(history_and_x, weight[:, None, :], groups=dim)[:, :, -seqlen:]
