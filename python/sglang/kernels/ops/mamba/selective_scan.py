"""Portable Mamba-1 selective scan.

This is intentionally expressed in PyTorch so it works on CUDA and ROCm and
provides a correctness-first fallback until a fused JIT kernel is available.
"""

import torch
import torch.nn.functional as F


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    initial_state=None,
):
    if u.ndim != 3:
        raise ValueError(f"u must have shape [batch, length, dim], got {u.shape}")
    batch, length, dim = u.shape
    state = (
        torch.zeros(batch, dim, A.shape[-1], device=u.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float()
    outputs = []
    for t in range(length):
        dt = delta[:, t]
        decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        state = decay * state + dt.unsqueeze(-1) * B[:, t].float().unsqueeze(1) * u[
            :, t
        ].float().unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", state, C[:, t].float())
        if D is not None:
            y = y + u[:, t].float() * D.float()
        if z is not None:
            y = y * F.silu(z[:, t].float())
        outputs.append(y)
    output = (
        torch.stack(outputs, dim=1).to(u.dtype)
        if outputs
        else u.new_empty((batch, 0, dim))
    )
    return (output, state) if return_last_state else output
