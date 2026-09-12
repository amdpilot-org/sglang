"""Exercise the actual Mamba donation method with a device-resident slot."""

from types import SimpleNamespace

import torch

from sglang.srt.mem_cache import memory_pool


class Pool:
    get_mamba_ping_pong_keep_idx = staticmethod(lambda req: 0)

    @staticmethod
    def set_mamba_ping_pong_slot(req, idx, value):
        req.kv.mamba_ping_pong_track_buffer[idx] = value


def run(debug: bool):
    req = SimpleNamespace(
        rid="gpu-regression",
        kv=SimpleNamespace(
            mamba_ping_pong_track_buffer=torch.tensor([-1, 7], device="cuda"),
            mamba_next_track_idx=0,
        ),
    )
    new_slot = torch.tensor([11], device="cuda")
    memory_pool._MAMBA_DEBUG_ASSERTS = debug
    outcome = "returned"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        try:
            memory_pool.HybridReqToTokenPool.donate_mamba_ping_pong_slot(
                Pool(), req, new_slot
            )
        except AssertionError:
            outcome = "asserted"
    scalar_ops = sum(
        event.count
        for event in profile.key_averages()
        if "local_scalar_dense" in event.key
    )
    return outcome, scalar_ops, req.kv.mamba_ping_pong_track_buffer.tolist()


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"architecture={torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"debug_off={run(False)}")
    print(f"debug_on={run(True)}")
