"""Measure shared graph-pool bridge lifetime on the assigned accelerator."""

import gc
import json

import torch

from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors


def run(shape: tuple[int, ...], weak: bool) -> dict:
    pool = torch.cuda.graph_pool_handle()
    x_a = torch.full(shape, 7.0, device="cuda")
    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a, pool=pool):
        bridge = x_a * 1.0
    bridge_ptr = bridge.data_ptr()
    graph_a.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(bridge, torch.full_like(bridge, 7.0))

    kept = weak_ref_tensors(bridge) if weak else bridge
    if weak:
        del bridge
        gc.collect()

    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b, pool=pool):
        junk = torch.zeros(shape, device="cuda")
    junk_ptr = junk.data_ptr()
    junk.fill_(-999.0)
    torch.cuda.synchronize()

    bridge_sample = kept.flatten()[:4].cpu().tolist()
    graph_a.replay()
    torch.cuda.synchronize()
    junk_sample = junk.flatten()[:4].cpu().tolist()
    result = {
        "shape": list(shape),
        "mode": "weak" if weak else "strong",
        "bridge_ptr": hex(bridge_ptr),
        "junk_ptr": hex(junk_ptr),
        "pointer_reused": junk_ptr == bridge_ptr,
        "bridge_sample_after_b_capture": bridge_sample,
        "bridge_corrupted": bridge_sample != [7.0] * 4,
        "junk_sample_after_a_replay": junk_sample,
        "reverse_clobbered": junk_sample != [-999.0] * 4,
    }
    del graph_a, graph_b, x_a, kept, junk, pool
    torch.cuda.synchronize()
    gc.collect()
    return result


if __name__ == "__main__":
    results = []
    for shape in ((64, 2048), (128, 1024)):
        results.append(run(shape, weak=True))
        results.append(run(shape, weak=False))
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(0),
                "results": results,
            },
            indent=2,
        )
    )
    for result in results:
        observed = (
            result["pointer_reused"],
            result["bridge_corrupted"],
            result["reverse_clobbered"],
        )
        expected = (True, True, True) if result["mode"] == "weak" else (
            False,
            False,
            False,
        )
        assert observed == expected, (result, expected)
