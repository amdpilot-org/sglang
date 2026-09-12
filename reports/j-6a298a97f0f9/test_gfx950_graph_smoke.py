import torch


def test_hip_graph_matches_cpu_reference():
    assert torch.cuda.is_available()
    assert torch.version.hip is not None
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")

    torch.manual_seed(7)
    x_cpu = torch.randn(64, 64, dtype=torch.float32)
    y_cpu = torch.randn(64, 64, dtype=torch.float32)
    expected = x_cpu @ y_cpu

    x = x_cpu.cuda()
    y = y_cpu.cuda()
    out = torch.empty_like(x)
    # Initialize hipBLASLt before capture, as serving runners do in warmup.
    torch.mm(x, y, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mm(x, y, out=out)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out.cpu(), expected, rtol=2e-5, atol=2e-5)
