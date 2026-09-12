from pathlib import Path


DOC = (
    Path(__file__).parents[4]
    / "docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx"
)


def _text():
    return DOC.read_text()


def test_torchnpu_release_and_package_versions_are_disambiguated():
    text = _text()
    assert "26.0.0 (`torch_npu==2.10.0`)" in text
    assert "26.1.0 (`torch_npu==2.11.0`)" in text
    assert "TorchNPU release 26.0.0 corresponds to the `torch_npu==2.10.0`" in text


def test_graph_capture_fallback_is_qualified():
    text = _text()
    assert "Graph capture is enabled by default and is supported on Ascend" in text
    assert "If graph\ncapture fails during startup" in text
    assert "`--disable-cuda-graph`" in text
    assert "can reduce performance" in text


def test_modelscope_instructions_cover_remote_and_local_paths():
    text = _text()
    assert "export SGLANG_USE_MODELSCOPE=true" in text
    assert "modelscope download --model Qwen/Qwen2.5-7B-Instruct" in text
    assert "pass the local directory printed by `modelscope download`" in text
