import os
import re
from pathlib import Path


SOURCE = Path(
    os.environ.get(
        "SGLANG_SILU_MUL_POST_QUANT_SOURCE",
        Path(__file__).parents[4]
        / "python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh",
    )
)


def _run_body(struct_name: str) -> str:
    source = SOURCE.read_text()
    start = source.index(f"struct {struct_name}")
    next_struct = source.find("\nstruct ", start + 1)
    return source[start : next_struct if next_struct != -1 else None]


def _assert_common_guards_precede_launch(body: str) -> None:
    block_guard = body.index(
        'RuntimeCheck(num_threads > 0 && num_threads <= 1024, '
        '"hidden_dim/8 exceeds CUDA block limit")'
    )
    launch = body.index("LaunchKernel(")
    assert block_guard < launch


def test_masked_launcher_guards_invalid_launch_boundaries() -> None:
    body = _run_body("SiluAndMulMaskedPostQuantKernel")
    _assert_common_guards_precede_launch(body)
    empty_guard = re.search(
        r"if \(num_tokens == 0 \|\| topk == 0\) \{\s*return;\s*\}", body
    )
    assert empty_guard is not None
    assert body.index("const auto params") < empty_guard.start() < body.index("LaunchKernel(")


def test_contiguous_launcher_guards_invalid_launch_boundaries() -> None:
    body = _run_body("SiluAndMulContigPostQuantKernel")
    _assert_common_guards_precede_launch(body)
    empty_guard = re.search(r"if \(num_tokens == 0\) \{\s*return;\s*\}", body)
    assert empty_guard is not None
    assert body.index("const auto params") < empty_guard.start() < body.index("LaunchKernel(")
