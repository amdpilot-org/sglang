"""Deterministic evidence for the already-fixed MLX EOS comparison bug."""

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).parents[2] / (
    "test/registered/unit/hardware_backend/mlx/test_mlx_reference_correctness.py"
)
spec = importlib.util.spec_from_file_location("mlx_reference_correctness", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

case = module.TestMlxReferenceCorrectness(
    methodName="test_batched_decode_matches_solo"
)
case.eos_ids = {2, 9}

boundary_cases = [
    ([], []),
    ([2, 7], [2]),
    ([1, 2, 7], [1, 2]),
    ([1, 9, 2, 7], [1, 9]),
    ([1, 3, 7], [1, 3, 7]),
]
for sequence, expected in boundary_cases:
    assert case._truncate_at_eos(sequence) == expected
print(f"boundary cases passed: {len(boundary_cases)}")

# Mirrors the issue: both streams agree through EOS, then differ in filler.
solo = [10, 11, 2, 20, 21, 22, 30]
batched = [10, 11, 2, 20, 21, 99, 30]
assert batched != solo
print("pre-fix full-horizon comparison: FAIL (expected)")
assert case._truncate_at_eos(batched) == case._truncate_at_eos(solo)
print("current EOS-bounded comparison: PASS")
