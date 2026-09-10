import unittest

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=3, suite="stage-b-test-1-gpu-small-amd")

torch.manual_seed(0)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


_CASES = [
    (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1),
    (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2),
    (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2),
    (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8),
    (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4),
    (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2),
]

_POSITIONS = (0, 1, 7, 63, 255, 1023, 4095, 8191, 16383)
_HEAD_DIMS = (
    (64, 32),
    (64, 64),
    (128, 64),
    (128, 128),
    (256, 128),
    (256, 256),
    (512, 256),
    (512, 512),
)
_DTYPES = (torch.float16, torch.bfloat16)
_PARITY_CASES = [
    (head_size, rotary_dim, is_neox, dtype)
    for head_size, rotary_dim in _HEAD_DIMS
    for is_neox in (True, False)
    for dtype in _DTYPES
]


def _independent_reference(
    head_size: int,
    rotary_dim: int,
    base: int,
    is_neox: bool,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    tensor_3d = tensor.float().view(positions.numel(), -1, head_size)
    inverse_frequency = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=tensor.device)
            / rotary_dim
        )
    )
    angles = positions.float()[:, None] * inverse_frequency[None, :]
    cosine = angles.cos()[:, None, :]
    sine = angles.sin()[:, None, :]
    rotary = tensor_3d[..., :rotary_dim]
    passthrough = tensor_3d[..., rotary_dim:]
    if is_neox:
        first = rotary[..., : rotary_dim // 2]
        second = rotary[..., rotary_dim // 2 :]
        rotated = torch.cat(
            (
                first * cosine - second * sine,
                second * cosine + first * sine,
            ),
            dim=-1,
        )
    else:
        even = rotary[..., 0::2]
        odd = rotary[..., 1::2]
        rotated = torch.stack(
            (
                even * cosine - odd * sine,
                odd * cosine + even * sine,
            ),
            dim=-1,
        ).flatten(-2)
    return torch.cat((rotated, passthrough), dim=-1).flatten(1).to(tensor.dtype)


@unittest.skipIf(
    _use_aiter or not (_is_hip and torch.cuda.is_available()),
    reason="Requires an AMD HIP GPU and the native rotary kernel without AITER",
)
class TestRotaryEmbeddingNative(CustomTestCase):
    def _run_case(
        self,
        head_size: int,
        rotary_dim: int,
        max_pos: int,
        base: int,
        is_neox: bool,
        dtype: torch.dtype,
        device: str,
        batch_size: int,
        seq_len: int,
        num_q: int,
        num_kv: int,
        independent_reference: bool = False,
    ) -> None:
        rope_hip = RotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len, num_q * head_size, dtype=dtype, device=device
        )
        key = torch.randn(
            batch_size * seq_len, num_kv * head_size, dtype=dtype, device=device
        )

        if independent_reference:
            q_ref = _independent_reference(
                head_size, rotary_dim, base, is_neox, pos_ids, query
            )
            k_ref = _independent_reference(
                head_size, rotary_dim, base, is_neox, pos_ids, key
            )
        else:
            q_ref, k_ref = rope_hip.forward_native(pos_ids, query.clone(), key.clone())
        q_hip, k_hip = rope_hip.forward_cuda(pos_ids, query.clone(), key.clone())
        torch.cuda.synchronize()

        if independent_reference:
            atol = 4e-3 if dtype == torch.float16 else 3e-2
            rtol = 2e-2
        else:
            atol = 1e-2
            rtol = 1e-2
        torch.testing.assert_close(q_ref, q_hip, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_ref, k_hip, atol=atol, rtol=rtol)

    def test_all_cases(self) -> None:
        """Drive over the full parameter matrix using subTest()."""
        for case in _CASES:
            with self.subTest(case=case):
                self._run_case(*case)

    def test_position_dtype_parity(self) -> None:
        for head_size, rotary_dim, is_neox, dtype in _PARITY_CASES:
            with self.subTest(
                head_size=head_size,
                rotary_dim=rotary_dim,
                is_neox=is_neox,
                dtype=dtype,
            ):
                self._run_case(
                    head_size,
                    rotary_dim,
                    max(_POSITIONS) + 1,
                    10000,
                    is_neox,
                    dtype,
                    "cuda",
                    1,
                    len(_POSITIONS),
                    2,
                    1,
                    independent_reference=True,
                )


@unittest.skipIf(not _use_aiter, reason="Requires AMD GPU plus SGLANG_USE_AITER=1")
class TestRotaryEmbeddingAITer(CustomTestCase):
    # NOTE: Slightly relaxed tolerance (2e-2 vs 1e-2) for AITER RoPE kernel.
    # Minor precision differences under investigation.
    # See: https://github.com/sgl-project/sglang/pull/15318

    @staticmethod
    def _run_case_aiter(
        head_size: int,
        rotary_dim: int,
        max_pos: int,
        base: int,
        is_neox: bool,
        dtype: torch.dtype,
        device: str,
        batch_size: int,
        seq_len: int,
        num_q: int,
        num_kv: int,
    ) -> None:
        from aiter.rotary_embedding import RotaryEmbedding as AiterRotaryEmbedding

        rope_ref = AiterRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)
        rope_hip = AiterRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len, num_q * head_size, dtype=dtype, device=device
        )
        key = torch.randn(
            batch_size * seq_len, num_kv * head_size, dtype=dtype, device=device
        )

        q_ref, k_ref = rope_ref.forward_native(pos_ids, query.clone(), key.clone())
        q_hip, k_hip = rope_hip.forward_hip(pos_ids, query.clone(), key.clone())

        torch.testing.assert_close(q_ref, q_hip, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=2e-2, rtol=2e-2)

    def test_all_cases(self) -> None:
        for case in _CASES:
            with self.subTest(case=case):
                self._run_case_aiter(*case)


if __name__ == "__main__":
    unittest.main()
