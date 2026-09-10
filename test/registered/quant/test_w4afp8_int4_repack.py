import pytest
import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8_moe import (
    _unpack_repack_int32_to_cutlass_int8,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=1, suite="stage-a-test-1-gpu-small-amd")


def _bit_reference(weight_packed: torch.Tensor) -> torch.Tensor:
    num_bits = 4
    pack_factor = 32 // num_bits
    pair_factor = pack_factor // 2
    offset = 1 << (num_bits - 1)
    mask = (1 << num_bits) - 1

    expected = []
    for word in weight_packed.reshape(-1).tolist():
        unsigned_word = word & 0xFFFFFFFF
        for pair_idx in range(pair_factor):
            low_shift = num_bits * (2 * pair_idx)
            high_shift = low_shift + num_bits
            low = ((unsigned_word >> low_shift) & mask) - offset
            high = ((unsigned_word >> high_shift) & mask) - offset
            byte = ((high & 0x0F) << 4) | (low & 0x0F)
            expected.append(byte - 256 if byte >= 128 else byte)

    return torch.tensor(expected, dtype=torch.int8).view(
        *weight_packed.shape[:-1], weight_packed.shape[-1] * pair_factor
    )


def _pack_nibbles(nibbles):
    return sum(int(nibble) << (4 * index) for index, nibble in enumerate(nibbles))


def test_w4afp8_int4_repack_matches_independent_bit_reference():
    device = torch.device("cuda")
    nibbles = torch.arange(16, dtype=torch.uint8)
    words = torch.tensor(
        [_pack_nibbles(nibbles[:8]), _pack_nibbles(nibbles[8:])],
        dtype=torch.uint32,
    ).view(torch.int32)
    weight_packed = words.view(1, 1, 2).to(device)

    output = _unpack_repack_int32_to_cutlass_int8(weight_packed, num_bits=4)

    assert output.dtype == torch.int8
    assert output.shape == (1, 1, 8)
    assert output.is_contiguous()
    torch.testing.assert_close(output.cpu(), _bit_reference(weight_packed.cpu()))


def test_w4afp8_int4_repack_odd_length_group_and_padding_boundaries():
    device = torch.device("cuda")
    logical_k = 257
    group_size = 128
    packed_cols = (logical_k + 7) // 8
    shape = (2, 3, packed_cols)
    prefix_elements = 7
    suffix_elements = 5
    packed_elements = shape[0] * shape[1] * shape[-1]

    storage = torch.randint(
        low=torch.iinfo(torch.int32).min,
        high=torch.iinfo(torch.int32).max,
        size=(prefix_elements + packed_elements + suffix_elements,),
        dtype=torch.int32,
        device=device,
    )
    weight_packed = storage[prefix_elements : prefix_elements + packed_elements].view(
        shape
    )

    words = []
    for word_idx in range(packed_cols):
        nibbles = [(word_idx * 8 + nibble_idx) % 16 for nibble_idx in range(8)]
        if word_idx == packed_cols - 1:
            nibbles[1:] = [0xA, 0x5] * 3 + [0xA]
        words.append(_pack_nibbles(nibbles))
    weight_packed.copy_(
        torch.tensor(words, dtype=torch.uint32).to(device).view(torch.int32)
    )
    storage_before = storage.clone()

    output = _unpack_repack_int32_to_cutlass_int8(weight_packed, num_bits=4)
    reference = _bit_reference(weight_packed.cpu())

    assert weight_packed.storage_offset() == prefix_elements
    assert weight_packed.is_contiguous()
    assert output.shape == (2, 3, packed_cols * 4)
    assert output.is_contiguous()
    torch.testing.assert_close(output.cpu(), reference)

    output_bytes = output.view(torch.uint8).reshape(-1)
    group_boundary_byte = group_size
    odd_logical_nibble = 0
    padding_nibble = 0xA
    expected_boundary_byte = ((padding_nibble - 8) & 0x0F) << 4 | (
        (odd_logical_nibble - 8) & 0x0F
    )
    assert int(output_bytes[group_boundary_byte]) == expected_boundary_byte
    assert torch.equal(storage, storage_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
