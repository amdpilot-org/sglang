"""Deterministic regression evidence for sglang#38022 / sglang#37968."""

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    processor = Glm4vImageProcessor.__new__(Glm4vImageProcessor)
    processor.IM_TOKEN_ID = 99
    processor.VIDEO_START_TOKEN_ID = 101
    processor.VIDEO_END_TOKEN_ID = 102
    mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=99)
    input_ids = torch.tensor(
        [1, 99, 99, 2, 99, 99, 99, 3, 101, 4, 99, 99, 5, 99, 99, 99, 6, 102],
        device=device,
    )

    # Before PR #37971, both modalities used this generic token-ID lookup.
    legacy_image_offsets = processor.get_mm_items_offset(input_ids, 99)
    legacy_video_offsets = processor.get_mm_items_offset(input_ids, 99)
    assert legacy_image_offsets == legacy_video_offsets
    assert len(legacy_image_offsets) == 4

    image_offsets = processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.IMAGE)
    video_offsets = processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.VIDEO)
    assert image_offsets == [(1, 2), (4, 6)]
    assert video_offsets == [(10, 11), (13, 15)]

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"legacy_image_offsets={legacy_image_offsets}")
    print(f"legacy_video_offsets={legacy_video_offsets}")
    print(f"fixed_image_offsets={image_offsets}")
    print(f"fixed_video_offsets={video_offsets}")
    print("legacy_reproduction=FAIL (both modalities own all four spans)")
    print("current_implementation=PASS (two spans assigned to each modality)")


if __name__ == "__main__":
    main()
