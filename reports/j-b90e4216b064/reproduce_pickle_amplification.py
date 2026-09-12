import pickle

import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


def measure(n=512, rows=8, width=64):
    feature = torch.zeros((n * rows, width), dtype=torch.float32)
    grids = torch.tensor([[1, 1, rows]] * n, dtype=torch.int64)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=feature,
        offsets=[(i, i) for i in range(n)],
        model_specific_data={"image_grid_thw": grids},
    )
    items = get_new_expanded_mm_items([item])
    logical_bytes = sum(
        split.feature.numel() * split.feature.element_size() for split in items
    )
    pickle_bytes = len(pickle.dumps(items, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"items: {len(items)}")
    print(f"logical_bytes: {logical_bytes}")
    print(f"pickle_bytes: {pickle_bytes}")
    print(f"amplification: {pickle_bytes / logical_bytes:.3f}")
    print(f"first_storage_bytes: {items[0].feature.untyped_storage().nbytes()}")
    print(f"first_logical_bytes: {items[0].feature.numel() * items[0].feature.element_size()}")


if __name__ == "__main__":
    measure()
