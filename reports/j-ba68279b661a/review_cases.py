import json
import pickle
import sys

import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


def stats(name, item):
    out = get_new_expanded_mm_items([item])
    tensors = [x.feature if x.feature is not None else x.precomputed_embeddings for x in out]
    logical = sum(t.numel() * t.element_size() for t in tensors)
    payload = len(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    return {"name": name, "items": len(out), "logical_bytes": logical,
            "pickle_bytes": payload, "amplification": payload / logical,
            "storage_bytes": [t.untyped_storage().nbytes() for t in tensors[:3]],
            "tensor_bytes": [t.numel() * t.element_size() for t in tensors[:3]],
            "contiguous": [t.is_contiguous() for t in tensors[:3]]}


cases = []
n, rows, width = 512, 8, 64
cases.append(stats("exact_issue", MultimodalDataItem(
    modality=Modality.IMAGE, feature=torch.zeros((n * rows, width)),
    offsets=[(i, i) for i in range(n)],
    model_specific_data={"image_grid_thw": torch.tensor([[1, 1, rows]] * n)})))

n2, rows2, width2 = 16, 8, 64
noncontiguous = torch.zeros((width2, n2 * rows2)).t()
cases.append(stats("noncontiguous_feature", MultimodalDataItem(
    modality=Modality.IMAGE, feature=noncontiguous,
    offsets=[(i, i) for i in range(n2)],
    model_specific_data={"image_grid_thw": torch.tensor([[1, 1, rows2]] * n2)})))

n3, width3 = 32, 64
emb = torch.arange(n3 * width3, dtype=torch.float32).reshape(n3, width3)
cases.append(stats("simple_precomputed", MultimodalDataItem(
    modality=Modality.IMAGE, precomputed_embeddings=emb,
    offsets=[(i, i) for i in range(n3)],
    model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 1]] * n3)})))

print(json.dumps({"python": sys.executable, "torch": torch.__version__,
    "torch_file": torch.__file__,
    "mm_utils_file": sys.modules["sglang.srt.managers.mm_utils"].__file__,
    "schedule_batch_file": sys.modules["sglang.srt.managers.schedule_batch"].__file__,
    "cases": cases}, indent=2))
