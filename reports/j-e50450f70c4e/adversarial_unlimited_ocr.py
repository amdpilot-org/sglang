import json
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM


def make_item(device, count, marker, flag):
    item = MultimodalDataItem(modality=Modality.IMAGE)
    item.feature = torch.full((1, 3, 2, 2), marker, device=device)
    item.images_crop = torch.full((1, count, 3, 2, 2), marker / 10, device=device)
    item.images_spatial_crop = torch.tensor([[[count, 1]]], device=device)
    item.has_local_crops = flag
    return item


def run(device):
    model = UnlimitedOCRForCausalLM.__new__(UnlimitedOCRForCausalLM)

    class Vision:
        dtype = torch.float32

    model.vision_model = Vision()
    calls = []

    def encoder(pixel_values, images_crop, images_spatial_crop, has_local_crops):
        calls.append({
            "batch": pixel_values.shape[0],
            "crops": images_crop.shape[2],
            "flags": has_local_crops,
            "markers": pixel_values[:, 0, 0, 0, 0].tolist(),
            "spatial": images_spatial_crop.flatten().tolist(),
        })
        return [
            torch.column_stack((
                torch.full((images_crop.shape[2],), pixel_values[i, 0, 0, 0, 0], device=device),
                torch.arange(images_crop.shape[2], device=device),
            ))
            for i in range(pixel_values.shape[0])
        ]

    items = [
        make_item(device, 12, 1.0, True),
        make_item(device, 1, 2.0, False),
        make_item(device, 3, 3.0, True),
    ]
    expected = torch.cat([
        torch.column_stack((
            torch.full((count,), marker, device=device),
            torch.arange(count, device=device),
        ))
        for count, marker in [(12, 1.0), (1, 2.0), (3, 3.0)]
    ])
    with patch.object(UnlimitedOCRForCausalLM, "_pixel_values_to_embedding", side_effect=encoder):
        actual = model._process_image_input(items)
    torch.testing.assert_close(actual, expected)
    assert [c["batch"] for c in calls] == [1, 1, 1]
    assert [c["flags"] for c in calls] == [[True], [False], [True]]
    return {"device": str(device), "calls": calls, "output_shape": list(actual.shape)}


print("module", __import__("sglang.srt.models.unlimited_ocr", fromlist=["x"]).__file__)
results = [run(torch.device("cpu"))]
if torch.cuda.is_available():
    results.append(run(torch.device("cuda")))
print(json.dumps(results, indent=2))
