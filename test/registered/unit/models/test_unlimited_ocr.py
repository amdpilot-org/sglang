import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM


class TestUnlimitedOCRProcessImageInput(unittest.TestCase):
    @staticmethod
    def _make_item(device, num_patches, tiles_w, tiles_h, marker, has_local_crops):
        item = MultimodalDataItem(modality=Modality.IMAGE)
        item.feature = torch.full(
            (1, 3, 2, 2), marker, dtype=torch.float32, device=device
        )
        item.images_crop = torch.zeros(
            1, num_patches, 3, 2, 2, dtype=torch.float32, device=device
        )
        item.images_spatial_crop = torch.tensor(
            [[[tiles_w, tiles_h]]], dtype=torch.long, device=device
        )
        item.has_local_crops = has_local_crops
        return item

    def _run_case(self, device, patch_counts, flags, expected_batch_sizes):
        items = [
            self._make_item(
                device,
                num_patches=count,
                tiles_w=count,
                tiles_h=1,
                marker=index + 1,
                has_local_crops=flags[index],
            )
            for index, count in enumerate(patch_counts)
        ]
        calls = []

        def fake_encode(
            pixel_values, images_crop, images_spatial_crop, has_local_crops
        ):
            calls.append(
                (
                    pixel_values.shape[0],
                    images_crop.shape[2],
                    has_local_crops,
                    pixel_values[:, 0, 0, 0, 0].tolist(),
                )
            )
            return [
                torch.full(
                    (images_crop.shape[2], 2),
                    pixel_values[index, 0, 0, 0, 0],
                    device=device,
                )
                for index in range(images_crop.shape[0])
            ]

        instance = UnlimitedOCRForCausalLM.__new__(UnlimitedOCRForCausalLM)

        class _VisionStub:
            dtype = torch.float32

        instance.vision_model = _VisionStub()
        with patch.object(
            UnlimitedOCRForCausalLM,
            "_pixel_values_to_embedding",
            side_effect=fake_encode,
            autospec=False,
        ):
            output = instance._process_image_input(items)

        self.assertEqual([call[0] for call in calls], expected_batch_sizes)
        self.assertEqual(
            output[:, 0].tolist(),
            [1.0] * patch_counts[0] + [2.0] * patch_counts[1],
        )
        return calls

    def test_heterogeneous_crop_counts_are_encoded_in_order(self):
        calls = self._run_case(
            torch.device("cpu"), [12, 1], [True, False], [1, 1]
        )
        self.assertEqual([call[2] for call in calls], [[True], [False]])

    def test_equal_crop_counts_keep_batched_path(self):
        calls = self._run_case(
            torch.device("cpu"), [2, 2], [True, True], [2]
        )
        self.assertEqual(calls[0][2], [True, True])

    def test_heterogeneous_crop_counts_without_explicit_crop_flags(self):
        calls = self._run_case(
            torch.device("cpu"), [2, 1], [None, None], [1, 1]
        )
        self.assertEqual([call[2] for call in calls], [None, None])

    @unittest.skipUnless(torch.cuda.is_available(), "requires an available GPU")
    def test_heterogeneous_crop_counts_on_gpu(self):
        self._run_case(torch.device("cuda"), [3, 1], [True, True], [1, 1])


if __name__ == "__main__":
    unittest.main()
