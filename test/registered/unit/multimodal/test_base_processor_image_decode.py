"""Unit tests for ``BaseMultimodalProcessor._load_single_item`` image decoding.

Regression test for the change that forces the (otherwise lazy) PIL decode inside
``_load_single_item`` — which runs in the ``io_executor`` worker thread — instead of
letting it fire lazily on the main event-loop thread later (inside
``pil_to_tensor``/``tobytes`` during processing). The behavior of the returned image
(mode, pixels) must be unchanged; only *when/where* the decode happens differs.

No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import asyncio
import concurrent.futures
import io
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import common
from sglang.srt.utils.nvjpeg_decoder import _NvJpegDecoderPool
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # gpu_image_decode=False forces the PIL (CPU) path so the test needs no GPU and
    # exercises exactly the lazy-decode branch the fix targets. The abstract methods
    # are never called: we only invoke the _load_single_item classmethod.
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _png_bytes(mode: str = "RGB", size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB").convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90, subsampling=2)
    return buf.getvalue()


def _oriented_jpeg_bytes(orientation: int, size=(24, 16)) -> bytes:
    arr = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
    image = Image.fromarray(arr, "RGB")
    exif = image.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _is_decoded(img: Image.Image) -> bool:
    """A lazily-opened PIL image has no decoded core yet; ``load()`` populates it.
    PIL's ``.im`` property requires a completed load and raises otherwise."""
    try:
        return img.im is not None
    except Exception:
        return False


class TestLoadSingleItemImageDecode(CustomTestCase):
    def test_plain_open_is_lazy(self):
        # Documents why the fix matters: a bare Image.open is not decoded yet, so
        # without the fix the decode would land on the caller (main) thread.
        lazy = Image.open(io.BytesIO(_png_bytes()))
        self.assertFalse(_is_decoded(lazy))

    def test_load_single_item_forces_decode(self):
        img = _StubProcessor._load_single_item(_png_bytes("RGB"), Modality.IMAGE)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        # The fix: decode is forced inside _load_single_item, not lazily later.
        self.assertTrue(_is_decoded(img))

    def test_rgba_converted_to_rgb_and_decoded(self):
        img = _StubProcessor._load_single_item(_png_bytes("RGBA"), Modality.IMAGE)
        # Existing alpha-discard behavior preserved.
        self.assertEqual(img.mode, "RGB")
        self.assertTrue(_is_decoded(img))

    def test_pixels_match_reference(self):
        # Output must be bit-identical to the pre-fix path (open -> [convert]).
        data = _png_bytes("RGB")
        img = _StubProcessor._load_single_item(data, Modality.IMAGE)
        ref = Image.open(io.BytesIO(data)).convert("RGB")
        np.testing.assert_array_equal(np.asarray(img), np.asarray(ref))

    def test_fast_loader_preserves_invalid_input_as_value_error(self):
        processor = object.__new__(_StubProcessor)
        future = concurrent.futures.Future()
        future.set_exception(ValueError("invalid base64 image"))
        processor._submit_mm_data_loading_tasks_simple = Mock(
            side_effect=[[(Modality.IMAGE, 0, future)], [], []]
        )

        with self.assertRaisesRegex(ValueError, "invalid base64 image"):
            asyncio.run(
                processor.fast_load_mm_data(
                    prompt="<image>",
                    multimodal_tokens=Mock(),
                    image_data=["bad-image"],
                )
            )

    def test_unreachable_image_url_is_a_client_error(self):
        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_image",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            with self.assertRaisesRegex(ValueError, "connection refused"):
                _StubProcessor._load_single_item(
                    "https://127.0.0.1:1/not-an-image.png", Modality.IMAGE
                )

    def test_invalid_image_bytes_are_a_client_error(self):
        with self.assertRaisesRegex(ValueError, "cannot identify image file"):
            _StubProcessor._load_single_item(b"not an image", Modality.IMAGE)

    def test_unexpected_loader_bug_remains_a_server_error(self):
        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_image",
            side_effect=TypeError("unexpected loader bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected loader bug"):
                _StubProcessor._load_single_item(b"image", Modality.IMAGE)

    def test_high_fidelity_gpu_jpeg_decoder_is_selected(self):
        data = _jpeg_bytes()
        expected = torch.zeros((3, 8, 8), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                return_value=expected,
            ) as decode,
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")

        self.assertIs(image, expected)
        decode.assert_called_once_with(data)

    def test_high_fidelity_gpu_jpeg_decoder_falls_back_to_pil(self):
        data = _jpeg_bytes()
        common._warn_fancy_jpeg_fallback.cache_clear()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                side_effect=ImportError("nvImageCodec is unavailable"),
            ),
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")

        self.assertIsInstance(image, Image.Image)
        reference = Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))

    def test_cpu_decode_applies_all_exif_orientations(self):
        for orientation in range(1, 9):
            data = _oriented_jpeg_bytes(orientation)
            image, _ = common.load_image(data, gpu_image_decode=False)
            reference = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            np.testing.assert_array_equal(
                np.asarray(image),
                np.asarray(reference),
                err_msg=f"orientation {orientation}",
            )

    def test_direct_pil_input_applies_exif_orientation_and_updates_size(self):
        data = _oriented_jpeg_bytes(6)
        image, image_size = common.load_image(Image.open(io.BytesIO(data)))
        reference = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))
        self.assertEqual(image_size, reference.size)

    def test_rotated_jpeg_skips_both_gpu_decoders(self):
        data = _oriented_jpeg_bytes(6)
        reference = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        for mode, target in (
            (True, "sglang.srt.utils.common.decode_jpeg"),
            (
                "nvjpeg_fancy",
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
            ),
        ):
            with (
                patch.object(common, "is_cuda", return_value=True),
                patch(target) as decode,
            ):
                image, _ = common.load_image(data, gpu_image_decode=mode)
            decode.assert_not_called()
            np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))

    def test_unrotated_jpeg_keeps_both_gpu_decoder_routes(self):
        data = _oriented_jpeg_bytes(1)
        expected = torch.zeros((3, 16, 24), dtype=torch.uint8)
        for mode, target in (
            (True, "sglang.srt.utils.common.decode_jpeg"),
            (
                "nvjpeg_fancy",
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
            ),
        ):
            with (
                patch.object(common, "is_cuda", return_value=True),
                patch(target, return_value=expected) as decode,
            ):
                image, _ = common.load_image(data, gpu_image_decode=mode)
            decode.assert_called_once()
            self.assertIs(image, expected)

    def test_invalid_orientation_is_not_transformed(self):
        data = _oriented_jpeg_bytes(99)
        image, _ = common.load_image(data, gpu_image_decode=False)
        reference = Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))

    def test_no_orientation_preserves_direct_pil_object(self):
        image = Image.new("RGB", (12, 10), "red")
        result, image_size = common.load_image(image, gpu_image_decode=False)
        self.assertIs(result, image)
        self.assertEqual(image_size, image.size)

    def test_high_fidelity_decoder_uses_fancy_planar_rgb_and_reuses_pool(self):
        expected = torch.zeros((3, 8, 8), dtype=torch.uint8)
        fake_format = object()

        class FakeImage:
            def to_dlpack(self, *, cuda_stream):
                self.cuda_stream = cuda_stream
                return object()

        class FakeDecoder:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.instances.append(self)

            def decode(self, data, *, params, cuda_stream):
                self.call = (data, params, cuda_stream)
                return FakeImage()

        class FakeDecodeParams:
            def __init__(self, *, sample_format, apply_exif_orientation):
                self.sample_format = sample_format
                self.apply_exif_orientation = apply_exif_orientation

        fake_codec = SimpleNamespace(
            DecodeParams=FakeDecodeParams,
            Decoder=FakeDecoder,
            SampleFormat=SimpleNamespace(P_RGB=fake_format),
        )
        nvidia = types.ModuleType("nvidia")
        nvidia.nvimgcodec = fake_codec

        with (
            patch.dict(sys.modules, {"nvidia": nvidia}),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=7),
            ),
            patch.object(torch, "from_dlpack", return_value=expected),
        ):
            pool = _NvJpegDecoderPool(device_id=2)
            self.assertIs(pool.decode(b"jpeg"), expected)
            self.assertIs(pool.decode(b"jpeg"), expected)

        self.assertEqual(len(FakeDecoder.instances), 1)
        decoder = FakeDecoder.instances[0]
        self.assertEqual(decoder.kwargs["device_id"], 2)
        self.assertEqual(decoder.kwargs["max_num_cpu_threads"], 1)
        self.assertIn(":fancy_upsampling=1", decoder.kwargs["options"])
        self.assertIs(pool._decode_params.sample_format, fake_format)
        self.assertFalse(pool._decode_params.apply_exif_orientation)


if __name__ == "__main__":
    unittest.main()
