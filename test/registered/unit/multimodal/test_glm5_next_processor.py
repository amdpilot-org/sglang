import base64
import io

import pytest
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.configs.glm5_next import Glm5NextConfig
from sglang.srt.configs.glm5_next_processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize,
)
from sglang.srt.multimodal.customized_mm_processor_utils import (
    _CUSTOMIZED_MM_PROCESSOR,
)
from sglang.srt.utils import load_image


def _jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_glm5_next_uses_vision_processor():
    assert _CUSTOMIZED_MM_PROCESSOR[Glm5NextConfig.model_type] is Glm5NextProcessor


def test_glm5_next_jpeg_data_url_reaches_image_processor():
    source = Image.new("RGB", (64, 48), color=(20, 90, 170))
    decoded, _ = load_image(_jpeg_data_url(source), gpu_image_decode=False)

    output = Glm5NextImageProcessor()(images=decoded, return_tensors="pt")

    assert decoded.size == (64, 48)
    assert output.image_grid_thw.tolist() == [[1, 8, 10]]
    assert output.pixel_values.shape == (80, 1176)
    assert output.pixel_values.isfinite().all()


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (28, 28, (112, 112)),
        (48, 64, (112, 140)),
        (4096, 4096, (2492, 2492)),
    ],
)
def test_glm5_next_smart_resize_boundaries(height, width, expected):
    resized = smart_resize(num_frames=2, height=height, width=width)
    assert resized == expected
    assert resized[0] % 28 == 0
    assert resized[1] % 28 == 0


def test_glm5_next_smart_resize_rejects_impossible_budget():
    with pytest.raises(ValueError, match="too small"):
        smart_resize(
            num_frames=2,
            height=64,
            width=64,
            temporal_factor=2,
            factor=28,
            max_pixels=0,
        )
