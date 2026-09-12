import io

import torch
from PIL import Image
from torchvision.io import decode_jpeg

from sglang.srt.utils import common


def jpeg_bytes(mode: str) -> bytes:
    color = 128 if mode == "L" else (130, 90, 60)
    buffer = io.BytesIO()
    Image.new(mode, (64, 48), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


print(f"torch={torch.__version__}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}")

decoded = {}
decode_calls = []
original_is_cuda = common.is_cuda
original_decode_jpeg = common.decode_jpeg
common.is_cuda = lambda: True


def decode_on_cpu(encoded_image, **kwargs):
    decode_calls.append(kwargs.copy())
    kwargs["device"] = "cpu"
    return original_decode_jpeg(encoded_image, **kwargs)


common.decode_jpeg = decode_on_cpu
try:
    for mode in ("L", "RGB"):
        image = common._load_image(image_bytes=jpeg_bytes(mode), gpu_image_decode=True)
        decoded[mode] = image
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else None
        device = image.device if isinstance(image, torch.Tensor) else None
        print(
            f"_load_image source={mode} type={type(image).__name__} "
            f"shape={shape} device={device}"
        )
finally:
    common.is_cuda = original_is_cuda
    common.decode_jpeg = original_decode_jpeg

print(f"decode_calls={decode_calls}")

try:
    torch.stack([decoded["L"], decoded["RGB"]])
except RuntimeError as error:
    print(f"mixed_stack_error={error}")

gray = decoded["L"].to(torch.float32) / 255
patches = gray.unfold(1, 16, 16).unfold(2, 16, 16)
patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, gray.shape[0] * 16 * 16)
patches = patches.to("cuda")
projection = torch.nn.Linear(16 * 16 * 3, 1152, device="cuda")
try:
    projected = projection(patches)
except RuntimeError as error:
    print(f"gemma4_projection_error={error}")
else:
    print(f"gemma4_projection_shape={tuple(projected.shape)}")

# Independent torchvision reference: its default mode preserves source channels.
encoded = torch.frombuffer(jpeg_bytes("L"), dtype=torch.uint8)
reference = decode_jpeg(encoded, device="cpu")
print(f"torchvision_default_gray_shape={tuple(reference.shape)}")
