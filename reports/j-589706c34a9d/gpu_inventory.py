"""Record assigned-device visibility without claiming model execution."""

import torch


print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0:", torch.cuda.get_device_name(0))
