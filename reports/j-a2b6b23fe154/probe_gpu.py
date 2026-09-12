import json

import torch


device = torch.device("cuda", 0)
x = torch.arange(8, device=device, dtype=torch.float32)
y = x.square() + 1
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
            "result": y.cpu().tolist(),
        },
        sort_keys=True,
    )
)
