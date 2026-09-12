import torch


print(
    {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
    }
)
try:
    torch.tensor([0, None, 1], dtype=torch.int64, pin_memory=True)
except TypeError as exc:
    print(f"{type(exc).__name__}: {exc}")
else:
    raise AssertionError("the historical expression unexpectedly succeeded")
