"""Assert the observable contract of the VILA 1.5 TinyChat AWQ artifact."""

import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()

state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
qweights = {name: value for name, value in state.items() if name.endswith(".qweight")}
scaled_zeros = {
    name: value for name, value in state.items() if name.endswith(".scaled_zeros")
}
qzeros = {name: value for name, value in state.items() if name.endswith(".qzeros")}

assert len(state) == 739, len(state)
assert qweights and all(value.dtype == torch.int16 for value in qweights.values())
assert len(scaled_zeros) == len(qweights)
assert not qzeros

sample_name, sample = next(iter(qweights.items()))
print(f"tensors={len(state)}")
print(f"qweights={len(qweights)} dtype={sample.dtype}")
print(f"scaled_zeros={len(scaled_zeros)} qzeros={len(qzeros)}")
print(f"sample={sample_name} shape={tuple(sample.shape)}")
