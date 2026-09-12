import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner


class Head:
    with_markov = True

    def __call__(self, hidden, markov):
        return hidden.float().sum(-1) + markov.float().sum(-1)

    @staticmethod
    def apply_sts(value):
        return torch.sigmoid(value)


class Markov:
    def get_prev_embeddings(self, tokens):
        return torch.stack((tokens.float(), -tokens.float()), dim=-1)


assert torch.cuda.is_available()
device = torch.device("cuda:0")
model = object.__new__(DeepseekV4ForCausalLMDSpark)
model.gamma = 5
model.confidence_head = Head()
model.markov_head = Markov()
planner = object.__new__(DSparkVerifyPlanner)
planner.draft_model = model
planner.gamma = 7
planner._confidence_head = model.confidence_head
anchor = torch.tensor([2, 3], device=device)
sampled = torch.arange(14, device=device).view(2, 7)
hidden = torch.arange(56, dtype=torch.float32, device=device).view(14, 4) / 10
actual = planner.compute_confidence_tensor(
    draft_hidden=None,
    anchor_tokens=anchor,
    draft_tokens=sampled,
    confidence_tap=hidden,
)
previous = torch.cat((anchor[:, None], sampled[:, :6]), dim=1)
markov = torch.stack((previous.float(), -previous.float()), dim=-1)
expected = torch.sigmoid(hidden.view(2, 7, 4).sum(-1) + markov.sum(-1))
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print(f"device={torch.cuda.get_device_name(0)}")
print(
    f"checkpoint_gamma={model.gamma} runtime_gamma={planner.gamma} "
    f"output_shape={tuple(actual.shape)}"
)
print(
    f"max_abs_error={(actual - expected).abs().max().item()} "
    f"checksum={actual.sum().item():.9f}"
)
