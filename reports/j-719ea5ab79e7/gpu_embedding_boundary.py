"""gfx950 evidence for the GLM-5.3 NextN multimodal embedding boundary."""

import json

import torch


VOCAB_SIZE = 154880
HIDDEN_SIZE = 8


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, device=device)

    # Two independent requests. Their interior multimodal positions contain
    # sentinels, while the final appended token of each request is in-vocabulary.
    input_ids = torch.tensor(
        [11, 1_000_003, 13, 17, 1_000_009, 19], device=device
    )
    extend_start_loc = torch.tensor([0, 3], device=device)
    extend_seq_lens = torch.tensor([3, 3], device=device)
    last_indices = extend_start_loc + extend_seq_lens - 1

    actual = embedding(input_ids[last_indices])
    expected = embedding.weight.detach().cpu()[torch.tensor([13, 19])]
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
                "all_input_ids": input_ids.cpu().tolist(),
                "selected_last_indices": last_indices.cpu().tolist(),
                "embedded_ids": input_ids[last_indices].cpu().tolist(),
                "max_embedded_id": int(input_ids[last_indices].max()),
                "vocab_size": VOCAB_SIZE,
                "matches_independent_cpu_gather": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
