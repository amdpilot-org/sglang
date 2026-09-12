import json
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, List, Union

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

EmbeddingInput = Union[str, List[str], List[int], List[List[int]]]


@dataclass
class EmbeddingDataset(BaseDataset):
    """OpenAI-compatible embedding requests stored as JSON Lines."""

    dataset_path: str
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "EmbeddingDataset":
        return cls(args.dataset_path, args.num_prompts)

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_embedding_requests(
            self.dataset_path, self.num_requests, tokenizer
        )


def _validate_input(value: Any, line_number: int) -> EmbeddingInput:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        if all(isinstance(item, str) and item.strip() for item in value):
            return value
        if all(isinstance(item, int) and item >= 0 for item in value):
            return value
        if all(
            isinstance(item, list)
            and item
            and all(isinstance(token_id, int) and token_id >= 0 for token_id in item)
            for item in value
        ):
            return value
    raise ValueError(
        f"Invalid embedding input on line {line_number}: expected a non-empty "
        "string, a non-empty list of non-empty strings, a non-empty token-ID "
        "list, or a non-empty list of non-empty token-ID lists"
    )


def sample_embedding_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[DatasetRow]:
    """Load embedding API payloads without silently discarding bad records."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than zero")

    requests: List[DatasetRow] = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if len(requests) == num_requests:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid embedding request on line {line_number}: expected an object"
                )

            input_value = _validate_input(data.get("input"), line_number)
            if isinstance(input_value, str):
                prompt_len = len(tokenizer.encode(input_value))
            elif isinstance(input_value[0], str):
                prompt_len = sum(len(tokenizer.encode(text)) for text in input_value)
            elif isinstance(input_value[0], int):
                prompt_len = len(input_value)
            else:
                prompt_len = sum(len(token_ids) for token_ids in input_value)
            extra_body = {key: value for key, value in data.items() if key != "input"}
            requests.append(
                DatasetRow(
                    prompt=input_value,
                    prompt_len=prompt_len,
                    output_len=0,
                    extra_request_body=extra_body,
                )
            )

    if len(requests) < num_requests:
        raise ValueError(
            f"Embedding dataset contains {len(requests)} requests, fewer than "
            f"--num-prompts={num_requests}"
        )
    print(f"#Input tokens: {sum(row.prompt_len for row in requests)}")
    return requests
