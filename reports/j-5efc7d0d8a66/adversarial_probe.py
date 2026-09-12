from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

manager = TokenizerManager.__new__(TokenizerManager)
manager.context_len = 10
manager.num_reserved_tokens = 11
manager.allow_auto_truncate = True
manager.validate_total_tokens = True
manager.is_generation = True
manager._validate_token_ids_logprob = Mock()
request = GenerateReqInput(input_ids=[1], sampling_params={"max_new_tokens": 1})

try:
    with patch(
        "sglang.srt.managers.tokenizer_manager.get_server_return_hidden_states_mode",
        return_value=CaptureHiddenMode.NULL,
    ):
        manager._validate_one_request(request, request.input_ids)
except ValueError as error:
    print({"rejected": str(error), "context_len": manager.context_len})
    raise SystemExit(0)

total = (
    len(request.input_ids)
    + manager.num_reserved_tokens
    + request.sampling_params["max_new_tokens"]
)
result = {
    "prompt": len(request.input_ids),
    "reserved": manager.num_reserved_tokens,
    "completion": request.sampling_params["max_new_tokens"],
    "accounted_total": total,
    "context_len": manager.context_len,
}
print(result)
assert total <= manager.context_len, (
    f"accepted total {total} exceeds context {manager.context_len}"
)
