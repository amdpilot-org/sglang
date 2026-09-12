import asyncio
from unittest.mock import Mock

from utils import make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ResponsesResponse

serving = make_serving()
serving.tokenizer_manager.abort_request = Mock()

response = ResponsesResponse.from_request(
    ResponsesRequest(
        model="x",
        input="hi",
        background=False,
        store=True,
    ),
    sampling_params={},
    model_name="x",
    created_time=0,
    output=[],
    status="completed",
    usage=None,
)
serving.response_store[response.id] = response

result = asyncio.run(serving.cancel_responses(response.id))

print("http_status:", getattr(result, "status_code", 200))
print("response_status:", getattr(result, "status", None))
print("abort_called:", serving.tokenizer_manager.abort_request.called)
