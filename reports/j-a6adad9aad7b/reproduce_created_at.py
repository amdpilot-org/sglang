import argparse
import json

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
)


parser = argparse.ArgumentParser()
parser.add_argument("--raw-sdk", action="store_true")
args = parser.parse_args()

response = Response(
    id="resp_123",
    created_at=1786588600,
    error=None,
    incomplete_details=None,
    instructions=None,
    metadata=None,
    model="test",
    object="response",
    output=[],
    parallel_tool_calls=True,
    temperature=None,
    tool_choice="auto",
    tools=[],
    top_p=None,
)

if args.raw_sdk:
    serialize = lambda event: event.model_dump_json(indent=None)
    expected_type = float
else:
    from sglang.srt.entrypoints.openai.serving_responses import (
        _serialize_responses_event_data,
    )

    serialize = _serialize_responses_event_data
    expected_type = int

for event_class, event_type in (
    (ResponseCreatedEvent, "response.created"),
    (ResponseInProgressEvent, "response.in_progress"),
    (ResponseCompletedEvent, "response.completed"),
):
    event = event_class(response=response, sequence_number=0, type=event_type)
    wire = serialize(event)
    created_at = json.loads(wire)["response"]["created_at"]
    print(event_type, type(created_at).__name__, repr(created_at), wire)
    assert type(created_at) is expected_type
