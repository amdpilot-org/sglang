# Investigation evidence

Source path: `python/sglang/srt/entrypoints/openai/serving_base.py`

Test paths:

- `test/registered/unit/entrypoints/openai/test_serving_base.py`
- `test/registered/unit/entrypoints/openai/test_serving_chat.py`
- `test/registered/unit/entrypoints/openai/test_serving_transcription.py`
- `test/registered/openai_server/basic/test_openai_server.py`

No native library was changed or rebuilt.

## Related change check

The source issue is open. Upstream PR
https://github.com/sgl-project/sglang/pull/33534 is also open and reports the same
one-line correction; it was `BLOCKED` and unmerged when inspected. The prepared
base still contained the defect.

## Before-fix reproduction

Calling the actual prepared checkout's response helpers produced:

```text
400 {'object': 'error', 'message': 'max_tokens=999999 cannot be greater than 4096', 'type': 'BadRequestError', 'param': 'max_tokens', 'code': 400}
422 {'object': 'error', 'message': 'unprocessable', 'type': 'ValidationError', 'param': None, 'code': 422}
streaming {'error': {'object': 'error', 'message': 'bad', 'type': 'BadRequestError', 'param': None, 'code': 400}}
```

This isolates the inconsistency to non-streaming serialization without requiring
model weights or GPU execution.

## Failing-before regression

With the new regression present and the source line temporarily restored to the
recorded base implementation:

```text
FFF [100%]
FAILED ...test_non_streaming_error_preserves_non_default_fields - KeyError: 'error'
FAILED ...test_non_streaming_error_uses_openai_envelope - AssertionError: flat body != {'error': ...}
FAILED ...test_streaming_and_non_streaming_error_shapes_match - AssertionError: flat body != {'error': ...}
3 failed, 20 warnings in 9.15s
```

After restoring the correction, the focused caller suites reported:

```text
160 passed, 61 warnings, 68 subtests passed in 9.38s
```

## OpenAI SDK check

The installed OpenAI client received the corrected helper response through
`httpx.MockTransport` and reported:

```text
BadRequestError
Error code: 400 - {'error': {'object': 'error', 'message': 'max_tokens=999999 cannot be greater than 4096', 'type': 'BadRequestError', 'param': 'max_tokens', 'code': 400}}
{'object': 'error', 'message': 'max_tokens=999999 cannot be greater than 4096', 'type': 'BadRequestError', 'param': 'max_tokens', 'code': 400}
```

The final line is `BadRequestError.body`, confirming that the SDK extracted the
inner error object, including `param` and the original message.
