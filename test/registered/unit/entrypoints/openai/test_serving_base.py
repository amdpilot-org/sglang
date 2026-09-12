"""Unit tests for OpenAIServingBase error response serialization."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import json
import unittest

import orjson

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubServing(OpenAIServingBase):
    def _request_id_prefix(self) -> str:
        return "stub-"

    def _convert_to_internal_request(self, request, raw_request=None):
        raise NotImplementedError


class TestCreateErrorResponse(unittest.TestCase):
    def setUp(self):
        self.serving = object.__new__(_StubServing)

    @staticmethod
    def _body(response):
        return orjson.loads(response.body)

    def test_non_streaming_error_uses_openai_envelope(self):
        response = self.serving.create_error_response(
            "max_tokens=999999 cannot be greater than the model context length",
            param="max_tokens",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            self._body(response),
            {
                "error": {
                    "object": "error",
                    "message": (
                        "max_tokens=999999 cannot be greater than the model "
                        "context length"
                    ),
                    "type": "BadRequestError",
                    "param": "max_tokens",
                    "code": 400,
                }
            },
        )

    def test_non_streaming_error_preserves_non_default_fields(self):
        response = self.serving.create_error_response(
            "unprocessable",
            err_type="ValidationError",
            status_code=422,
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self._body(response)["error"],
            {
                "object": "error",
                "message": "unprocessable",
                "type": "ValidationError",
                "param": None,
                "code": 422,
            },
        )

    def test_streaming_and_non_streaming_error_shapes_match(self):
        kwargs = {
            "message": "bad request",
            "err_type": "BadRequestError",
            "status_code": 400,
        }

        non_streaming = self._body(self.serving.create_error_response(**kwargs))
        streaming = json.loads(self.serving.create_streaming_error_response(**kwargs))

        self.assertEqual(non_streaming, streaming)


if __name__ == "__main__":
    unittest.main(verbosity=2)
