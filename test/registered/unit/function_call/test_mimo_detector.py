import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMiMoDetector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                ),
            )
        ]
        self.tool_call = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

    def test_streaming_emits_text_after_completed_tool_call(self):
        detector = MiMoDetector()
        call_result = detector.parse_streaming_increment(self.tool_call, self.tools)
        text_result = detector.parse_streaming_increment(
            "Done, the files are listed.", self.tools
        )

        self.assertEqual([call.name for call in call_result.calls], ["search"])
        self.assertEqual(text_result.normal_text, "Done, the files are listed.")
        self.assertEqual(detector._buffer, "")

    def test_streaming_recognizes_tool_call_at_every_marker_split(self):
        prefix = "I will run it.\n"

        for split in range(1, len("<tool_call>")):
            with self.subTest(split=split):
                detector = MiMoDetector()
                first = detector.parse_streaming_increment(
                    prefix + self.tool_call[:split], self.tools
                )
                second = detector.parse_streaming_increment(
                    self.tool_call[split:], self.tools
                )

                self.assertEqual(first.normal_text, prefix)
                self.assertEqual(first.calls, [])
                self.assertEqual([call.name for call in second.calls], ["search"])
                self.assertEqual(second.normal_text, "")

    def test_streaming_only_holds_partial_marker_suffix(self):
        detector = MiMoDetector()

        result = detector.parse_streaming_increment("ordinary text<tool", self.tools)

        self.assertEqual(result.normal_text, "ordinary text")
        self.assertEqual(detector._buffer, "<tool")

    def test_streaming_emits_text_between_tool_calls(self):
        detector = MiMoDetector()

        first_call = detector.parse_streaming_increment(self.tool_call, self.tools)
        text = detector.parse_streaming_increment(
            "Next, I will search again. ", self.tools
        )
        second_call = detector.parse_streaming_increment(self.tool_call, self.tools)

        self.assertEqual([call.tool_index for call in first_call.calls], [0])
        self.assertEqual(text.normal_text, "Next, I will search again. ")
        self.assertEqual([call.tool_index for call in second_call.calls], [1])

    def test_final_chunk_emits_call_and_trailing_text(self):
        parser = FunctionCallParser(self.tools, "mimo")

        chunk_text, chunk_calls = parser.parse_stream_chunk(self.tool_call + "After.")
        end_text, end_calls = parser.parse_stream_end()

        self.assertEqual(chunk_text + end_text, "After.")
        self.assertEqual([call.name for call in chunk_calls + end_calls], ["search"])
        self.assertEqual(parser.detector._buffer, "")

    def test_final_chunk_drains_multiple_calls_and_surrounding_text(self):
        parser = FunctionCallParser(self.tools, "mimo")
        generated = "Before." + self.tool_call + "Between." + self.tool_call + "After."

        chunk_text, chunk_calls = parser.parse_stream_chunk(generated)
        end_text, end_calls = parser.parse_stream_end()

        self.assertEqual(chunk_text + end_text, "Before.Between.After.")
        self.assertEqual(
            [call.name for call in chunk_calls + end_calls], ["search", "search"]
        )
        self.assertEqual([call.tool_index for call in chunk_calls + end_calls], [0, 1])
        self.assertEqual(parser.detector._buffer, "")

    def test_non_stream_preserves_text_around_multiple_calls(self):
        detector = MiMoDetector()
        generated = "Before." + self.tool_call + "Between." + self.tool_call + "After."

        result = detector.detect_and_parse(generated, self.tools)

        self.assertEqual(result.normal_text, "Before.Between.After.")
        self.assertEqual([call.name for call in result.calls], ["search", "search"])

    def test_streaming_matches_non_stream_at_every_two_chunk_boundary(self):
        generated = "Before." + self.tool_call + "Between." + self.tool_call + "After."
        expected = MiMoDetector().detect_and_parse(generated, self.tools)

        for split in range(len(generated) + 1):
            with self.subTest(split=split):
                parser = FunctionCallParser(self.tools, "mimo")
                text_parts = []
                calls = []
                for chunk in (generated[:split], generated[split:]):
                    text, chunk_calls = parser.parse_stream_chunk(chunk)
                    text_parts.append(text)
                    calls.extend(chunk_calls)
                text, end_calls = parser.parse_stream_end()
                text_parts.append(text)
                calls.extend(end_calls)

                self.assertEqual("".join(text_parts), expected.normal_text)
                self.assertEqual(
                    [call.name for call in calls],
                    [call.name for call in expected.calls],
                )
                self.assertEqual(parser.detector._buffer, "")

    def test_stream_end_releases_incomplete_marker_as_text(self):
        parser = FunctionCallParser(self.tools, "mimo")

        chunk_text, chunk_calls = parser.parse_stream_chunk("ordinary text<tool")
        end_text, end_calls = parser.parse_stream_end()

        self.assertEqual(chunk_text + end_text, "ordinary text<tool")
        self.assertEqual(chunk_calls + end_calls, [])
        self.assertEqual(parser.detector._buffer, "")


if __name__ == "__main__":
    unittest.main()
