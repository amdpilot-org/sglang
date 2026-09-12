"""Emit a real OTLP span and verify its exported service.name resource value."""

import argparse
import os
import time

from sglang.srt.observability import trace as tracing
from sglang.test.otel_collector import LightweightOtlpCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configured-name")
    parser.add_argument("--environment-name")
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()

    if args.environment_name is None:
        os.environ.pop("OTEL_SERVICE_NAME", None)
    else:
        os.environ["OTEL_SERVICE_NAME"] = args.environment_name

    collector = LightweightOtlpCollector(port=14317)
    collector.start()
    try:
        tracing.process_tracing_init("127.0.0.1:14317", args.configured_name)
        with tracing.tracer.start_as_current_span("service-name-probe"):
            pass

        deadline = time.time() + 5
        while time.time() < deadline and not collector.get_spans():
            time.sleep(0.05)
        assert collector.get_spans(), "collector received no spans"

        resource_spans = collector._raw_traces[0]["resource_spans"]
        attributes = resource_spans[0]["resource"]["attributes"]
        resource = {
            item["key"]: item["value"].get("string_value") for item in attributes
        }
        actual = resource["service.name"]
        assert actual == args.expected, resource
        print(f"service.name={actual}")
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
