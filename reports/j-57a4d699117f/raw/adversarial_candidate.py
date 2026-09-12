from reproduce_contract import FakeChat, request

from jinja2 import Environment

from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.anthropic.tool_reference import (
    template_supports_deferred_tool_loading,
)


UNRELATED_TYPE_COMPARE = r'''
{% for tool in tools if not tool.function.defer_loading %}{{ tool.function.name }}{% endfor %}
{% if telemetry.type == "tool_reference" %}diagnostic{% endif %}
{% for message in messages %}
  {% if message.content is string %}{{ message.content }}
  {% else %}{% for item in message.content %}
    {% if item.type == "text" %}{{ item.text }}
    {% else %}{{ raise_exception("Unexpected item type in content.") }}{% endif %}
  {% endfor %}{% endif %}
{% endfor %}
'''

FULLY_BRACKETED_NATIVE = r'''
{% for tool in tools if not tool["function"]["defer_loading"] %}{{ tool["function"]["name"] }}{% endfor %}
{% for message in messages if message["role"] == "tool" %}
  {% if message["content"] is not string and message["content"][0]["type"] == "tool_reference" %}
    {% for reference in message["content"] %}
      {% for tool in tools if tool["function"]["name"] == reference["name"] %}{{ tool["function"]["name"] }}{% endfor %}
    {% endfor %}
  {% endif %}
{% endfor %}
'''


def converted(template):
    return AnthropicServing(FakeChat(template))._convert_to_chat_completion_request(
        request()
    ).model_dump(exclude_none=True)


def main():
    print("false_positive_detected=", template_supports_deferred_tool_loading(UNRELATED_TYPE_COMPARE))
    false_positive_payload = converted(UNRELATED_TYPE_COMPARE)
    print("false_positive_tool_message=", false_positive_payload["messages"][3])
    env = Environment()
    env.globals["telemetry"] = {"type": "ordinary"}
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    try:
        env.from_string(UNRELATED_TYPE_COMPARE).render(**false_positive_payload)
    except Exception as exc:
        print("false_positive_render_error=", type(exc).__name__, str(exc))

    print("bracketed_native_detected=", template_supports_deferred_tool_loading(FULLY_BRACKETED_NATIVE))
    false_negative_payload = converted(FULLY_BRACKETED_NATIVE)
    print("bracketed_native_tool_message=", false_negative_payload["messages"][3])
    print("bracketed_native_tools=", false_negative_payload["tools"])


if __name__ == "__main__":
    main()
