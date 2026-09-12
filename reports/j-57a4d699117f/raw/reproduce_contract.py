from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from jinja2 import Environment

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing


STRICT_TEMPLATE = r'''
{% macro render_content(content) %}
  {% if content is string %}{{ content }}
  {% else %}{% for item in content %}
    {% if item.type == "text" %}{{ item.text }}
    {% else %}{{ raise_exception("Unexpected item type in content.") }}{% endif %}
  {% endfor %}{% endif %}
{% endmacro %}
{% for message in messages %}{{ render_content(message.content) }}{% endfor %}
'''

UNRELATED_TEMPLATE = r'''
{% set diagnostic_label = "tool_reference" %}
{% for tool in tools if not tool.function.defer_loading %}{{ tool.function.name }}{% endfor %}
{% for message in messages %}
  {% if message.content is string %}{{ message.content }}
  {% else %}{% for item in message.content %}
    {% if item.type == "text" %}{{ item.text }}
    {% else %}{{ raise_exception("Unexpected item type in content.") }}{% endif %}
  {% endfor %}{% endif %}
{% endfor %}
'''


class FakeChat:
    def __init__(self, template):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=template)
        )


def request():
    return AnthropicMessagesRequest.model_validate(
        {
            "model": "fixture",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "list deferred tools"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "ToolSearch",
                            "input": {"query": "select:DemoTool"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "Found 1 tool:"},
                                {"type": "tool_reference", "tool_name": "DemoTool"},
                            ],
                        }
                    ],
                },
                {"role": "user", "content": "now use it"},
            ],
            "tools": [
                {
                    "name": "ToolSearch",
                    "description": "search",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": False,
                },
                {
                    "name": "DemoTool",
                    "description": "demo",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                },
            ],
        }
    )


def run(template):
    converted = AnthropicServing(FakeChat(template))._convert_to_chat_completion_request(
        request()
    ).model_dump(exclude_none=True)
    print("converted_tool_message=", converted["messages"][2])
    print("forwarded_tools=", [t["function"]["name"] for t in converted["tools"]])
    env = Environment()
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        ValueError(message)
    )
    print("rendered=", env.from_string(template).render(**converted))


if __name__ == "__main__":
    import sys

    run(UNRELATED_TEMPLATE if sys.argv[1:] == ["unrelated"] else STRICT_TEMPLATE)
