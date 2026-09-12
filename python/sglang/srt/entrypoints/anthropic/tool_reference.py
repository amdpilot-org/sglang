"""Tool-reference compatibility decisions for Anthropic requests.

CALLING SPEC:
    native = template_supports_deferred_tool_loading(chat_template)
    part = make_tool_reference_part(name, native_support=native)
    visible = should_forward_tool(
        name=name,
        defer_loading=defer_loading,
        referenced_names=referenced_names,
        native_support=native,
    )

Inputs and outputs are plain values. Functions are deterministic and have no
side effects, so template capability and deferred-tool routing can be tested
without constructing a serving stack.
"""

from collections.abc import Mapping
from typing import Any

import jinja2
import transformers.utils.chat_template_utils as hf_chat_utils


def _template_sources(chat_template: Any) -> list[str]:
    """Return string template sources from tokenizer template configuration."""
    if isinstance(chat_template, str):
        return [chat_template]
    if isinstance(chat_template, Mapping):
        return [value for value in chat_template.values() if isinstance(value, str)]
    return []


def _is_tool_reference_type_check(node: jinja2.nodes.Compare) -> bool:
    """Return whether a comparison dispatches on a tool_reference part type."""
    expressions = [node.expr, *(operand.expr for operand in node.ops)]
    has_reference = any(
        isinstance(expression, jinja2.nodes.Const)
        and expression.value == "tool_reference"
        for expression in expressions
    )
    has_type_access = any(
        (
            isinstance(expression, jinja2.nodes.Getattr)
            and expression.attr == "type"
        )
        or (
            isinstance(expression, jinja2.nodes.Getitem)
            and isinstance(expression.arg, jinja2.nodes.Const)
            and expression.arg.value == "type"
        )
        for expression in expressions
    )
    return has_reference and has_type_access


def template_supports_deferred_tool_loading(chat_template: Any) -> bool:
    """Return whether a template implements native deferred-tool expansion.

    Jinja comments are absent from the parsed AST, avoiding the false positive
    caused by a raw substring check. Requiring both protocol fields also keeps
    templates that merely render a reference as text on the generic path.
    """
    for source in _template_sources(chat_template):
        try:
            compiled = hf_chat_utils._compile_jinja_template(source)
            template_ast = compiled.environment.parse(source)
        except (jinja2.TemplateError, TypeError, ValueError):
            continue
        attributes = {node.attr for node in template_ast.find_all(jinja2.nodes.Getattr)}
        handles_reference_parts = any(
            _is_tool_reference_type_check(node)
            for node in template_ast.find_all(jinja2.nodes.Compare)
        )
        if handles_reference_parts and "defer_loading" in attributes:
            return True
    return False


def make_tool_reference_part(name: str, *, native_support: bool) -> dict[str, str]:
    """Build a native reference or a text marker for a generic template."""
    if native_support:
        return {"type": "tool_reference", "name": name}
    return {"type": "text", "text": f"[tool reference: {name}]"}


def should_forward_tool(
    *,
    name: str,
    defer_loading: bool | None,
    referenced_names: set[str],
    native_support: bool,
) -> bool:
    """Return whether a tool belongs in the converted request.

    Native templates receive the complete catalog and expand referenced tools
    inline. Generic templates receive only immediately available tools plus
    deferred tools already discovered in the conversation history.
    """
    return native_support or defer_loading is not True or name in referenced_names
