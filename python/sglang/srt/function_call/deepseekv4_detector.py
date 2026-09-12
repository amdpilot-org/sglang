import json
import logging
from typing import Any

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"

    def _stream_partial_parameters(self, invoke_content: str) -> bool:
        # V4 occasionally wraps the real arguments in an `arguments`/`args`
        # envelope. Wait for the complete object so schema-aware repair cannot
        # contradict argument fragments already sent to the client.
        return not invoke_content.lstrip().startswith("{")

    @staticmethod
    def _unwrap_argument_envelope(value: Any, property_names: set[str]) -> Any:
        """Unwrap a model-added envelope only when the tool schema disambiguates it."""
        current = value
        while isinstance(current, dict) and len(current) == 1:
            envelope = next(iter(current))
            if envelope not in {"arguments", "args"} or envelope in property_names:
                break
            candidate = current[envelope]
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    break
            if not isinstance(candidate, dict) or not candidate:
                break
            if not set(candidate).issubset(property_names):
                # A further repeated envelope may need peeling before its keys
                # can be checked against the actual function schema.
                if set(candidate) not in ({"arguments"}, {"args"}):
                    break
            current = candidate
        return current

    def _normalize_parameters(self, func_name, parameters, tools):
        try:
            parsed = json.loads(parameters)
        except (json.JSONDecodeError, TypeError):
            return parameters

        tool = next((t for t in tools if t.function.name == func_name), None)
        if tool is None:
            return parameters
        property_names = set(get_schema_properties(tool.function.parameters))
        if not property_names:
            return parameters

        normalized = self._unwrap_argument_envelope(parsed, property_names)
        return json.dumps(normalized, ensure_ascii=False)
