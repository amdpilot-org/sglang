use serde_json::{json, Value};
use smg::protocols::responses::ResponsesRequest;

fn roundtrip(tool: Value) -> Value {
    let request = json!({"model": "test-model", "input": "hello", "tools": [tool]});
    let parsed: ResponsesRequest = serde_json::from_value(request).unwrap();
    serde_json::to_value(parsed).unwrap()["tools"][0].clone()
}

#[test]
fn preserves_reported_custom_tool_payload() {
    let tool = json!({"type": "custom", "name": "test_tool", "description": "test"});
    assert_eq!(roundtrip(tool.clone()), tool);
}

#[test]
fn preserves_python_namespace_tool_payload() {
    let tool = json!({
        "type": "namespace",
        "name": "ops",
        "description": "nested tools",
        "tools": [{"type": "custom", "name": "shell", "description": "run"}]
    });
    assert_eq!(roundtrip(tool.clone()), tool);
}
