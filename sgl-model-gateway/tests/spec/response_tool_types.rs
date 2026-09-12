use smg::protocols::responses::ResponsesRequest;
use validator::Validate;

fn request_with_tool_type(tool_type: &str) -> String {
    format!(
        r#"{{
            "model": "test-model",
            "input": "hello",
            "tools": [{{
                "type": "{tool_type}",
                "name": "test_tool",
                "server_url": "https://example.com/mcp"
            }}]
        }}"#
    )
}

#[test]
fn accepts_python_supported_response_tool_types() {
    for tool_type in [
        "function",
        "web_search",
        "web_search_preview",
        "code_interpreter",
        "file_search",
        "image_generation",
        "computer_use_preview",
        "local_shell",
        "mcp",
        "custom",
        "namespace",
        "tool_search",
    ] {
        serde_json::from_str::<ResponsesRequest>(&request_with_tool_type(tool_type))
            .unwrap_or_else(|error| panic!("{tool_type} should deserialize: {error}"));
    }
}

#[test]
fn custom_response_tool_passes_request_validation() {
    let request = serde_json::from_str::<ResponsesRequest>(&request_with_tool_type("custom"))
        .expect("custom response tool should deserialize");

    request
        .validate()
        .expect("custom response tool should pass gateway validation");
}

#[test]
fn rejects_unknown_response_tool_type() {
    let error = serde_json::from_str::<ResponsesRequest>(&request_with_tool_type("not_a_tool"))
        .expect_err("unknown response tool types must remain invalid");

    assert!(error.to_string().contains("unknown variant `not_a_tool`"));
}
