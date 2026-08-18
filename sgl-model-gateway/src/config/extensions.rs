/// Configuration for optional gateway extensions.
#[derive(Debug, Clone)]
pub struct ExtensionConfig {
    pub mcp_config: Option<smg_mcp::McpConfig>,
    pub enable_wasm: bool,
}

impl Default for ExtensionConfig {
    fn default() -> Self {
        Self {
            mcp_config: None,
            enable_wasm: false,
        }
    }
}
