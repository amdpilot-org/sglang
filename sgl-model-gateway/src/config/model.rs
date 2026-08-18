use serde::{Deserialize, Serialize};

/// Configuration for the model and its request/output adapters.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub chat_template: Option<String>,
    pub tokenizer_cache: TokenizerCacheConfig,
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            chat_template: None,
            tokenizer_cache: TokenizerCacheConfig::default(),
            reasoning_parser: None,
            tool_call_parser: None,
        }
    }
}

/// Tokenizer cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerCacheConfig {
    /// Whole-string exact match cache
    #[serde(default = "default_enable_l0")]
    pub enable_l0: bool,
    #[serde(default = "default_l0_max_entries")]
    pub l0_max_entries: usize,
    /// Prefix matching at fixed boundaries
    #[serde(default = "default_enable_l1")]
    pub enable_l1: bool,
    #[serde(default = "default_l1_max_memory")]
    pub l1_max_memory: usize,
}

fn default_enable_l0() -> bool {
    false
}

fn default_l0_max_entries() -> usize {
    10_000
}

fn default_enable_l1() -> bool {
    false
}

fn default_l1_max_memory() -> usize {
    50 * 1024 * 1024 // 50MB
}

impl TokenizerCacheConfig {
    /// Returns Some(self) if any caching is enabled, None otherwise.
    /// Use this when passing cache config to tokenizer registration workflow.
    pub fn to_option(&self) -> Option<Self> {
        if self.enable_l0 || self.enable_l1 {
            Some(self.clone())
        } else {
            None
        }
    }
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: default_enable_l0(),
            l0_max_entries: default_l0_max_entries(),
            enable_l1: default_enable_l1(),
            l1_max_memory: default_l1_max_memory(),
        }
    }
}
