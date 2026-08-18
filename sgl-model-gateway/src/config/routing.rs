use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Configuration for request admission and worker selection.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    pub mode: RoutingMode,
    pub policy: PolicyConfig,
    pub dp_aware: bool,
    pub enable_igw: bool,
    /// Set to -1 to disable concurrent-request limiting.
    pub max_concurrent_requests: i32,
    pub queue_size: usize,
    pub queue_timeout_secs: u64,
    pub rate_limit_tokens_per_second: Option<i32>,
}

impl RoutingConfig {
    /// Creates routing configuration with the legacy router defaults.
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            mode,
            policy,
            ..Default::default()
        }
    }

    pub fn is_pd_mode(&self) -> bool {
        self.mode.is_pd_mode()
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Regular {
                worker_urls: Vec::new(),
            },
            policy: PolicyConfig::Random,
            dp_aware: false,
            enable_igw: false,
            max_concurrent_requests: -1,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
        }
    }
}

/// Routing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    #[serde(rename = "regular")]
    Regular { worker_urls: Vec<String> },
    #[serde(rename = "prefill_decode")]
    PrefillDecode {
        /// With optional bootstrap ports
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
    #[serde(rename = "openai")]
    OpenAI { worker_urls: Vec<String> },
}

impl RoutingMode {
    pub fn is_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::PrefillDecode { .. })
    }

    pub fn worker_count(&self) -> usize {
        match self {
            RoutingMode::Regular { worker_urls } => worker_urls.len(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls.len() + decode_urls.len(),
            RoutingMode::OpenAI { .. } => 1,
        }
    }

    /// Get the effective prefill policy for PD mode
    /// Falls back to the main policy if no specific prefill policy is set
    pub fn get_prefill_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective decode policy for PD mode
    /// Falls back to the main policy if no specific decode policy is set
    pub fn get_decode_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }
}

/// Assignment mode for manual policy when encountering a new routing key
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum ManualAssignmentMode {
    /// Random selection (default)
    #[default]
    Random,
    /// Select worker with minimum running requests
    MinLoad,
    /// Select worker with minimum active routing keys
    MinGroup,
}

/// Policy configuration for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyConfig {
    #[serde(rename = "random")]
    Random,

    #[serde(rename = "round_robin")]
    RoundRobin,

    #[serde(rename = "cache_aware")]
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    },

    #[serde(rename = "power_of_two")]
    PowerOfTwo { load_check_interval_secs: u64 },

    #[serde(rename = "bucket")]
    Bucket {
        /// Absolute load difference threshold for load balancing
        balance_abs_threshold: usize,
        /// Relative load ratio threshold for load balancing
        balance_rel_threshold: f32,
        /// Interval between bucket boundary adjustment cycles (seconds)
        bucket_adjust_interval_secs: usize,
    },

    /// Manual routing policy with sticky sessions using DashMap.
    /// - X-SMG-Routing-Key: Routes to a cached worker or assigns a new one
    /// - Provides true sticky sessions with zero key redistribution on worker add
    /// - Falls back to random selection if no routing key is provided
    /// - Supports LRU eviction when cache size exceeds max_entries
    #[serde(rename = "manual")]
    Manual {
        /// Interval between TTL eviction cycles (seconds, default: 60)
        #[serde(default = "default_manual_eviction_interval_secs")]
        eviction_interval_secs: u64,
        /// Maximum idle time before eviction (seconds, default: 14400 = 4 hours)
        #[serde(default = "default_manual_max_idle_secs")]
        max_idle_secs: u64,
        /// Assignment mode for new routing keys (default: random)
        #[serde(default)]
        assignment_mode: ManualAssignmentMode,
    },

    /// Consistent hashing policy using hash ring for session affinity:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by URL
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    /// - Provides O(log n) lookup with minimal redistribution (~1/N keys) on topology change
    #[serde(rename = "consistent_hashing")]
    ConsistentHashing,

    /// Prefix hash policy for KV cache-aware load balancing.
    /// A lightweight alternative to cache_aware radix tree.
    /// Routes requests based on prefix token hash for cache locality.
    /// - Uses consistent hash ring with bounded load balancing
    /// - Walks ring if worker is overloaded (load > avg * load_factor)
    /// - O(log n) lookup instead of O(prefix_len) radix tree traversal
    #[serde(rename = "prefix_hash")]
    PrefixHash {
        /// Number of prefix tokens to hash (default: 256)
        #[serde(default = "default_prefix_token_count")]
        prefix_token_count: usize,
        /// Load factor threshold - walk ring if load > avg * factor (default: 1.25)
        #[serde(default = "default_load_factor")]
        load_factor: f64,
    },
}

fn default_prefix_token_count() -> usize {
    256
}

fn default_load_factor() -> f64 {
    1.25
}

fn default_manual_eviction_interval_secs() -> u64 {
    60
}

fn default_manual_max_idle_secs() -> u64 {
    4 * 3600
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
            PolicyConfig::Bucket { .. } => "bucket",
            PolicyConfig::Manual { .. } => "manual",
            PolicyConfig::ConsistentHashing => "consistent_hashing",
            PolicyConfig::PrefixHash { .. } => "prefix_hash",
        }
    }
}
