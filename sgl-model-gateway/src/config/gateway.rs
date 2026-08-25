use std::{collections::HashMap, net::SocketAddr};

use clap::ValueEnum;
pub use data_connector::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig};
use serde::{Deserialize, Serialize};

use super::validation::ConfigValidator;
use crate::core::ConnectionMode;

pub const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 50;
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 500;
pub const DEFAULT_TCP_KEEPALIVE_SECS: u64 = 30;

/// Complete, resolved configuration for one gateway process.
///
/// Each field maps to a component started by the gateway. The hierarchy stops
/// here so consumers can depend on only the configuration they need.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub server: HttpServerConfig,
    pub routing: RoutingConfig,
    pub workers: WorkerConfig,
    pub model: ModelConfig,
    pub storage: StorageConfig,
    pub extensions: ExtensionConfig,
    pub observability: ObservabilityConfig,
    pub security: SecurityConfig,
    /// Kubernetes discovery settings; absent when discovery is disabled.
    pub discovery: Option<DiscoveryConfig>,
    /// Mesh node settings; absent when mesh is disabled.
    pub mesh: Option<MeshConfig>,
}

impl GatewayConfig {
    /// Creates a gateway configuration with the legacy router defaults.
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            routing: RoutingConfig::new(mode, policy),
            ..Default::default()
        }
    }

    pub fn validate(&self) -> super::ConfigResult<()> {
        ConfigValidator::validate_gateway(self)
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            server: HttpServerConfig::default(),
            routing: RoutingConfig::default(),
            workers: WorkerConfig::default(),
            model: ModelConfig::default(),
            storage: StorageConfig::default(),
            extensions: ExtensionConfig::default(),
            observability: ObservabilityConfig::default(),
            security: SecurityConfig::default(),
            discovery: None,
            mesh: None,
        }
    }
}

/// Configuration for the gateway's public HTTP listener.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    pub max_payload_size: usize,
    pub cors_allowed_origins: Vec<String>,
    pub request_id_headers: Option<Vec<String>>,
    pub shutdown_grace_period_secs: u64,
}

impl Default for HttpServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3001,
            max_payload_size: 536_870_912,
            cors_allowed_origins: Vec::new(),
            request_id_headers: None,
            shutdown_grace_period_secs: 180,
        }
    }
}

/// Security settings used by the public server and its outbound clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication for inbound gateway requests.
    pub api_key: Option<String>,
    pub control_plane_auth: Option<crate::auth::ControlPlaneAuthConfig>,

    /// TLS identity presented by the public HTTP server.
    pub server_tls: Option<ServerTlsConfig>,

    /// Client certificate material used for outbound connections.
    #[serde(skip)]
    pub client_identity: Option<Vec<u8>>,
    #[serde(default)]
    pub ca_certificates: Vec<Vec<u8>>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            control_plane_auth: None,
            server_tls: None,
            client_identity: None,
            ca_certificates: Vec::new(),
        }
    }
}

/// Loaded PEM material for the public server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTlsConfig {
    #[serde(skip)]
    pub certificate: Vec<u8>,
    #[serde(skip)]
    pub private_key: Vec<u8>,
}

/// Configuration for request admission and worker selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Configuration for connections to and lifecycle management of workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Protocol used for gateway-to-worker calls.
    pub connection_mode: ConnectionMode,
    /// Deadline for one request sent to a worker.
    pub request_timeout_secs: u64,
    /// Total time allowed for configured workers to become available at startup.
    pub startup_timeout_secs: u64,
    /// Polling interval while waiting for worker startup.
    pub startup_check_interval_secs: u64,
    /// How long an unused HTTP connection remains in the pool.
    pub pool_idle_timeout_secs: u64,
    /// Deadline for opening a new HTTP connection to a worker.
    pub connect_timeout_secs: u64,
    /// Maximum idle HTTP connections retained per worker host.
    pub pool_max_idle_per_host: usize,
    /// Idle time before TCP keepalive probes begin.
    pub tcp_keepalive_secs: u64,
    /// Retry policy for failed worker requests.
    pub retry: RetryConfig,
    /// Policy for temporarily excluding repeatedly failing workers.
    pub circuit_breaker: CircuitBreakerConfig,
    /// Forces each failed request to use only its initial attempt.
    pub disable_retries: bool,
    /// Prevents workers from being temporarily excluded after failures.
    pub disable_circuit_breaker: bool,
    /// Background probes used to determine worker availability.
    pub health_check: HealthCheckConfig,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            connection_mode: ConnectionMode::Http,
            request_timeout_secs: 1800,
            startup_timeout_secs: 1800,
            startup_check_interval_secs: 30,
            pool_idle_timeout_secs: default_pool_idle_timeout_secs(),
            connect_timeout_secs: default_connect_timeout_secs(),
            pool_max_idle_per_host: default_pool_max_idle_per_host(),
            tcp_keepalive_secs: default_tcp_keepalive_secs(),
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
        }
    }
}

impl WorkerConfig {
    pub fn effective_retry_config(&self) -> RetryConfig {
        let mut config = self.retry.clone();
        if self.disable_retries {
            config.max_retries = 1;
        }
        config
    }

    pub fn effective_circuit_breaker_config(&self) -> CircuitBreakerConfig {
        let mut config = self.circuit_breaker.clone();
        if self.disable_circuit_breaker {
            config.failure_threshold = u32::MAX;
        }
        config
    }
}

fn default_pool_idle_timeout_secs() -> u64 {
    DEFAULT_POOL_IDLE_TIMEOUT_SECS
}

fn default_connect_timeout_secs() -> u64 {
    DEFAULT_CONNECT_TIMEOUT_SECS
}

fn default_pool_max_idle_per_host() -> usize {
    DEFAULT_POOL_MAX_IDLE_PER_HOST
}

fn default_tcp_keepalive_secs() -> u64 {
    DEFAULT_TCP_KEEPALIVE_SECS
}

/// Retry configuration for request handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f32,
    /// D' = D * (1 + U[-j, +j]) where j is jitter factor
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// Circuit breaker configuration for worker reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_secs: u64,
    /// Failure-counting window reserved for circuit-breaker accounting.
    pub window_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 3,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        }
    }
}

/// Health check configuration for worker monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_secs: u64,
    pub check_interval_secs: u64,
    pub endpoint: String,
    pub disable_health_check: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 5,
            check_interval_secs: 60,
            endpoint: "/health".to_string(),
            disable_health_check: false,
        }
    }
}

/// Configuration for the model and its request/output adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Configuration for conversation and request history storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub history_backend: HistoryBackend,
    pub oracle: Option<OracleConfig>,
    pub postgres: Option<PostgresConfig>,
    pub redis: Option<RedisConfig>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            history_backend: HistoryBackend::Memory,
            oracle: None,
            postgres: None,
            redis: None,
        }
    }
}

/// Configuration for optional gateway extensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionConfig {
    #[serde(skip)]
    pub mcp_config: Option<smg_mcp::McpConfig>,
    #[serde(default)]
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

/// Logging, metrics, and tracing settings for the gateway process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub json_log: bool,
    pub prometheus_host: String,
    pub prometheus_port: u16,
    pub prometheus_duration_buckets: Option<Vec<f64>>,
    pub enable_trace: bool,
    pub otlp_traces_endpoint: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_dir: None,
            log_level: None,
            json_log: false,
            prometheus_port: 29000,
            prometheus_host: "0.0.0.0".to_string(),
            prometheus_duration_buckets: None,
            enable_trace: false,
            otlp_traces_endpoint: "localhost:4317".to_string(),
        }
    }
}

/// Kubernetes-based worker and router discovery settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub selector: HashMap<String, String>,
    pub namespace: Option<String>,
    pub port: u16,
    pub check_interval_secs: u64,
    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    pub bootstrap_port_annotation: String,
    pub router_selector: HashMap<String, String>,
    pub router_mesh_port_annotation: String,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            selector: HashMap::new(),
            namespace: None,
            port: 8000,
            check_interval_secs: 60,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
        }
    }
}

/// Configuration for this gateway's mesh node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    pub self_name: String,
    pub self_addr: SocketAddr,
    pub init_peer: Option<SocketAddr>,
}
