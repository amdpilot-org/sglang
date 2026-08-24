use std::collections::HashMap;

use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use rand::{distr::Alphanumeric, Rng};

use crate::{
    auth::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role},
    config::{
        extensions::ExtensionConfig,
        gateway::GatewayConfig,
        infrastructure::{
            DiscoveryConfig as GatewayDiscoveryConfig, MeshConfig, ObservabilityConfig,
        },
        model::{ModelConfig, TokenizerCacheConfig},
        routing::{ManualAssignmentMode, PolicyConfig, RoutingConfig, RoutingMode},
        server::{HttpServerConfig, SecurityConfig, ServerTlsConfig},
        storage::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig, StorageConfig},
        worker_pool::{
            CircuitBreakerConfig, HealthCheckConfig, RetryConfig, WorkerPoolConfig,
            DEFAULT_CONNECT_TIMEOUT_SECS, DEFAULT_POOL_IDLE_TIMEOUT_SECS,
            DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_TCP_KEEPALIVE_SECS,
        },
        ConfigError, ConfigResult,
    },
    core::ConnectionMode,
    version,
};

#[derive(Parser, Debug)]
#[command(name = "sglang-router")]
#[command(about = "SGLang Model Gateway - High-performance inference gateway")]
#[command(args_conflicts_with_subcommands = true)]
#[command(version = version::get_version_string())]
#[command(long_about = "SGLang Model Gateway - Rust-based inference gateway")]
#[command(after_long_help = r#"
Examples:
  # Regular mode
  smg --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode
  smg --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --policy cache_aware

  # With different policies
  smg --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --prefill-policy cache_aware --decode-policy power_of_two

"#)]
pub struct Cli {
    #[arg(
        long,
        global = true,
        exclusive = true,
        help = "Print detailed version information and exit"
    )]
    version_verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    args: CliArgs,
}

impl Cli {
    /// Selects the root or `launch` arguments and resolves them into startup config.
    pub fn try_into_config(self) -> Result<GatewayConfig> {
        let args = match self.command {
            Some(Commands::Launch { args }) => args,
            None => self.args,
        };

        args.resolve_config()
    }

    /// Returns whether the detailed version output was requested.
    pub fn wants_verbose_version(&self) -> bool {
        self.version_verbose
    }
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Launch the router (same as running without subcommand)
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: CliArgs,
    },
}

#[derive(Parser, Debug)]
pub struct CliArgs {
    // Server Configuration
    /// Host address to bind the router server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Server Configuration")]
    host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000, help_heading = "Server Configuration")]
    port: u16,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 536870912, help_heading = "Request Handling")]
    max_payload_size: usize,

    /// CORS allowed origins
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    cors_allowed_origins: Vec<String>,

    /// Custom HTTP headers to check for request IDs
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    request_id_headers: Vec<String>,

    /// Grace period in seconds to wait for in-flight requests during shutdown
    #[arg(long, default_value_t = 180, help_heading = "Request Handling")]
    shutdown_grace_period_secs: u64,

    // Worker Configuration
    /// List of worker URLs (supports IPv4 and IPv6)
    #[arg(long, num_args = 0.., help_heading = "Worker Configuration")]
    worker_urls: Vec<String>,

    /// API key used to authorize requests to workers
    #[arg(long, help_heading = "Worker Configuration")]
    api_key: Option<String>,

    /// Maximum time to wait for a worker to become reachable and register
    #[arg(long, default_value_t = 1800, help_heading = "Worker Configuration")]
    worker_startup_timeout_secs: u64,

    /// Interval in seconds between worker load and availability checks
    #[arg(long, default_value_t = 30, help_heading = "Worker Configuration")]
    worker_startup_check_interval: u64,

    // Routing
    /// Load balancing policy to use
    #[arg(long, value_enum, default_value_t = PolicyKind::CacheAware, help_heading = "Routing Policy")]
    policy: PolicyKind,

    /// Cache threshold (0.0-1.0) for cache-aware routing
    #[arg(long, default_value_t = 0.3, help_heading = "Routing Policy")]
    cache_threshold: f32,

    /// Absolute threshold for load balancing trigger
    #[arg(long, default_value_t = 64, help_heading = "Routing Policy")]
    balance_abs_threshold: usize,

    /// Relative threshold for load balancing trigger
    #[arg(long, default_value_t = 1.5, help_heading = "Routing Policy")]
    balance_rel_threshold: f32,

    /// Interval in seconds between cache eviction operations
    #[arg(long, default_value_t = 120, help_heading = "Routing Policy")]
    eviction_interval: u64,

    /// Maximum size of the approximation tree for cache-aware routing
    #[arg(long, default_value_t = 67108864, help_heading = "Routing Policy")]
    max_tree_size: usize,

    /// Maximum idle time in seconds before eviction (for manual policy)
    #[arg(long, default_value_t = 14400, help_heading = "Routing Policy")]
    max_idle_secs: u64,

    /// Assignment mode for manual policy when encountering a new routing key
    #[arg(long, value_enum, default_value_t = ManualAssignmentMode::Random, help_heading = "Routing Policy")]
    assignment_mode: ManualAssignmentMode,

    /// Number of prefix tokens to use for prefix_hash policy
    #[arg(long, default_value_t = 256, help_heading = "Routing Policy")]
    prefix_token_count: usize,

    /// Load factor threshold for prefix_hash policy
    #[arg(long, default_value_t = 1.25, help_heading = "Routing Policy")]
    prefix_hash_load_factor: f64,

    /// Enable data parallelism aware scheduling
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    dp_aware: bool,

    /// Enable IGW (Inference Gateway) mode for multi-model support
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    enable_igw: bool,

    // PD Disaggregation
    /// Enable PD (Prefill-Decode) disaggregated mode
    #[arg(long, default_value_t = false, help_heading = "PD Disaggregation")]
    pd_disaggregation: bool,

    /// Prefill server URL and optional bootstrap port; may be specified multiple times
    #[arg(
        long,
        value_names = ["URL", "BOOTSTRAP_PORT"],
        num_args = 1..=2,
        action = ArgAction::Append,
        help_heading = "PD Disaggregation"
    )]
    prefill: Vec<Vec<String>>,

    /// Decode server URLs; may be specified multiple times
    #[arg(long, action = ArgAction::Append, help_heading = "PD Disaggregation")]
    decode: Vec<String>,

    /// Specific policy for prefill nodes in PD mode
    #[arg(long, value_enum, help_heading = "PD Disaggregation")]
    prefill_policy: Option<PolicyKind>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_enum, help_heading = "PD Disaggregation")]
    decode_policy: Option<PolicyKind>,

    // Admission control
    /// Maximum concurrent requests (-1 to disable)
    #[arg(long, default_value_t = -1, help_heading = "Rate Limiting")]
    max_concurrent_requests: i32,

    /// Queue size for pending requests when limit reached
    #[arg(long, default_value_t = 100, help_heading = "Rate Limiting")]
    queue_size: usize,

    /// Maximum time in seconds a request can wait in queue
    #[arg(long, default_value_t = 60, help_heading = "Rate Limiting")]
    queue_timeout_secs: u64,

    /// Token bucket refill rate (tokens per second)
    #[arg(long, help_heading = "Rate Limiting")]
    rate_limit_tokens_per_second: Option<i32>,

    // Backend
    /// Backend runtime to use
    #[arg(
        long,
        value_enum,
        default_value_t = Backend::Sglang,
        alias = "runtime",
        help_heading = "Backend"
    )]
    backend: Backend,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 1800, help_heading = "HTTP Client")]
    request_timeout_secs: u64,

    /// Idle timeout in seconds for pooled upstream HTTP connections
    #[arg(
        long,
        env = "SMG_POOL_IDLE_TIMEOUT_SECS",
        default_value_t = DEFAULT_POOL_IDLE_TIMEOUT_SECS,
        help_heading = "HTTP Client"
    )]
    pool_idle_timeout_secs: u64,

    /// Timeout in seconds for new upstream HTTP connections
    #[arg(
        long,
        env = "SMG_CONNECT_TIMEOUT_SECS",
        default_value_t = DEFAULT_CONNECT_TIMEOUT_SECS,
        help_heading = "HTTP Client"
    )]
    connect_timeout_secs: u64,

    /// Maximum idle upstream HTTP connections to keep per host
    #[arg(
        long,
        env = "SMG_POOL_MAX_IDLE_PER_HOST",
        default_value_t = DEFAULT_POOL_MAX_IDLE_PER_HOST,
        help_heading = "HTTP Client"
    )]
    pool_max_idle_per_host: usize,

    /// TCP keepalive idle time in seconds for upstream HTTP connections
    #[arg(
        long,
        env = "SMG_TCP_KEEPALIVE_SECS",
        default_value_t = DEFAULT_TCP_KEEPALIVE_SECS,
        help_heading = "HTTP Client"
    )]
    tcp_keepalive_secs: u64,

    // Retry Configuration
    /// Maximum number of retry attempts
    #[arg(long, default_value_t = 5, help_heading = "Retry Configuration")]
    retry_max_retries: u32,

    /// Initial backoff delay in milliseconds
    #[arg(long, default_value_t = 50, help_heading = "Retry Configuration")]
    retry_initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds
    #[arg(long, default_value_t = 30000, help_heading = "Retry Configuration")]
    retry_max_backoff_ms: u64,

    /// Multiplier for exponential backoff
    #[arg(long, default_value_t = 1.5, help_heading = "Retry Configuration")]
    retry_backoff_multiplier: f32,

    /// Jitter factor (0.0-1.0) for retry delays
    #[arg(long, default_value_t = 0.2, help_heading = "Retry Configuration")]
    retry_jitter_factor: f32,

    /// Disable automatic retries
    #[arg(long, default_value_t = false, help_heading = "Retry Configuration")]
    disable_retries: bool,

    // Circuit breaker
    /// Number of failures before circuit opens
    #[arg(long, default_value_t = 10, help_heading = "Circuit Breaker")]
    cb_failure_threshold: u32,

    /// Successes needed in half-open state to close
    #[arg(long, default_value_t = 3, help_heading = "Circuit Breaker")]
    cb_success_threshold: u32,

    /// Seconds before attempting to close open circuit
    #[arg(long, default_value_t = 60, help_heading = "Circuit Breaker")]
    cb_timeout_duration_secs: u64,

    /// Sliding window duration for tracking failures
    #[arg(long, default_value_t = 120, help_heading = "Circuit Breaker")]
    cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false, help_heading = "Circuit Breaker")]
    disable_circuit_breaker: bool,

    // Health checks
    /// Failures before marking worker unhealthy
    #[arg(long, default_value_t = 3, help_heading = "Health Checks")]
    health_failure_threshold: u32,

    /// Successes before marking worker healthy
    #[arg(long, default_value_t = 2, help_heading = "Health Checks")]
    health_success_threshold: u32,

    /// Timeout in seconds for health check requests
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    health_check_timeout_secs: u64,

    /// Interval in seconds between health checks
    #[arg(long, default_value_t = 60, help_heading = "Health Checks")]
    health_check_interval_secs: u64,

    /// Health check endpoint path
    #[arg(long, default_value = "/health", help_heading = "Health Checks")]
    health_check_endpoint: String,

    /// Disable all worker health checks
    #[arg(long, default_value_t = false, help_heading = "Health Checks")]
    disable_health_check: bool,

    // Tokenizer
    /// Model path for loading tokenizer (HuggingFace ID or local path)
    #[arg(long, help_heading = "Tokenizer")]
    model_path: Option<String>,

    /// Explicit tokenizer path (overrides model_path)
    #[arg(long, help_heading = "Tokenizer")]
    tokenizer_path: Option<String>,

    /// Chat template path
    #[arg(long, help_heading = "Tokenizer")]
    chat_template: Option<String>,

    /// Enable L0 (exact match) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l0: bool,

    /// Maximum entries in L0 tokenizer cache
    #[arg(long, default_value_t = 10000, help_heading = "Tokenizer")]
    tokenizer_cache_l0_max_entries: usize,

    /// Enable L1 (prefix matching) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l1: bool,

    /// Maximum memory for L1 tokenizer cache in bytes
    #[arg(long, default_value_t = 52428800, help_heading = "Tokenizer")]
    tokenizer_cache_l1_max_memory: usize,

    // Model parsers
    /// Parser for reasoning models (e.g., deepseek-r1, qwen3)
    #[arg(long, help_heading = "Parsers")]
    reasoning_parser: Option<String>,

    /// Parser for tool-call interactions
    #[arg(long, help_heading = "Parsers")]
    tool_call_parser: Option<String>,

    // Extensions
    /// Path to MCP server configuration file
    #[arg(long, help_heading = "Extensions")]
    mcp_config_path: Option<String>,

    /// Enable WebAssembly support
    #[arg(long, default_value_t = false, help_heading = "Extensions")]
    enable_wasm: bool,

    // Storage
    /// History storage backend
    #[arg(long,value_enum, default_value_t = HistoryBackendKind::Memory, help_heading = "Storage")]
    history_backend: HistoryBackendKind,

    // Oracle storage
    /// Path to Oracle ATP wallet directory
    #[arg(long, env = "ATP_WALLET_PATH", help_heading = "Oracle Database")]
    oracle_wallet_path: Option<String>,

    /// Oracle TNS alias from tnsnames.ora
    #[arg(long, env = "ATP_TNS_ALIAS", help_heading = "Oracle Database")]
    oracle_tns_alias: Option<String>,

    /// Oracle connection descriptor/DSN
    #[arg(long, env = "ATP_DSN", help_heading = "Oracle Database")]
    oracle_dsn: Option<String>,

    /// Oracle database username
    #[arg(long, env = "ATP_USER", help_heading = "Oracle Database")]
    oracle_user: Option<String>,

    /// Oracle database password
    #[arg(long, env = "ATP_PASSWORD", help_heading = "Oracle Database")]
    oracle_password: Option<String>,

    /// Minimum Oracle connection pool size
    #[arg(long, env = "ATP_POOL_MIN", help_heading = "Oracle Database")]
    oracle_pool_min: Option<usize>,

    /// Maximum Oracle connection pool size
    #[arg(long, env = "ATP_POOL_MAX", help_heading = "Oracle Database")]
    oracle_pool_max: Option<usize>,

    /// Oracle connection pool timeout in seconds
    #[arg(long, env = "ATP_POOL_TIMEOUT_SECS", help_heading = "Oracle Database")]
    oracle_pool_timeout_secs: Option<u64>,

    // PostgreSQL storage
    /// PostgreSQL database connection URL
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_db_url: Option<String>,

    /// Maximum PostgreSQL connection pool size
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_pool_max_size: Option<usize>,

    // Redis storage
    /// Redis connection URL
    #[arg(long, help_heading = "Redis Database")]
    redis_url: Option<String>,

    /// Maximum Redis connection pool size
    #[arg(long, help_heading = "Redis Database")]
    redis_pool_max_size: Option<usize>,

    /// Redis data retention in days (-1 for persistent, default 30)
    #[arg(long, help_heading = "Redis Database")]
    redis_retention_days: Option<i64>,

    // Security
    /// Path to server TLS certificate (PEM format)
    #[arg(long, help_heading = "TLS/mTLS Security")]
    tls_cert_path: Option<String>,

    /// Path to server TLS private key (PEM format)
    #[arg(long, help_heading = "TLS/mTLS Security")]
    tls_key_path: Option<String>,

    // Control-plane authentication
    /// JWT issuer URL for OIDC authentication
    #[arg(
        long,
        env = "JWT_ISSUER",
        help_heading = "Control Plane Authentication"
    )]
    jwt_issuer: Option<String>,

    /// Expected JWT audience claim
    #[arg(
        long,
        env = "JWT_AUDIENCE",
        help_heading = "Control Plane Authentication"
    )]
    jwt_audience: Option<String>,

    /// Explicit JWKS URI (discovered from issuer if not set)
    #[arg(
        long,
        env = "JWT_JWKS_URI",
        help_heading = "Control Plane Authentication"
    )]
    jwt_jwks_uri: Option<String>,

    /// JWT claim name containing the role
    #[arg(
        long,
        default_value = "roles",
        help_heading = "Control Plane Authentication"
    )]
    jwt_role_claim: String,

    /// Role mapping from IDP to gateway role (format: idp_role=gateway_role)
    #[arg(long, action = ArgAction::Append, help_heading = "Control Plane Authentication")]
    jwt_role_mapping: Vec<String>,

    /// API keys for control plane access (format: id:name:role:key)
    #[arg(long = "control-plane-api-keys", action = ArgAction::Append, env = "CONTROL_PLANE_API_KEYS", help_heading = "Control Plane Authentication")]
    control_plane_api_keys: Vec<String>,

    /// Disable audit logging for control plane operations
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Control Plane Authentication"
    )]
    disable_audit_logging: bool,

    // Discovery
    /// Enable Kubernetes service discovery
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery: bool,

    /// Label selector for Kubernetes service discovery (format: key=value)
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    selector: Vec<String>,

    /// Port to use for discovered worker pods
    #[arg(
        long,
        default_value_t = 80,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery_port: u16,

    /// Kubernetes namespace to watch for pods
    #[arg(long, help_heading = "Service Discovery (Kubernetes)")]
    service_discovery_namespace: Option<String>,

    /// Label selector for prefill server pods in PD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    prefill_selector: Vec<String>,

    /// Label selector for decode server pods in PD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    decode_selector: Vec<String>,

    // Mesh
    #[arg(long, default_value_t = false, help_heading = "Mesh")]
    enable_mesh: bool,

    #[arg(long, help_heading = "Mesh")]
    mesh_server_name: Option<String>,

    #[arg(long, default_value = "0.0.0.0", help_heading = "Mesh")]
    mesh_host: String,

    #[arg(long, default_value_t = 39527, help_heading = "Mesh")]
    mesh_port: u16,

    #[arg(long, num_args = 0.., help_heading = "Mesh")]
    mesh_peer_urls: Vec<String>,

    // Observability
    /// Directory to store log files
    #[arg(long, help_heading = "Logging")]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// Enable structured JSON log output instead of plain text
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    json_log: bool,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Prometheus Metrics")]
    prometheus_host: String,

    /// Port to expose the Prometheus metrics server
    #[arg(long, default_value_t = 29000, help_heading = "Prometheus Metrics")]
    prometheus_port: u16,

    /// Custom buckets for Prometheus duration metrics
    #[arg(long, num_args = 0.., help_heading = "Prometheus Metrics")]
    prometheus_duration_buckets: Vec<f64>,

    /// Enable OpenTelemetry tracing
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Tracing (OpenTelemetry)"
    )]
    enable_trace: bool,

    /// OTLP collector endpoint (format: host:port)
    #[arg(
        long,
        default_value = "localhost:4317",
        help_heading = "Tracing (OpenTelemetry)"
    )]
    otlp_traces_endpoint: String,
}

impl CliArgs {
    /// Parses repeated `--prefill URL [BOOTSTRAP_PORT]` arguments.
    fn parse_prefill_args(&self) -> ConfigResult<Vec<(String, Option<u16>)>> {
        self.prefill
            .iter()
            .map(|values| match values.as_slice() {
                [url] => Ok((url.clone(), None)),
                [url, port] if port.eq_ignore_ascii_case("none") => Ok((url.clone(), None)),
                [url, port] => {
                    let port = port.parse::<u16>().map_err(|_| ConfigError::InvalidValue {
                        field: "prefill bootstrap port".to_string(),
                        value: port.clone(),
                        reason: "must be a valid u16 port or `none`".to_string(),
                    })?;

                    Ok((url.clone(), Some(port)))
                }
                _ => unreachable!("Clap restricts --prefill to one or two values"),
            })
            .collect()
    }

    /// Build control plane authentication configuration from CLI args.
    fn build_control_plane_auth_config(&self) -> ControlPlaneAuthConfig {
        // Build JWT config if issuer and audience are provided
        let jwt = match (&self.jwt_issuer, &self.jwt_audience) {
            (Some(issuer), Some(audience)) => {
                let role_mapping: HashMap<String, Role> = self
                    .jwt_role_mapping
                    .iter()
                    .filter_map(|m| parse_role_mapping(m))
                    .collect();

                let mut jwt_config = JwtConfig::new(issuer.clone(), audience.clone());
                jwt_config.role_claim = self.jwt_role_claim.clone();
                jwt_config.role_mapping = role_mapping;
                if let Some(jwks_uri) = &self.jwt_jwks_uri {
                    jwt_config.jwks_uri = Some(jwks_uri.clone());
                }
                Some(jwt_config)
            }
            (Some(_), None) => {
                eprintln!("WARNING: --jwt-issuer provided but --jwt-audience is missing. JWT auth disabled.");
                None
            }
            (None, Some(_)) => {
                eprintln!("WARNING: --jwt-audience provided but --jwt-issuer is missing. JWT auth disabled.");
                None
            }
            (None, None) => None,
        };

        // Build API keys from CLI args
        let api_keys: Vec<ApiKeyEntry> = self
            .control_plane_api_keys
            .iter()
            .filter_map(|k| parse_control_plane_api_key(k))
            .collect();

        ControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled: !self.disable_audit_logging,
        }
    }

    /// Combines a selected policy with its CLI tuning parameters.
    fn build_policy_config(&self, policy: PolicyKind) -> PolicyConfig {
        match policy {
            PolicyKind::Random => PolicyConfig::Random,
            PolicyKind::RoundRobin => PolicyConfig::RoundRobin,
            PolicyKind::CacheAware => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            PolicyKind::PowerOfTwo => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
            PolicyKind::PrefixHash => PolicyConfig::PrefixHash {
                prefix_token_count: self.prefix_token_count,
                load_factor: self.prefix_hash_load_factor,
            },
            PolicyKind::Manual => PolicyConfig::Manual {
                eviction_interval_secs: self.eviction_interval,
                max_idle_secs: self.max_idle_secs,
                assignment_mode: self.assignment_mode,
            },
            // TODO: impl bucket policy
            _ => PolicyConfig::RoundRobin,
        }
    }

    /// Builds Oracle settings from either a DSN or wallet/TNS inputs.
    fn build_oracle_config(&self) -> ConfigResult<OracleConfig> {
        let (wallet_path, connect_descriptor) = match &self.oracle_dsn {
            Some(dsn) => (None, dsn.clone()),
            None => {
                let wallet_path =
                    self.oracle_wallet_path
                        .clone()
                        .ok_or(ConfigError::MissingRequired {
                            field: "oracle_wallet_path or ATP_WALLET_PATH".to_string(),
                        })?;
                let tns_alias =
                    self.oracle_tns_alias
                        .clone()
                        .ok_or(ConfigError::MissingRequired {
                            field: "oracle_tns_alias or ATP_TNS_ALIAS".to_string(),
                        })?;
                (Some(wallet_path), tns_alias)
            }
        };
        let username = self
            .oracle_user
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_user or ATP_USER".to_string(),
            })?;
        let password = self
            .oracle_password
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_password or ATP_PASSWORD".to_string(),
            })?;

        let pool_min = self
            .oracle_pool_min
            .unwrap_or_else(OracleConfig::default_pool_min);
        let pool_max = self
            .oracle_pool_max
            .unwrap_or_else(OracleConfig::default_pool_max);

        if pool_min == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_min".to_string(),
                value: pool_min.to_string(),
                reason: "pool minimum must be at least 1".to_string(),
            });
        }

        if pool_max < pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_max".to_string(),
                value: pool_max.to_string(),
                reason: "pool maximum must be greater than or equal to minimum".to_string(),
            });
        }

        let pool_timeout_secs = self
            .oracle_pool_timeout_secs
            .unwrap_or_else(OracleConfig::default_pool_timeout_secs);

        Ok(OracleConfig {
            wallet_path,
            connect_descriptor,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
        })
    }

    /// Builds and validates PostgreSQL storage settings.
    fn build_postgres_config(&self) -> ConfigResult<PostgresConfig> {
        let db_url = self.postgres_db_url.clone().unwrap_or_default();
        let pool_max = self
            .postgres_pool_max_size
            .unwrap_or_else(PostgresConfig::default_pool_max);

        let pcf = PostgresConfig { db_url, pool_max };
        pcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;

        Ok(pcf)
    }

    /// Builds and validates Redis storage settings.
    fn build_redis_config(&self) -> ConfigResult<RedisConfig> {
        let url = self.redis_url.clone().unwrap_or_default();
        let pool_max = self.redis_pool_max_size.unwrap_or(16);

        let retention_days = match self.redis_retention_days {
            Some(d) if d < 0 => None, // Persistent
            Some(d) => Some(d as u64),
            None => Some(30), // Default 30 days
        };

        let rcf = RedisConfig {
            url,
            pool_max,
            retention_days,
        };
        rcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(rcf)
    }

    /// Loads TLS material when both certificate and key paths are supplied.
    fn load_server_tls_config(&self) -> ConfigResult<Option<ServerTlsConfig>> {
        match (&self.tls_cert_path, &self.tls_key_path) {
            (Some(cert_path), Some(key_path)) => {
                let certificate =
                    std::fs::read(cert_path).map_err(|error| ConfigError::ValidationFailed {
                        reason: format!(
                            "Failed to read server certificate from {cert_path}: {error}"
                        ),
                    })?;
                let private_key =
                    std::fs::read(key_path).map_err(|error| ConfigError::ValidationFailed {
                        reason: format!("Failed to read server key from {key_path}: {error}"),
                    })?;
                Ok(Some(ServerTlsConfig {
                    certificate,
                    private_key,
                }))
            }
            (None, None) => Ok(None),
            _ => Err(ConfigError::ValidationFailed {
                reason: "Both --tls-cert-path and --tls-key-path must be specified together"
                    .to_string(),
            }),
        }
    }

    /// Loads and parses the optional MCP YAML configuration.
    fn load_mcp_config(&self) -> ConfigResult<Option<smg_mcp::McpConfig>> {
        let Some(path) = &self.mcp_config_path else {
            return Ok(None);
        };

        let contents =
            std::fs::read_to_string(path).map_err(|error| ConfigError::ValidationFailed {
                reason: format!("Failed to read MCP config from {path}: {error}"),
            })?;
        let config =
            serde_yaml::from_str(&contents).map_err(|error| ConfigError::ValidationFailed {
                reason: format!("Failed to parse MCP config from {path}: {error}"),
            })?;
        Ok(Some(config))
    }

    /// Builds mesh node settings, returning None when mesh is disabled or invalid.
    fn build_mesh_config(&self) -> ConfigResult<Option<MeshConfig>> {
        if !self.enable_mesh {
            return Ok(None);
        }

        let self_name = self.mesh_server_name.clone().unwrap_or_else(|| {
            let mut rng = rand::rng();
            let suffix: String = (0..4).map(|_| rng.sample(Alphanumeric) as char).collect();
            format!("Mesh_{suffix}")
        });

        let address = format!("{}:{}", self.mesh_host, self.mesh_port);
        let self_addr = address.parse().map_err(|_| ConfigError::InvalidValue {
            field: "mesh perr".to_string(),
            value: address.clone(),
            reason: "must be a valid socket address".to_string(),
        })?;

        let init_peer = match self.mesh_peer_urls.first() {
            Some(peer) => Some(peer.parse().map_err(|_| ConfigError::InvalidValue {
                field: "mesh peer".to_string(),
                value: peer.clone(),
                reason: "must be a valid socket address".to_string(),
            })?),
            None => None,
        };

        Ok(Some(MeshConfig {
            self_name,
            self_addr,
            init_peer,
        }))
    }

    /// Resolves CLI arguments into the canonical gateway configuration.
    ///
    /// File-backed settings are loaded here so the resulting configuration is
    /// self-contained and does not retain CLI paths.
    fn resolve_config(&self) -> Result<GatewayConfig> {
        let prefill_urls = self.parse_prefill_args()?;
        let enable_igw = self.enable_igw || self.service_discovery;

        let mode = if matches!(self.backend, Backend::Openai) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls: self.decode.clone(),
                prefill_policy: self.prefill_policy.map(|p| self.build_policy_config(p)),
                decode_policy: self.decode_policy.map(|p| self.build_policy_config(p)),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let mut all_urls = Vec::new();
        match &mode {
            RoutingMode::Regular { worker_urls } => all_urls.extend(worker_urls.clone()),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                all_urls.extend(prefill_urls.iter().map(|(url, _)| url.clone()));
                all_urls.extend(decode_urls.clone());
            }
            RoutingMode::OpenAI { .. } => {}
        }
        let connection_mode = match &mode {
            RoutingMode::OpenAI { .. } => ConnectionMode::Http,
            _ => determine_connection_mode(&all_urls),
        };

        let oracle = (self.history_backend == HistoryBackendKind::Oracle)
            .then(|| self.build_oracle_config())
            .transpose()?;
        let postgres = (self.history_backend == HistoryBackendKind::Postgres)
            .then(|| self.build_postgres_config())
            .transpose()?;
        let redis = (self.history_backend == HistoryBackendKind::Redis)
            .then(|| self.build_redis_config())
            .transpose()?;

        let discovery = self.service_discovery.then(|| GatewayDiscoveryConfig {
            selector: parse_label_selectors(&self.selector),
            namespace: self.service_discovery_namespace.clone(),
            port: self.service_discovery_port,
            check_interval_secs: 60,
            pd_mode: self.pd_disaggregation,
            prefill_selector: parse_label_selectors(&self.prefill_selector),
            decode_selector: parse_label_selectors(&self.decode_selector),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
            igw_mode: enable_igw,
        });

        let control_plane_auth = self.build_control_plane_auth_config();
        let control_plane_auth = control_plane_auth
            .is_enabled()
            .then_some(control_plane_auth);

        let config = GatewayConfig {
            server: HttpServerConfig {
                host: self.host.clone(),
                port: self.port,
                max_payload_size: self.max_payload_size,
                cors_allowed_origins: self.cors_allowed_origins.clone(),
                request_id_headers: (!self.request_id_headers.is_empty())
                    .then(|| self.request_id_headers.clone()),
                shutdown_grace_period_secs: self.shutdown_grace_period_secs,
            },
            routing: RoutingConfig {
                mode,
                policy: self.build_policy_config(self.policy),
                dp_aware: self.dp_aware,
                enable_igw,
                max_concurrent_requests: self.max_concurrent_requests,
                queue_size: self.queue_size,
                queue_timeout_secs: self.queue_timeout_secs,
                rate_limit_tokens_per_second: self.rate_limit_tokens_per_second,
            },
            workers: WorkerPoolConfig {
                connection_mode,
                request_timeout_secs: self.request_timeout_secs,
                startup_timeout_secs: self.worker_startup_timeout_secs,
                startup_check_interval_secs: self.worker_startup_check_interval,
                pool_idle_timeout_secs: self.pool_idle_timeout_secs,
                connect_timeout_secs: self.connect_timeout_secs,
                pool_max_idle_per_host: self.pool_max_idle_per_host,
                tcp_keepalive_secs: self.tcp_keepalive_secs,
                retry: RetryConfig {
                    max_retries: self.retry_max_retries,
                    initial_backoff_ms: self.retry_initial_backoff_ms,
                    max_backoff_ms: self.retry_max_backoff_ms,
                    backoff_multiplier: self.retry_backoff_multiplier,
                    jitter_factor: self.retry_jitter_factor,
                },
                circuit_breaker: CircuitBreakerConfig {
                    failure_threshold: self.cb_failure_threshold,
                    success_threshold: self.cb_success_threshold,
                    timeout_duration_secs: self.cb_timeout_duration_secs,
                    window_duration_secs: self.cb_window_duration_secs,
                },
                disable_retries: self.disable_retries,
                disable_circuit_breaker: self.disable_circuit_breaker,
                health_check: HealthCheckConfig {
                    failure_threshold: self.health_failure_threshold,
                    success_threshold: self.health_success_threshold,
                    timeout_secs: self.health_check_timeout_secs,
                    check_interval_secs: self.health_check_interval_secs,
                    endpoint: self.health_check_endpoint.clone(),
                    disable_health_check: self.disable_health_check,
                },
            },
            model: ModelConfig {
                model_path: self.model_path.clone(),
                tokenizer_path: self.tokenizer_path.clone(),
                chat_template: self.chat_template.clone(),
                tokenizer_cache: TokenizerCacheConfig {
                    enable_l0: self.tokenizer_cache_enable_l0,
                    l0_max_entries: self.tokenizer_cache_l0_max_entries,
                    enable_l1: self.tokenizer_cache_enable_l1,
                    l1_max_memory: self.tokenizer_cache_l1_max_memory,
                },
                reasoning_parser: self.reasoning_parser.clone(),
                tool_call_parser: self.tool_call_parser.clone(),
            },
            storage: StorageConfig {
                history_backend: self.history_backend.into(),
                oracle,
                postgres,
                redis,
            },
            extensions: ExtensionConfig {
                mcp_config: self.load_mcp_config()?,
                enable_wasm: self.enable_wasm,
            },
            observability: ObservabilityConfig {
                log_dir: self.log_dir.clone(),
                log_level: Some(self.log_level.clone()),
                json_log: self.json_log,
                prometheus_host: self.prometheus_host.clone(),
                prometheus_port: self.prometheus_port,
                prometheus_duration_buckets: (!self.prometheus_duration_buckets.is_empty())
                    .then(|| self.prometheus_duration_buckets.clone()),
                enable_trace: self.enable_trace,
                otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
            },
            security: SecurityConfig {
                api_key: self.api_key.clone(),
                control_plane_auth,
                server_tls: self.load_server_tls_config()?,
                ..Default::default()
            },
            discovery,
            mesh: self.build_mesh_config()?,
        };

        config.validate()?;
        Ok(config)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    #[value(name = "sglang")]
    Sglang,
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "trtllm")]
    Trtllm,
    #[value(name = "openai")]
    Openai,
    #[value(name = "anthropic")]
    Anthropic,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
            Self::Trtllm => "trtllm",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, ValueEnum)]
#[value(rename_all = "snake_case")]
enum PolicyKind {
    Random,
    RoundRobin,

    #[default]
    CacheAware,
    PowerOfTwo,
    PrefixHash,
    Bucket,
    Manual,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lowercase")]
enum HistoryBackendKind {
    Memory,
    None,
    Oracle,
    Postgres,
    Redis,
}

impl From<HistoryBackendKind> for HistoryBackend {
    fn from(value: HistoryBackendKind) -> Self {
        match value {
            HistoryBackendKind::Memory => Self::Memory,
            HistoryBackendKind::Oracle => Self::Oracle,
            HistoryBackendKind::Postgres => Self::Postgres,
            HistoryBackendKind::Redis => Self::Redis,
            HistoryBackendKind::None => Self::None,
        }
    }
}

/// Parse role mapping from CLI format "idp_role=gateway_role"
fn parse_role_mapping(mapping: &str) -> Option<(String, Role)> {
    let parts: Vec<&str> = mapping.splitn(2, '=').collect();
    if parts.len() != 2 {
        eprintln!(
            "WARNING: Invalid role mapping format '{}'. Expected 'idp_role=gateway_role'",
            mapping
        );
        return None;
    }
    let idp_role = parts[0].to_string();
    let gateway_role = match parts[1].to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        other => {
            eprintln!(
                "WARNING: Invalid gateway role '{}' in mapping. Valid roles: admin, user",
                other
            );
            return None;
        }
    };
    Some((idp_role, gateway_role))
}

/// Parse control plane API key from CLI format "id:name:role:key"
fn parse_control_plane_api_key(key_str: &str) -> Option<ApiKeyEntry> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        eprintln!(
            "WARNING: Invalid control-plane-api-key format '{}'. Expected 'id:name:role:key'",
            key_str
        );
        return None;
    }
    let id = parts[0];
    let name = parts[1];
    let role_str = parts[2];
    let key = parts[3];

    let role = match role_str.to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        other => {
            eprintln!(
                "WARNING: Invalid role '{}' in control-plane-api-key. Valid roles: admin, user",
                other
            );
            return None;
        }
    };

    Some(ApiKeyEntry::new(id, name, key, role))
}

fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
    for url in worker_urls {
        if url.starts_with("grpc://") || url.starts_with("grpcs://") {
            return ConnectionMode::Grpc { port: None };
        }
    }
    ConnectionMode::Http
}

fn parse_label_selectors(selector_list: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for item in selector_list {
        if let Some(eq_pos) = item.find('=') {
            let key = item[..eq_pos].to_string();
            let value = item[eq_pos + 1..].to_string();
            map.insert(key, value);
        }
    }
    map
}
