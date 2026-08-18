pub mod cli;
pub mod extensions;
pub mod gateway;
pub mod infrastructure;
pub mod model;
pub mod routing;
pub mod server;
pub mod storage;
pub(crate) mod validation;
pub mod worker_pool;

pub use extensions::ExtensionConfig;
pub use gateway::GatewayConfig;
pub use infrastructure::{DiscoveryConfig, MeshConfig, ObservabilityConfig};
pub use model::{ModelConfig, TokenizerCacheConfig};
pub use routing::{ManualAssignmentMode, PolicyConfig, RoutingConfig, RoutingMode};
pub use server::{HttpServerConfig, SecurityConfig, ServerTlsConfig};
pub use storage::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig, StorageConfig};
pub use worker_pool::{
    CircuitBreakerConfig, HealthCheckConfig, RetryConfig, WorkerPoolConfig,
    DEFAULT_CONNECT_TIMEOUT_SECS, DEFAULT_POOL_IDLE_TIMEOUT_SECS, DEFAULT_POOL_MAX_IDLE_PER_HOST,
    DEFAULT_TCP_KEEPALIVE_SECS,
};

#[derive(Debug, thiserror::Error)]
/// Errors produced while resolving or validating gateway configuration.
pub enum ConfigError {
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Invalid value for field '{field}': {value} - {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("Incompatible configuration: {reason}")]
    IncompatibleConfig { reason: String },

    #[error("Missing required field: {field}")]
    MissingRequired { field: String },
}

/// Result type used by configuration construction and validation.
pub type ConfigResult<T> = Result<T, ConfigError>;
