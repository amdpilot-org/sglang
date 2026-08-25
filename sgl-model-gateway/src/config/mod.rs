pub mod cli;
pub mod gateway;
pub(crate) mod validation;

pub use gateway::{
    CircuitBreakerConfig, DiscoveryConfig, ExtensionConfig, GatewayConfig, HealthCheckConfig,
    HistoryBackend, HttpServerConfig, ManualAssignmentMode, MeshConfig, ModelConfig,
    ObservabilityConfig, OracleConfig, PolicyConfig, PostgresConfig, RedisConfig, RetryConfig,
    RoutingConfig, RoutingMode, SecurityConfig, ServerTlsConfig, StorageConfig,
    TokenizerCacheConfig, WorkerConfig, DEFAULT_CONNECT_TIMEOUT_SECS,
    DEFAULT_POOL_IDLE_TIMEOUT_SECS, DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_TCP_KEEPALIVE_SECS,
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
