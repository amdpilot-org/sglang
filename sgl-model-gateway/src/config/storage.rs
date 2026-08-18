pub use data_connector::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig};

/// Configuration for conversation and request history storage.
#[derive(Debug, Clone)]
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
