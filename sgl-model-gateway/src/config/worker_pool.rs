use serde::{Deserialize, Serialize};

use crate::core::ConnectionMode;

pub const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 50;
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 500;
pub const DEFAULT_TCP_KEEPALIVE_SECS: u64 = 30;

/// Configuration for connections to and lifecycle management of the worker pool.
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
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

impl Default for WorkerPoolConfig {
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

impl WorkerPoolConfig {
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
