//! Test configuration builders to reduce duplication across tests
//!
//! Provides pre-configured GatewayConfig and MockWorkerConfig builders
//! for common test scenarios.

use smg::config::{
    CircuitBreakerConfig, ConfigResult, GatewayConfig, HistoryBackend, ManualAssignmentMode,
    OracleConfig, PolicyConfig, RetryConfig, RoutingMode,
};

use super::mock_worker::{HealthStatus, MockWorkerConfig, WorkerType};

/// Default test configuration values
pub mod defaults {
    pub const HOST: &str = "127.0.0.1";
    pub const MAX_PAYLOAD_SIZE: usize = 256 * 1024 * 1024; // 256MB
    pub const REQUEST_TIMEOUT_SECS: u64 = 600;
    pub const WORKER_STARTUP_TIMEOUT_SECS: u64 = 5;
    pub const WORKER_STARTUP_CHECK_INTERVAL_SECS: u64 = 1;
    pub const MAX_CONCURRENT_REQUESTS: i32 = 64;
    pub const QUEUE_TIMEOUT_SECS: u64 = 60;
}

/// Factory for common gateway configuration patterns used by tests.
pub struct TestGatewayConfig;

impl TestGatewayConfig {
    fn base(port: u16, policy: PolicyConfig) -> GatewayConfig {
        let mut config = GatewayConfig::default();
        config.server.host = defaults::HOST.to_string();
        config.server.port = port;
        config.server.max_payload_size = defaults::MAX_PAYLOAD_SIZE;
        config.routing.mode = RoutingMode::Regular {
            worker_urls: Vec::new(),
        };
        config.routing.policy = policy;
        config.routing.max_concurrent_requests = defaults::MAX_CONCURRENT_REQUESTS;
        config.routing.queue_timeout_secs = defaults::QUEUE_TIMEOUT_SECS;
        config.workers.request_timeout_secs = defaults::REQUEST_TIMEOUT_SECS;
        config.workers.startup_timeout_secs = defaults::WORKER_STARTUP_TIMEOUT_SECS;
        config.workers.startup_check_interval_secs = defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS;
        config
    }

    /// Create a basic round-robin config for routing tests
    pub fn round_robin(port: u16) -> GatewayConfig {
        Self::base(port, PolicyConfig::RoundRobin)
    }

    /// Create a random load balancing config
    pub fn random(port: u16) -> GatewayConfig {
        Self::base(port, PolicyConfig::Random)
    }

    /// Create a cache-aware config for routing tests
    pub fn cache_aware(port: u16) -> GatewayConfig {
        Self::base(
            port,
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.5,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        )
    }

    /// Create a power-of-two config
    pub fn power_of_two(port: u16) -> GatewayConfig {
        Self::base(
            port,
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
        )
    }

    /// Create a manual routing config (for sticky routing tests)
    pub fn manual(port: u16) -> GatewayConfig {
        Self::manual_with_mode(port, ManualAssignmentMode::Random)
    }

    /// Create a manual routing config with min_group assignment mode
    pub fn manual_min_group(port: u16) -> GatewayConfig {
        Self::manual_with_mode(port, ManualAssignmentMode::MinGroup)
    }

    /// Create a manual routing config with specified assignment mode
    pub fn manual_with_mode(port: u16, assignment_mode: ManualAssignmentMode) -> GatewayConfig {
        Self::base(
            port,
            PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 3600,
                assignment_mode,
            },
        )
    }

    /// Create a config with custom concurrent request limit (for rate limiting tests)
    pub fn with_concurrency(port: u16, max_concurrent: i32) -> GatewayConfig {
        let mut config = Self::round_robin(port);
        config.routing.max_concurrent_requests = max_concurrent;
        config
    }

    /// Create a config with custom payload size limit
    pub fn with_payload_limit(port: u16, max_payload_size: usize) -> GatewayConfig {
        let mut config = Self::round_robin(port);
        config.server.max_payload_size = max_payload_size;
        config
    }

    /// Create a config with short timeouts (for timeout/retry tests)
    pub fn with_short_timeouts(port: u16) -> GatewayConfig {
        let mut config = Self::round_robin(port);
        config.workers.request_timeout_secs = 5;
        config.workers.startup_timeout_secs = 2;
        config.workers.startup_check_interval_secs = 1;
        config.routing.queue_timeout_secs = 5;
        config
    }

    /// Create a round-robin config with retry settings
    pub fn round_robin_with_retry(port: u16, retry_config: RetryConfig) -> GatewayConfig {
        let mut config = Self::round_robin(port);
        config.workers.retry = retry_config;
        config
    }

    /// Create a round-robin config with circuit breaker
    pub fn round_robin_with_circuit_breaker(
        port: u16,
        circuit_breaker: CircuitBreakerConfig,
    ) -> GatewayConfig {
        let mut config = Self::round_robin(port);
        config.workers.circuit_breaker = circuit_breaker;
        config
    }

    /// Create a round-robin config with both retry and circuit breaker
    pub fn round_robin_with_reliability(
        port: u16,
        retry_config: RetryConfig,
        circuit_breaker: CircuitBreakerConfig,
    ) -> GatewayConfig {
        let mut config = Self::round_robin(port);
        config.workers.retry = retry_config;
        config.workers.circuit_breaker = circuit_breaker;
        config
    }
}

/// Fluent test-data factory for existing integration tests.
///
/// This is intentionally local to `tests/`: production configuration is
/// constructed directly through `GatewayConfig` and has no builder API.
pub struct TestGatewayConfigBuilder {
    config: GatewayConfig,
}

impl TestGatewayConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: GatewayConfig::default(),
        }
    }

    pub fn regular_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.routing.mode = RoutingMode::Regular { worker_urls };
        self
    }

    pub fn openai_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.routing.mode = RoutingMode::OpenAI { worker_urls };
        self
    }

    pub fn prefill_decode_mode(
        mut self,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    ) -> Self {
        self.config.routing.mode = RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy: None,
            decode_policy: None,
        };
        self
    }

    pub fn policy(mut self, policy: PolicyConfig) -> Self {
        self.config.routing.policy = policy;
        self
    }

    pub fn random_policy(self) -> Self {
        self.policy(PolicyConfig::Random)
    }

    pub fn round_robin_policy(self) -> Self {
        self.policy(PolicyConfig::RoundRobin)
    }

    pub fn power_of_two_policy(self, load_check_interval_secs: u64) -> Self {
        self.policy(PolicyConfig::PowerOfTwo {
            load_check_interval_secs,
        })
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.server.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.server.port = port;
        self
    }

    pub fn max_payload_size(mut self, max_payload_size: usize) -> Self {
        self.config.server.max_payload_size = max_payload_size;
        self
    }

    pub fn request_id_headers(mut self, request_id_headers: Vec<String>) -> Self {
        self.config.server.request_id_headers = Some(request_id_headers);
        self
    }

    pub fn request_timeout_secs(mut self, request_timeout_secs: u64) -> Self {
        self.config.workers.request_timeout_secs = request_timeout_secs;
        self
    }

    pub fn worker_startup_timeout_secs(mut self, startup_timeout_secs: u64) -> Self {
        self.config.workers.startup_timeout_secs = startup_timeout_secs;
        self
    }

    pub fn worker_startup_check_interval_secs(mut self, check_interval_secs: u64) -> Self {
        self.config.workers.startup_check_interval_secs = check_interval_secs;
        self
    }

    pub fn max_concurrent_requests(mut self, max_concurrent_requests: i32) -> Self {
        self.config.routing.max_concurrent_requests = max_concurrent_requests;
        self
    }

    pub fn queue_size(mut self, queue_size: usize) -> Self {
        self.config.routing.queue_size = queue_size;
        self
    }

    pub fn queue_timeout_secs(mut self, queue_timeout_secs: u64) -> Self {
        self.config.routing.queue_timeout_secs = queue_timeout_secs;
        self
    }

    pub fn rate_limit_tokens_per_second(mut self, tokens_per_second: i32) -> Self {
        self.config.routing.rate_limit_tokens_per_second = Some(tokens_per_second);
        self
    }

    pub fn retry_config(mut self, retry: RetryConfig) -> Self {
        self.config.workers.retry = retry;
        self
    }

    pub fn circuit_breaker_config(mut self, circuit_breaker: CircuitBreakerConfig) -> Self {
        self.config.workers.circuit_breaker = circuit_breaker;
        self
    }

    pub fn disable_retries(mut self) -> Self {
        self.config.workers.disable_retries = true;
        self
    }

    pub fn disable_circuit_breaker(mut self) -> Self {
        self.config.workers.disable_circuit_breaker = true;
        self
    }

    pub fn log_level(mut self, log_level: impl Into<String>) -> Self {
        self.config.observability.log_level = Some(log_level.into());
        self
    }

    pub fn enable_trace(mut self, endpoint: impl Into<String>) -> Self {
        self.config.observability.enable_trace = true;
        self.config.observability.otlp_traces_endpoint = endpoint.into();
        self
    }

    pub fn history_backend(mut self, history_backend: HistoryBackend) -> Self {
        self.config.storage.history_backend = history_backend;
        self
    }

    pub fn oracle_history(mut self, oracle: OracleConfig) -> Self {
        self.config.storage.history_backend = HistoryBackend::Oracle;
        self.config.storage.oracle = Some(oracle);
        self
    }

    pub fn build(self) -> ConfigResult<GatewayConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    pub fn build_unchecked(self) -> GatewayConfig {
        self.config
    }
}

impl Default for TestGatewayConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for common MockWorkerConfig patterns
pub struct TestWorkerConfig;

impl TestWorkerConfig {
    /// Create a healthy worker config
    pub fn healthy(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }

    /// Create multiple healthy workers with sequential ports
    pub fn healthy_workers(start_port: u16, count: u16) -> Vec<MockWorkerConfig> {
        (0..count).map(|i| Self::healthy(start_port + i)).collect()
    }

    /// Create an unhealthy worker config
    pub fn unhealthy(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Unhealthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }

    /// Create a slow worker config (for timeout tests)
    pub fn slow(port: u16, delay_ms: u64) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: delay_ms,
            fail_rate: 0.0,
        }
    }

    /// Create multiple slow workers with sequential ports
    pub fn slow_workers(start_port: u16, count: u16, delay_ms: u64) -> Vec<MockWorkerConfig> {
        (0..count)
            .map(|i| Self::slow(start_port + i, delay_ms))
            .collect()
    }

    /// Create a flaky worker config (for retry/fault tolerance tests)
    pub fn flaky(port: u16, fail_rate: f32) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate,
        }
    }

    /// Create a decode worker config (for PD routing tests)
    pub fn decode(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }

    /// Create a prefill worker config (for PD routing tests)
    pub fn prefill(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_workers() {
        let workers = TestWorkerConfig::healthy_workers(8000, 3);
        assert_eq!(workers.len(), 3);
        assert_eq!(workers[0].port, 8000);
        assert_eq!(workers[1].port, 8001);
        assert_eq!(workers[2].port, 8002);
    }
}
