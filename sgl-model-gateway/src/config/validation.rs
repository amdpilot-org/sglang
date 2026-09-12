use super::{
    gateway::{
        CircuitBreakerConfig, DiscoveryConfig, GatewayConfig, HistoryBackend, HttpServerConfig,
        MeshConfig, ObservabilityConfig, OracleConfig, PolicyConfig, RetryConfig, RoutingConfig,
        RoutingMode, SecurityConfig, StorageConfig, TokenizerCacheConfig, WorkerConfig,
    },
    ConfigError, ConfigResult,
};

/// Validates resolved configuration before gateway startup.
pub(crate) struct ConfigValidator;

impl ConfigValidator {
    /// Validates each configuration section and their compatibility rules.
    pub(crate) fn validate_gateway(config: &GatewayConfig) -> ConfigResult<()> {
        Self::validate_http_server(&config.server)?;
        Self::validate_routing(&config.routing)?;
        Self::validate_worker_pool(&config.workers)?;
        Self::validate_tokenizer_cache(&config.model.tokenizer_cache)?;
        Self::validate_storage(&config.storage)?;
        Self::validate_observability(&config.observability)?;
        Self::validate_security(&config.security)?;

        if let Some(discovery) = &config.discovery {
            Self::validate_gateway_discovery(discovery, &config.routing)?;
        }
        if let Some(mesh) = &config.mesh {
            Self::validate_mesh(mesh)?;
        }

        Self::validate_gateway_compatibility(config)
    }

    fn validate_oracle(oracle: &OracleConfig) -> ConfigResult<()> {
        if oracle.username.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle.username".to_string(),
            });
        }

        if oracle.password.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle.password".to_string(),
            });
        }

        if oracle.connect_descriptor.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle_dsn or oracle_tns_alias".to_string(),
            });
        }

        if oracle.pool_min < 1 {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_min".to_string(),
                value: oracle.pool_min.to_string(),
                reason: "Must be at least 1".to_string(),
            });
        }

        if oracle.pool_max < oracle.pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_max".to_string(),
                value: oracle.pool_max.to_string(),
                reason: "Must be >= oracle.pool_min".to_string(),
            });
        }

        if oracle.pool_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_timeout_secs".to_string(),
                value: oracle.pool_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        Ok(())
    }

    fn validate_http_server(server: &HttpServerConfig) -> ConfigResult<()> {
        if server.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.port".to_string(),
                value: server.port.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if server.max_payload_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.max_payload_size".to_string(),
                value: server.max_payload_size.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_routing(routing: &RoutingConfig) -> ConfigResult<()> {
        Self::validate_mode(&routing.mode)?;
        Self::validate_policy(&routing.policy)?;
        if routing.queue_size > 0 && routing.queue_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "routing.queue_timeout_secs".to_string(),
                value: routing.queue_timeout_secs.to_string(),
                reason: "Must be > 0 when queue_size > 0".to_string(),
            });
        }
        if let Some(tokens_per_second) = routing.rate_limit_tokens_per_second {
            if tokens_per_second < 0 {
                return Err(ConfigError::InvalidValue {
                    field: "routing.rate_limit_tokens_per_second".to_string(),
                    value: tokens_per_second.to_string(),
                    reason: "Must be >= 0 when specified".to_string(),
                });
            }
        }
        Ok(())
    }

    fn validate_worker_pool(workers: &WorkerConfig) -> ConfigResult<()> {
        for (field, value) in [
            ("workers.request_timeout_secs", workers.request_timeout_secs),
            ("workers.startup_timeout_secs", workers.startup_timeout_secs),
            (
                "workers.startup_check_interval_secs",
                workers.startup_check_interval_secs,
            ),
            (
                "workers.pool_idle_timeout_secs",
                workers.pool_idle_timeout_secs,
            ),
            ("workers.connect_timeout_secs", workers.connect_timeout_secs),
            ("workers.tcp_keepalive_secs", workers.tcp_keepalive_secs),
            (
                "workers.health_check.timeout_secs",
                workers.health_check.timeout_secs,
            ),
            (
                "workers.health_check.check_interval_secs",
                workers.health_check.check_interval_secs,
            ),
        ] {
            if value == 0 {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: value.to_string(),
                    reason: "Must be > 0".to_string(),
                });
            }
        }
        if workers.pool_max_idle_per_host == 0 {
            return Err(ConfigError::InvalidValue {
                field: "workers.pool_max_idle_per_host".to_string(),
                value: "0".to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if workers.health_check.failure_threshold == 0
            || workers.health_check.success_threshold == 0
        {
            return Err(ConfigError::ValidationFailed {
                reason: "Health check thresholds must be > 0".to_string(),
            });
        }

        let mut retry = workers.retry.clone();
        if workers.disable_retries {
            retry.max_retries = 1;
        }
        Self::validate_retry(&retry)?;

        let mut circuit_breaker = workers.circuit_breaker.clone();
        if workers.disable_circuit_breaker {
            circuit_breaker.failure_threshold = u32::MAX;
        }
        Self::validate_circuit_breaker(&circuit_breaker)
    }

    fn validate_storage(storage: &StorageConfig) -> ConfigResult<()> {
        match &storage.history_backend {
            HistoryBackend::None | HistoryBackend::Memory => Ok(()),
            HistoryBackend::Oracle => storage
                .oracle
                .as_ref()
                .ok_or_else(|| ConfigError::MissingRequired {
                    field: "storage.oracle".to_string(),
                })
                .and_then(Self::validate_oracle),
            HistoryBackend::Postgres => storage
                .postgres
                .as_ref()
                .ok_or_else(|| ConfigError::MissingRequired {
                    field: "storage.postgres".to_string(),
                })?
                .validate()
                .map_err(|error| ConfigError::ValidationFailed {
                    reason: error.to_string(),
                }),
            HistoryBackend::Redis => storage
                .redis
                .as_ref()
                .ok_or_else(|| ConfigError::MissingRequired {
                    field: "storage.redis".to_string(),
                })?
                .validate()
                .map_err(|error| ConfigError::ValidationFailed {
                    reason: error.to_string(),
                }),
        }
    }

    fn validate_observability(observability: &ObservabilityConfig) -> ConfigResult<()> {
        if observability.prometheus_port == 0 || observability.prometheus_host.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "Prometheus host must be non-empty and port must be > 0".to_string(),
            });
        }
        if !observability.enable_trace {
            return Ok(());
        }
        Self::validate_otlp_endpoint(
            &observability.otlp_traces_endpoint,
            "observability.otlp_traces_endpoint",
        )
    }

    fn validate_security(security: &SecurityConfig) -> ConfigResult<()> {
        if security.client_identity.as_ref().is_some_and(Vec::is_empty) {
            return Err(ConfigError::ValidationFailed {
                reason: "Client identity cannot be empty".to_string(),
            });
        }
        if security.ca_certificates.iter().any(Vec::is_empty) {
            return Err(ConfigError::ValidationFailed {
                reason: "CA certificates cannot be empty".to_string(),
            });
        }
        if let Some(tls) = &security.server_tls {
            if tls.certificate.is_empty() || tls.private_key.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Server TLS certificate and private key cannot be empty".to_string(),
                });
            }
        }
        Ok(())
    }

    fn validate_gateway_discovery(
        discovery: &DiscoveryConfig,
        routing: &RoutingConfig,
    ) -> ConfigResult<()> {
        if discovery.port == 0 || discovery.check_interval_secs == 0 {
            return Err(ConfigError::ValidationFailed {
                reason: "Discovery port and check interval must be > 0".to_string(),
            });
        }
        match &routing.mode {
            RoutingMode::Regular { .. } if discovery.selector.is_empty() => {
                Err(ConfigError::ValidationFailed {
                    reason: "Regular mode with service discovery requires a non-empty selector"
                        .to_string(),
                })
            }
            RoutingMode::PrefillDecode { .. }
                if discovery.prefill_selector.is_empty()
                    && discovery.decode_selector.is_empty() =>
            {
                Err(ConfigError::ValidationFailed {
                    reason: "PD mode with service discovery requires a prefill or decode selector"
                        .to_string(),
                })
            }
            RoutingMode::OpenAI { .. } => Err(ConfigError::ValidationFailed {
                reason: "OpenAI mode does not support service discovery".to_string(),
            }),
            _ => Ok(()),
        }
    }

    fn validate_mesh(mesh: &MeshConfig) -> ConfigResult<()> {
        if mesh.self_name.trim().is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "mesh.self_name".to_string(),
            });
        }
        Ok(())
    }

    /// Checks restrictions that span routing, discovery, and worker settings.
    fn validate_gateway_compatibility(config: &GatewayConfig) -> ConfigResult<()> {
        if config.routing.enable_igw || config.discovery.is_some() {
            return Ok(());
        }

        if let PolicyConfig::PowerOfTwo { .. } = &config.routing.policy {
            if config.routing.mode.worker_count() < 2 {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Power-of-two policy requires at least 2 workers".to_string(),
                });
            }
        }

        if let RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
        } = &config.routing.mode
        {
            if matches!(prefill_policy, Some(PolicyConfig::PowerOfTwo { .. }))
                && prefill_urls.len() < 2
            {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Power-of-two policy for prefill requires at least 2 prefill workers"
                        .to_string(),
                });
            }
            if matches!(decode_policy, Some(PolicyConfig::PowerOfTwo { .. }))
                && decode_urls.len() < 2
            {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Power-of-two policy for decode requires at least 2 decode workers"
                        .to_string(),
                });
            }
            if matches!(decode_policy, Some(PolicyConfig::Bucket { .. })) {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Decode policy should not be allowed to be bucket".to_string(),
                });
            }
        }

        Ok(())
    }

    fn validate_otlp_endpoint(endpoint: &str, field: &str) -> ConfigResult<()> {
        let Some((host, port)) = endpoint.rsplit_once(':') else {
            return Err(ConfigError::InvalidValue {
                field: field.to_string(),
                value: endpoint.to_string(),
                reason: "expected format <host>:<port>".to_string(),
            });
        };
        if host.is_empty() || !matches!(port.parse::<u16>(), Ok(port) if port > 0) {
            return Err(ConfigError::InvalidValue {
                field: field.to_string(),
                value: endpoint.to_string(),
                reason: "host must be non-empty and port must be between 1 and 65535".to_string(),
            });
        }
        Ok(())
    }

    fn validate_mode(mode: &RoutingMode) -> ConfigResult<()> {
        match mode {
            RoutingMode::Regular { worker_urls } => {
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
                // Allow empty URLs without service discovery to match legacy behavior
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
            } => {
                // Allow empty URLs even without service discovery to support dynamic worker addition
                // URLs will be validated if provided
                if !prefill_urls.is_empty() {
                    let prefill_url_strings: Vec<String> =
                        prefill_urls.iter().map(|(url, _)| url.clone()).collect();
                    Self::validate_urls(&prefill_url_strings)?;
                }
                if !decode_urls.is_empty() {
                    Self::validate_urls(decode_urls)?;
                }

                for (_url, port) in prefill_urls {
                    if let Some(port) = port {
                        if *port == 0 {
                            return Err(ConfigError::InvalidValue {
                                field: "bootstrap_port".to_string(),
                                value: port.to_string(),
                                reason: "Port must be between 1 and 65535".to_string(),
                            });
                        }
                    }
                }

                if let Some(p_policy) = prefill_policy {
                    Self::validate_policy(p_policy)?;
                }
                if let Some(d_policy) = decode_policy {
                    Self::validate_policy(d_policy)?;
                }
            }
            RoutingMode::OpenAI { worker_urls } => {
                // Allow empty URLs to support dynamic worker addition
                // URLs will be validated if provided
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
            }
        }
        Ok(())
    }

    fn validate_policy(policy: &PolicyConfig) -> ConfigResult<()> {
        match policy {
            PolicyConfig::Random
            | PolicyConfig::RoundRobin
            | PolicyConfig::Manual { .. }
            | PolicyConfig::ConsistentHashing => {}
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold: _,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                if !(0.0..=1.0).contains(cache_threshold) {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_threshold".to_string(),
                        value: cache_threshold.to_string(),
                        reason: "Must be between 0.0 and 1.0".to_string(),
                    });
                }

                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *eviction_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "eviction_interval_secs".to_string(),
                        value: eviction_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *max_tree_size == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "max_tree_size".to_string(),
                        value: max_tree_size.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                if *load_check_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_check_interval_secs".to_string(),
                        value: load_check_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::Bucket {
                balance_abs_threshold: _,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *bucket_adjust_interval_secs < 1 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be >= 1s".to_string(),
                    });
                }
                if *bucket_adjust_interval_secs >= 4294967296 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be < 4294967296s".to_string(),
                    });
                }
            }
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                if *prefix_token_count == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "prefix_token_count".to_string(),
                        value: prefix_token_count.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *load_factor < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_factor".to_string(),
                        value: load_factor.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_retry(retry: &RetryConfig) -> ConfigResult<()> {
        if retry.max_retries < 1 {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_retries".to_string(),
                value: retry.max_retries.to_string(),
                reason: "Must be >= 1 (set to 1 to effectively disable retries)".to_string(),
            });
        }
        if retry.initial_backoff_ms == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.initial_backoff_ms".to_string(),
                value: retry.initial_backoff_ms.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if retry.max_backoff_ms < retry.initial_backoff_ms {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_backoff_ms".to_string(),
                value: retry.max_backoff_ms.to_string(),
                reason: "Must be >= initial_backoff_ms".to_string(),
            });
        }
        if retry.backoff_multiplier < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.backoff_multiplier".to_string(),
                value: retry.backoff_multiplier.to_string(),
                reason: "Must be >= 1.0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&retry.jitter_factor) {
            return Err(ConfigError::InvalidValue {
                field: "retry.jitter_factor".to_string(),
                value: retry.jitter_factor.to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_circuit_breaker(cb: &CircuitBreakerConfig) -> ConfigResult<()> {
        if cb.failure_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.failure_threshold".to_string(),
                value: cb.failure_threshold.to_string(),
                reason: "Must be >= 1 (set to u32::MAX to effectively disable CB)".to_string(),
            });
        }
        if cb.success_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.success_threshold".to_string(),
                value: cb.success_threshold.to_string(),
                reason: "Must be >= 1".to_string(),
            });
        }
        if cb.timeout_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.timeout_duration_secs".to_string(),
                value: cb.timeout_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if cb.window_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.window_duration_secs".to_string(),
                value: cb.window_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_tokenizer_cache(cache: &TokenizerCacheConfig) -> ConfigResult<()> {
        if cache.enable_l0 && cache.l0_max_entries == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l0_max_entries".to_string(),
                value: cache.l0_max_entries.to_string(),
                reason: "Must be > 0 when L0 cache is enabled".to_string(),
            });
        }

        if cache.enable_l1 && cache.l1_max_memory == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l1_max_memory".to_string(),
                value: cache.l1_max_memory.to_string(),
                reason: "Must be > 0 when L1 cache is enabled".to_string(),
            });
        }

        Ok(())
    }

    fn validate_urls(urls: &[String]) -> ConfigResult<()> {
        for url in urls {
            if url.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL cannot be empty".to_string(),
                });
            }

            if !url.starts_with("http://")
                && !url.starts_with("https://")
                && !url.starts_with("grpc://")
            {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL must start with http://, https://, or grpc://".to_string(),
                });
            }

            match ::url::Url::parse(url) {
                Ok(parsed) => {
                    if parsed.host_str().is_none() {
                        return Err(ConfigError::InvalidValue {
                            field: "worker_url".to_string(),
                            value: url.clone(),
                            reason: "URL must have a valid host".to_string(),
                        });
                    }
                }
                Err(e) => {
                    return Err(ConfigError::InvalidValue {
                        field: "worker_url".to_string(),
                        value: url.clone(),
                        reason: format!("Invalid URL format: {}", e),
                    });
                }
            }
        }
        Ok(())
    }
}
