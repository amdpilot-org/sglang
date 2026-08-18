use super::{
    extensions::ExtensionConfig,
    infrastructure::{DiscoveryConfig, MeshConfig, ObservabilityConfig},
    model::ModelConfig,
    routing::{PolicyConfig, RoutingConfig, RoutingMode},
    server::{HttpServerConfig, SecurityConfig},
    storage::StorageConfig,
    worker_pool::WorkerPoolConfig,
};

/// Complete, resolved configuration for one gateway process.
///
/// Each field maps to a component started by the gateway. The hierarchy stops
/// here so consumers can depend on only the configuration they need.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub server: HttpServerConfig,
    pub routing: RoutingConfig,
    pub workers: WorkerPoolConfig,
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

    /// Enables service discovery with a fully resolved configuration.
    pub fn with_discovery(mut self, discovery: DiscoveryConfig) -> Self {
        self.discovery = Some(discovery);
        self
    }

    pub fn has_discovery(&self) -> bool {
        self.discovery.is_some()
    }

    pub fn validate(&self) -> super::ConfigResult<()> {
        crate::config::validation::ConfigValidator::validate_gateway(self)
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            server: HttpServerConfig::default(),
            routing: RoutingConfig::default(),
            workers: WorkerPoolConfig::default(),
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
