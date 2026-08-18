use std::{collections::HashMap, net::SocketAddr};

/// Logging, metrics, and tracing settings for the gateway process.
#[derive(Debug, Clone)]
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

/// Kubernetes-based worker and router discovery settings.
///
/// `GatewayConfig.discovery = None` disables discovery, so this type does not
/// carry a second, redundant `enabled` flag.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub selector: HashMap<String, String>,
    pub namespace: Option<String>,
    pub port: u16,
    pub check_interval_secs: u64,
    pub pd_mode: bool,
    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    pub bootstrap_port_annotation: String,
    pub router_selector: HashMap<String, String>,
    pub router_mesh_port_annotation: String,
    pub igw_mode: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            selector: HashMap::new(),
            namespace: None,
            port: 8000,
            check_interval_secs: 120,
            pd_mode: false,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            igw_mode: false,
        }
    }
}

/// Configuration for this gateway's mesh node.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    pub self_name: String,
    pub self_addr: SocketAddr,
    pub init_peer: Option<SocketAddr>,
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
