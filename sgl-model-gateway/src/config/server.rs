/// Configuration for the gateway's public HTTP listener.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub api_key: Option<String>,
    pub control_plane_auth: Option<crate::auth::ControlPlaneAuthConfig>,
    pub server_tls: Option<ServerTlsConfig>,
    pub client_identity: Option<Vec<u8>>,
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
#[derive(Debug, Clone)]
pub struct ServerTlsConfig {
    pub certificate: Vec<u8>,
    pub private_key: Vec<u8>,
}
