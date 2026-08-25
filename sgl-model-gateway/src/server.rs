use std::{sync::Arc, time::Duration};

use anyhow::{anyhow, Context, Error, Result};
use axum::Router;
use rustls::crypto::ring;
use smg_mesh::{rate_limit_window::RateLimitWindow, MeshServerHandler, MeshSyncManager};
use tokio::{signal, spawn, task::JoinHandle};
use tracing::{debug, error, info, warn};
use wfaas::LoggingSubscriber;

use crate::{
    app_context::AppContext,
    app_state::AppState,
    config::{GatewayConfig, ObservabilityConfig, RoutingMode},
    core::{
        job_queue::{JobQueue, JobQueueConfig},
        steps::{TokenizerConfigRequest, WorkflowEngines},
        worker_manager::WorkerManager,
        Job,
    },
    middleware::{self, AuthConfig, QueuedRequest},
    observability::{
        logging::{self, LogGuard, LoggingConfig},
        metrics::{self, PrometheusConfig},
        otel_trace,
    },
    routers::{app::build_app, router_manager::RouterManager, RouterTrait},
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    tokenizer::TokenizerRegistry,
};

/// Starts gateway components, serves HTTP traffic, then stops owned background services.
pub async fn startup(config: GatewayConfig) -> Result<()> {
    log_startup(&config);

    // start prometheus
    metrics::start_prometheus(PrometheusConfig {
        host: config.observability.prometheus_host.clone(),
        port: config.observability.prometheus_port,
        duration_buckets: config.observability.prometheus_duration_buckets.clone(),
    });

    let (mesh_handler, mesh_sync_manager) = start_mesh(config.mesh.as_ref());
    let mesh_shutdown_handler = mesh_handler.clone();
    let app_context = initialize_app_context(&config).await?;
    submit_startup_jobs(&app_context, &config).await?;

    // start mcp manager
    if let Some(mcp_manager) = app_context.mcp_manager.get() {
        Arc::clone(mcp_manager).spawn_background_refresh_all(Duration::from_secs(600));
        debug!("Started background refresh for all MCP servers (every 10 minutes)");
    }

    let health_checker = start_health_checker(&app_context, &config);

    // start load monitor
    if let Some(load_monitor) = &app_context.load_monitor {
        load_monitor.start().await;
        debug!("Started LoadMonitor for PowerOfTwo policies");
    }

    let (concurrency_queue_tx, concurrency_processor) =
        start_concurrency_queue(&app_context, &config);

    configure_mesh_sync(&app_context, mesh_sync_manager.as_ref());

    let router_manager = create_router_manager(&app_context, &config).await?;

    let app_state = build_app_state(
        app_context.clone(),
        router_manager,
        concurrency_queue_tx,
        mesh_handler,
        mesh_sync_manager,
    );

    start_service_discovery_if_enabled(&config, &app_state).await;

    let app = build_http_app(&config, app_state).await;
    let server_result = serve_http(app, &config).await;

    shutdown_background_services(
        &app_context,
        health_checker,
        concurrency_processor,
        mesh_shutdown_handler,
    )
    .await;

    server_result
}

/// Stops background services that outlive the HTTP server.
async fn shutdown_background_services(
    app_context: &Arc<AppContext>,
    health_checker: Option<crate::core::worker::HealthChecker>,
    concurrency_processor: Option<JoinHandle<()>>,
    mesh_handler: Option<Arc<MeshServerHandler>>,
) {
    const BACKGROUND_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

    if let Some(processor) = concurrency_processor {
        processor.abort();
        let _ = processor.await;
    }

    if let Some(load_monitor) = &app_context.load_monitor {
        load_monitor.stop().await;
    }

    if let Some(health_checker) = health_checker {
        if tokio::time::timeout(BACKGROUND_SHUTDOWN_TIMEOUT, health_checker.shutdown())
            .await
            .is_err()
        {
            warn!("Worker health checker did not stop before timeout");
        }
    }

    if let Some(mesh_handler) = mesh_handler {
        match tokio::time::timeout(
            BACKGROUND_SHUTDOWN_TIMEOUT,
            mesh_handler.graceful_shutdown(),
        )
        .await
        {
            Ok(Ok(())) => info!("Mesh shut down gracefully"),
            Ok(Err(error)) => warn!(%error, "Mesh graceful shutdown failed"),
            Err(_) => warn!("Mesh graceful shutdown timed out"),
        }
    }
}

/// Logs the effective HTTP, routing policy, and worker configuration.
fn log_startup(config: &GatewayConfig) {
    info!(
        host = %config.server.host,
        port = config.server.port,
        "HTTP server configured"
    );

    if config.routing.enable_igw {
        return;
    }

    info!(policy = config.routing.policy.name(), "routing configured");

    match &config.routing.mode {
        RoutingMode::Regular { worker_urls } | RoutingMode::OpenAI { worker_urls } => {
            info!(
                worker_count = worker_urls.len(),
                worker_urls = ?worker_urls,
                "workers configured"
            );
        }
        RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } => {
            info!(
                prefill_worker_count = prefill_urls.len(),
                decode_worker_count = decode_urls.len(),
                prefill_worker_urls = ?prefill_urls,
                decode_worker_urls = ?decode_urls,
                "PD workers configured"
            );
        }
    }
}

/// Starts the optional mesh server and returns its request and synchronization handles.
fn start_mesh(
    mesh_config: Option<&crate::config::MeshConfig>,
) -> (Option<Arc<MeshServerHandler>>, Option<Arc<MeshSyncManager>>) {
    let Some(mesh_config) = mesh_config else {
        return (None, None);
    };

    use smg_mesh::{partition::PartitionDetector, stores::StateStores, sync::MeshSyncManager};

    let stores = Arc::new(StateStores::with_self_name(mesh_config.self_name.clone()));
    let sync_manager = Arc::new(MeshSyncManager::new(
        stores.clone(),
        mesh_config.self_name.clone(),
    ));
    let partition_detector = Arc::new(PartitionDetector::default());
    sync_manager.update_rate_limit_membership();

    let window_manager = RateLimitWindow::new(sync_manager.clone(), 1);
    spawn(async move {
        window_manager.start_reset_task().await;
    });

    use smg_mesh::service::MeshServerBuilder;
    let builder = MeshServerBuilder::new(
        mesh_config.self_name.clone(),
        mesh_config.self_addr,
        mesh_config.init_peer,
    );
    let (mesh_server, handler) = builder.build_with_stores(Some(stores.clone()));

    let stores_for_server = stores.clone();
    let sync_manager_for_server = sync_manager.clone();
    spawn(async move {
        if let Err(error) = mesh_server
            .start_serve_with_stores(
                Some(stores_for_server),
                Some(sync_manager_for_server),
                Some(partition_detector),
            )
            .await
        {
            error!("Mesh server failed: {error}");
        }
    });

    (Some(Arc::new(handler)), Some(sync_manager))
}

/// Creates shared application state and the job-processing infrastructure.
async fn initialize_app_context(config: &GatewayConfig) -> Result<Arc<AppContext>> {
    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.server.host,
        config.server.port,
        config.routing.mode,
        config.routing.policy,
        config.server.max_payload_size / (1024 * 1024)
    );

    let app_context = Arc::new(
        AppContext::from_config(config.clone())
            .await
            .map_err(Error::msg)
            .context("create app context")?,
    );
    app_context.inflight_tracker.start_sampler(20);

    let worker_job_queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&app_context));
    app_context
        .worker_job_queue
        .set(worker_job_queue)
        .expect("JobQueue should only be initialized once");

    let engines = WorkflowEngines::new(config);
    engines.subscribe_all(Arc::new(LoggingSubscriber)).await;
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");
    debug!(
        "Workflow engines initialized (health check timeout: {}s)",
        config.workers.health_check.timeout_secs
    );

    Ok(app_context)
}

/// Queues startup work for configured tokenizers, workers, and MCP servers.
async fn submit_startup_jobs(app_context: &Arc<AppContext>, config: &GatewayConfig) -> Result<()> {
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");

    if let Some(tokenizer_source) = config
        .model
        .tokenizer_path
        .as_ref()
        .or(config.model.model_path.as_ref())
    {
        info!("Loading startup tokenizer from: {tokenizer_source}");
        job_queue
            .submit(Job::AddTokenizer {
                config: Box::new(TokenizerConfigRequest {
                    id: TokenizerRegistry::generate_id(),
                    name: tokenizer_source.clone(),
                    source: tokenizer_source.clone(),
                    chat_template_path: config.model.chat_template.clone(),
                    cache_config: config.model.tokenizer_cache.to_option(),
                    fail_on_duplicate: false,
                }),
            })
            .await
            .map_err(|error| anyhow!("Failed to submit startup tokenizer job: {error}"))?;
        info!("Startup tokenizer job submitted (will complete in background)");
    }

    info!(
        "Initializing workers for routing mode: {:?}",
        config.routing.mode
    );
    job_queue
        .submit(Job::InitializeWorkersFromConfig {
            gateway_config: Box::new(config.clone()),
        })
        .await
        .map_err(|error| anyhow!("Failed to submit worker initialization job: {error}"))?;
    info!("Worker initialization job submitted (will complete in background)");

    if let Some(mcp_config) = &config.extensions.mcp_config {
        info!("Found {} MCP server(s) in config", mcp_config.servers.len());
        job_queue
            .submit(Job::InitializeMcpServers {
                mcp_config: Box::new(mcp_config.clone()),
            })
            .await
            .map_err(|error| anyhow!("Failed to submit MCP initialization job: {error}"))?;
    } else {
        info!("No MCP config provided, skipping MCP server initialization");
    }

    Ok(())
}

/// Builds the router implementation for the resolved gateway configuration.
async fn create_router_manager(
    app_context: &Arc<AppContext>,
    config: &GatewayConfig,
) -> Result<Arc<RouterManager>> {
    let router_manager = RouterManager::from_config(config, app_context)
        .await
        .map_err(Error::msg)
        .context("create router manager")?;

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );
    Ok(router_manager)
}

/// Starts periodic worker health probes unless they are disabled.
fn start_health_checker(
    app_context: &Arc<AppContext>,
    config: &GatewayConfig,
) -> Option<crate::core::worker::HealthChecker> {
    if config.workers.health_check.disable_health_check {
        return None;
    }

    let health_checker = app_context
        .worker_registry
        .start_health_checker(config.workers.health_check.check_interval_secs);
    debug!(
        "Started health checker for workers with {}s interval",
        config.workers.health_check.check_interval_secs
    );
    Some(health_checker)
}

/// Starts the optional queue used when concurrent-request limiting is enabled.
fn start_concurrency_queue(
    app_context: &Arc<AppContext>,
    config: &GatewayConfig,
) -> (
    Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    Option<JoinHandle<()>>,
) {
    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.routing.queue_size,
        Duration::from_secs(config.routing.queue_timeout_secs),
    );
    if app_context.rate_limiter.is_none() {
        info!("Rate limiting is disabled (max_concurrent_requests = -1)");
    }

    match processor {
        Some(processor) => {
            let processor = spawn(processor.run());
            debug!(
                "Started request queue (size: {}, timeout: {}s)",
                config.routing.queue_size, config.routing.queue_timeout_secs
            );
            (limiter.queue_tx.clone(), Some(processor))
        }
        None => {
            debug!(
                "Rate limiting enabled (max_concurrent_requests = {}, queue disabled)",
                config.routing.max_concurrent_requests
            );
            (limiter.queue_tx.clone(), None)
        }
    }
}

/// Shares mesh synchronization with registries that replicate worker and policy state.
fn configure_mesh_sync(app_context: &Arc<AppContext>, sync_manager: Option<&Arc<MeshSyncManager>>) {
    let Some(sync_manager) = sync_manager else {
        return;
    };

    app_context
        .worker_registry
        .set_mesh_sync(Some(sync_manager.clone()));
    app_context
        .policy_registry
        .set_mesh_sync(Some(sync_manager.clone()));
    info!("Mesh sync manager set on worker and policy registries");
}

/// Combines routing and component handles into the state injected into HTTP handlers.
fn build_app_state(
    app_context: Arc<AppContext>,
    router_manager: Arc<RouterManager>,
    concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    mesh_handler: Option<Arc<MeshServerHandler>>,
    mesh_sync_manager: Option<Arc<MeshSyncManager>>,
) -> Arc<AppState> {
    let router: Arc<dyn RouterTrait> = router_manager.clone();
    Arc::new(AppState {
        router,
        context: app_context,
        concurrency_queue_tx,
        router_manager: Some(router_manager),
        mesh_handler,
        mesh_sync_manager,
    })
}

/// Starts Kubernetes discovery when discovery configuration is present.
async fn start_service_discovery_if_enabled(config: &GatewayConfig, app_state: &Arc<AppState>) {
    let Some(discovery) = config.discovery.clone() else {
        return;
    };

    let service_discovery_config = ServiceDiscoveryConfig {
        enabled: true,
        selector: discovery.selector,
        check_interval: Duration::from_secs(discovery.check_interval_secs),
        port: discovery.port,
        namespace: discovery.namespace,
        pd_mode: config.routing.is_pd_mode(),
        prefill_selector: discovery.prefill_selector,
        decode_selector: discovery.decode_selector,
        bootstrap_port_annotation: discovery.bootstrap_port_annotation,
        router_selector: discovery.router_selector,
        router_mesh_port_annotation: discovery.router_mesh_port_annotation,
        igw_mode: config.routing.enable_igw,
    };
    let mesh_cluster_state = app_state
        .mesh_handler
        .as_ref()
        .map(|handler| handler.state.clone());
    let mesh_port = config.mesh.as_ref().map(|mesh| mesh.self_addr.port());

    match start_service_discovery(
        service_discovery_config,
        Arc::clone(&app_state.context),
        mesh_cluster_state,
        mesh_port,
    )
    .await
    {
        Ok(handle) => {
            info!("Service discovery started");
            spawn(async move {
                if let Err(error) = handle.await {
                    error!("Service discovery task failed: {error:?}");
                }
            });
        }
        Err(error) => {
            error!("Failed to start service discovery: {error}");
            warn!("Continuing without service discovery");
        }
    }
}

/// Builds the Axum application with authentication and request middleware.
async fn build_http_app(config: &GatewayConfig, app_state: Arc<AppState>) -> Router {
    info!(
        "Router ready | workers: {:?}",
        WorkerManager::get_worker_urls(&app_state.context.worker_registry)
    );

    let request_id_headers = config.server.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });
    let auth_config = AuthConfig {
        api_key: config.security.api_key.clone(),
    };
    let control_plane_auth_state =
        crate::auth::ControlPlaneAuthState::try_init(config.security.control_plane_auth.as_ref())
            .await;

    build_app(
        app_state,
        auth_config,
        control_plane_auth_state,
        config.server.max_payload_size,
        request_id_headers,
        config.server.cors_allowed_origins.clone(),
    )
}

/// Binds and serves HTTP or HTTPS until a shutdown signal is received.
async fn serve_http(app: Router, config: &GatewayConfig) -> Result<()> {
    let bind_addr = format!("{}:{}", config.server.host, config.server.port);
    info!("Starting server on {bind_addr}");

    let addr: std::net::SocketAddr = bind_addr
        .parse()
        .map_err(|error| anyhow!("Invalid address: {error}"))?;
    let handle = axum_server::Handle::new();

    // graceful shutdown for server
    let shutdown_handle = handle.clone();
    let grace_period = Duration::from_secs(config.server.shutdown_grace_period_secs);
    spawn(async move {
        shutdown_signal().await;
        shutdown_handle.graceful_shutdown(Some(grace_period));
    });

    // https server
    if let Some(tls) = &config.security.server_tls {
        info!("TLS enabled");
        ring::default_provider()
            .install_default()
            .map_err(|error| anyhow!("Failed to install rustls ring provider: {error:?}"))?;
        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(
            tls.certificate.clone(),
            tls.private_key.clone(),
        )
        .await
        .map_err(|error| anyhow!("Failed to create TLS config: {error}"))?;

        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .context("run TLS server")?;
    } else {
        axum_server::bind(addr)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .context("run HTTP server")?;
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

/// Initializes OpenTelemetry and the global tracing subscriber.
pub fn init_tracing(config: &ObservabilityConfig) -> Result<LogGuard> {
    // init OTel first
    otel_trace::otel_tracing_init(config.enable_trace, Some(&config.otlp_traces_endpoint))
        .context("initialize OpenTelemetry")?;

    // initilzer logging
    match logging::init_logging(LoggingConfig::from_config(config), Some(config)) {
        Ok(guard) => Ok(guard),
        Err(error) => {
            if otel_trace::is_otel_enabled() {
                otel_trace::shutdown_otel();
            }
            return Err(error).context("initialize logging");
        }
    }
}

/// Flushes tracing data and releases logging resources.
pub async fn shutdown_tracing(log_guard: LogGuard) -> Result<()> {
    let enabled = otel_trace::is_otel_enabled();

    let flush_result = if enabled {
        otel_trace::flush_spans_async()
            .await
            .context("flush OpenTelemetry spans")
    } else {
        Ok(())
    };
    if enabled {
        otel_trace::shutdown_otel();
    }

    drop(log_guard);
    flush_result
}
