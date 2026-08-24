use std::{sync::Arc, time::Duration};

use axum::{
    extract::{Path, Query, Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use smg_mesh::{RateLimitConfig, GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY};
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    app_state::AppState,
    config::RoutingMode,
    core::{
        job_queue::Job,
        steps::{WasmModuleConfigRequest, WasmModuleRemovalRequest},
        worker::WorkerType,
        worker_manager::WorkerManager,
    },
    middleware::{self, AuthConfig},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
        rerank::V1RerankReqInput,
        responses::{ResponsesGetParams, ResponsesRequest},
        tokenize::{AddTokenizerRequest, DetokenizeRequest, TokenizeRequest},
        validated::ValidatedJson,
        worker_spec::{WorkerConfigRequest, WorkerUpdateRequest},
    },
    routers::{conversations, parse, tokenize},
    wasm::{
        WasmMetrics, WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAddResult,
        WasmModuleListResponse,
    },
};

pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    control_plane_auth_state: Option<crate::auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Router {
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/classify", post(v1_classify))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input_items",
            get(v1_responses_list_input_items),
        )
        .route("/v1/conversations", post(v1_conversations_create))
        .route(
            "/v1/conversations/{conversation_id}",
            get(v1_conversations_get)
                .post(v1_conversations_update)
                .delete(v1_conversations_delete),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            get(v1_conversations_list_items).post(v1_conversations_create_items),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(v1_conversations_get_item).delete(v1_conversations_delete_item),
        )
        // Tokenize / Detokenize endpoints
        .route("/v1/tokenize", post(v1_tokenize))
        .route("/v1/detokenize", post(v1_detokenize))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::wasm_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/model_info", get(get_model_info))
        // TODO: Remove `/get_model_info` alias after one release-cycle deprecation window.
        .route("/get_model_info", get(get_model_info))
        .route("/server_info", get(get_server_info))
        // TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
        .route("/get_server_info", get(get_server_info));

    // Build admin routes with control plane auth if configured, otherwise use simple API key auth
    let admin_routes = Router::new()
        .route("/flush_cache", post(flush_cache))
        .route("/v1/loads", get(get_loads))
        // TODO: Remove `/get_loads` alias after one release-cycle deprecation window.
        .route("/get_loads", get(get_loads))
        .route("/parse/function_call", post(parse_function_call_http))
        .route("/parse/reasoning", post(parse_reasoning_http))
        .route("/wasm", post(add_wasm_module))
        .route("/wasm/{module_uuid}", delete(remove_wasm_module))
        .route("/wasm", get(list_wasm_modules))
        // Tokenizer management endpoints
        .route(
            "/v1/tokenizers",
            post(v1_tokenizers_add).get(v1_tokenizers_list),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}",
            get(v1_tokenizers_get).delete(v1_tokenizers_remove),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}/status",
            get(v1_tokenizers_status),
        );

    // Build worker routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker).get(list_workers_rest))
        .route(
            "/workers/{worker_id}",
            get(get_worker).put(update_worker).delete(delete_worker),
        );

    // Apply authentication middleware to control plane routes
    let apply_control_plane_auth = |routes: Router<Arc<AppState>>| {
        if let Some(ref cp_state) = control_plane_auth_state {
            routes.route_layer(axum::middleware::from_fn_with_state(
                cp_state.clone(),
                crate::auth::control_plane_auth_middleware,
            ))
        } else {
            routes.route_layer(axum::middleware::from_fn_with_state(
                auth_config.clone(),
                middleware::auth_middleware,
            ))
        }
    };
    let admin_routes = apply_control_plane_auth(admin_routes);
    let worker_routes = apply_control_plane_auth(worker_routes);

    // HA management routes
    let mesh_routes = Router::new()
        .route("/ha/status", get(get_cluster_status))
        .route("/ha/health", get(get_mesh_health))
        .route("/ha/workers", get(get_worker_states))
        .route("/ha/workers/{worker_id}", get(get_worker_state))
        .route("/ha/policies", get(get_policy_states))
        .route("/ha/policies/{model_id}", get(get_policy_state))
        .route("/ha/config/{key}", get(get_app_config))
        .route("/ha/config", post(update_app_config))
        .route("/ha/rate-limit", post(set_global_rate_limit))
        .route("/ha/rate-limit", get(get_global_rate_limit))
        .route("/ha/rate-limit/stats", get(get_global_rate_limit_stats))
        .route("/ha/shutdown", post(trigger_graceful_shutdown))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .merge(mesh_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state)
}

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([http::Method::GET, http::Method::POST, http::Method::OPTIONS])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> Response {
    let model_id = body.model.as_deref();
    state
        .router
        .route_generate(Some(&headers), &body, model_id)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    state
        .router
        .route_chat(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    let rerank_body = &body.into();
    state
        .router
        .route_rerank(Some(&headers), rerank_body, Some(&rerank_body.model))
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    state
        .router
        .route_responses(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    state
        .router
        .route_classify(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
    Query(params): Query<ResponsesGetParams>,
) -> Response {
    state
        .router
        .get_response(Some(&headers), &response_id, &params)
        .await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .delete_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .list_response_input_items(Some(&headers), &response_id)
        .await
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers: Vec<_> = workers
        .iter()
        .filter(|worker| worker.is_healthy())
        .collect();

    let is_ready = if state.context.gateway_config.routing.enable_igw {
        !healthy_workers.is_empty()
    } else {
        match &state.context.gateway_config.routing.mode {
            RoutingMode::PrefillDecode { .. } => {
                let has_prefill = healthy_workers
                    .iter()
                    .any(|worker| matches!(worker.worker_type(), WorkerType::Prefill { .. }));
                let has_decode = healthy_workers
                    .iter()
                    .any(|worker| matches!(worker.worker_type(), WorkerType::Decode));
                has_prefill && has_decode
            }
            RoutingMode::Regular { .. } | RoutingMode::OpenAI { .. } => !healthy_workers.is_empty(),
        }
    };

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "insufficient healthy workers"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    WorkerManager::get_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::get_all_worker_loads(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    match state.context.worker_service.create_worker(config).await {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    state.context.worker_service.list_workers().into_response()
}

async fn get_worker(State(state): State<Arc<AppState>>, Path(worker_id): Path<String>) -> Response {
    match state.context.worker_service.get_worker(&worker_id) {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Response {
    match state.context.worker_service.delete_worker(&worker_id).await {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn update_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
    Json(update): Json<WorkerUpdateRequest>,
) -> Response {
    match state
        .context
        .worker_service
        .update_worker(&worker_id, update)
        .await
    {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn v1_conversations_create(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation(&state.context.conversation_storage, body).await
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::get_conversation(&state.context.conversation_storage, &conversation_id).await
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::update_conversation(&state.context.conversation_storage, &conversation_id, body)
        .await
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::delete_conversation(&state.context.conversation_storage, &conversation_id).await
}

#[derive(Deserialize, Default)]
struct ListItemsQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

async fn v1_conversations_list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(ListItemsQuery {
        limit,
        order,
        after,
    }): Query<ListItemsQuery>,
) -> Response {
    conversations::list_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        limit,
        order.as_deref(),
        after.as_deref(),
    )
    .await
}

#[derive(Deserialize, Default)]
struct GetItemQuery {
    include: Option<Vec<String>>,
}

async fn v1_conversations_create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        body,
    )
    .await
}

async fn v1_conversations_get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
) -> Response {
    conversations::get_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
        query.include,
    )
    .await
}

async fn v1_conversations_delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Response {
    conversations::delete_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
    )
    .await
}

async fn parse_function_call_http(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ParseFunctionCallRequest>,
) -> Response {
    parse::parse_function_call(&state.context, &request).await
}

async fn parse_reasoning_http(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SeparateReasoningRequest>,
) -> Response {
    parse::parse_reasoning(&state.context, &request).await
}

async fn v1_tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    tokenize::tokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    tokenize::detokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_tokenizers_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddTokenizerRequest>,
) -> Response {
    tokenize::add_tokenizer(&state.context, request).await
}

async fn v1_tokenizers_list(State(state): State<Arc<AppState>>) -> Response {
    tokenize::list_tokenizers(&state.context.tokenizer_registry).await
}

async fn v1_tokenizers_get(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_info(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_status(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_status(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_remove(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::remove_tokenizer(&state.context, &tokenizer_id).await
}

#[derive(Debug, Serialize, Deserialize)]
struct ClusterStatusResponse {
    node_name: String,
    node_count: usize,
    nodes: Vec<NodeInfo>,
    stores: StoreStatus,
}

#[derive(Debug, Serialize, Deserialize)]
struct NodeInfo {
    name: String,
    address: String,
    status: String,
    version: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoreStatus {
    membership_count: usize,
    worker_count: usize,
    policy_count: usize,
    app_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct MeshHealthResponse {
    status: String,
    node_name: String,
    cluster_size: usize,
    stores_healthy: bool,
}

async fn get_cluster_status(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    let state = handler.state.read();
    let nodes: Vec<NodeInfo> = state
        .values()
        .map(|node| NodeInfo {
            name: node.name.clone(),
            address: node.address.clone(),
            status: format!("{:?}", node.status),
            version: node.version,
        })
        .collect();
    let stores = StoreStatus {
        membership_count: state.len(),
        worker_count: 0,
        policy_count: 0,
        app_count: 0,
    };

    (
        StatusCode::OK,
        Json(ClusterStatusResponse {
            node_name: handler.self_name.clone(),
            node_count: nodes.len(),
            nodes,
            stores,
        }),
    )
        .into_response()
}

async fn get_mesh_health(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    let cluster_size = handler.state.read().len();
    (
        StatusCode::OK,
        Json(MeshHealthResponse {
            status: "healthy".to_string(),
            node_name: handler.self_name.clone(),
            cluster_size,
            stores_healthy: true,
        }),
    )
        .into_response()
}

async fn get_worker_states(State(app_state): State<Arc<AppState>>) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => (StatusCode::OK, Json(manager.get_all_worker_states())).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

async fn get_policy_states(State(app_state): State<Arc<AppState>>) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => (StatusCode::OK, Json(manager.get_all_policy_states())).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

async fn get_worker_state(
    Path(worker_id): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => match manager.get_worker_state(&worker_id) {
            Some(state) => (StatusCode::OK, Json(state)).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Worker not found"})),
            )
                .into_response(),
        },
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

async fn get_policy_state(
    Path(model_id): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => match manager.get_policy_state(&model_id) {
            Some(state) => (StatusCode::OK, Json(state)).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Policy not found"})),
            )
                .into_response(),
        },
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct UpdateAppConfigRequest {
    key: String,
    value: String,
}

async fn update_app_config(
    State(app_state): State<Arc<AppState>>,
    Json(request): Json<UpdateAppConfigRequest>,
) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    let value = if request.value.len() % 2 == 0 {
        match (0..request.value.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&request.value[index..index + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
        {
            Ok(value) => value,
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid hex encoding"})),
                )
                    .into_response();
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Hex string must have even length"})),
        )
            .into_response();
    };

    handler.write_data(request.key.clone(), value);
    info!(key = %request.key, "Updated mesh app config");
    (
        StatusCode::OK,
        Json(json!({"status": "updated", "key": request.key})),
    )
        .into_response()
}

async fn get_app_config(
    Path(key): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    match handler.read_data(key.clone()) {
        Some(value) => {
            let hex_value: String = value.iter().map(|byte| format!("{byte:02x}")).collect();
            (
                StatusCode::OK,
                Json(json!({"key": key, "value": hex_value, "format": "hex"})),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Config not found"})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct SetRateLimitRequest {
    limit_per_second: u64,
}

async fn set_global_rate_limit(
    State(app_state): State<Arc<AppState>>,
    Json(request): Json<SetRateLimitRequest>,
) -> Response {
    let config = RateLimitConfig {
        limit_per_second: request.limit_per_second,
    };
    let config_bytes = match serde_json::to_vec(&config) {
        Ok(config_bytes) => config_bytes,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to serialize rate limit config"})),
            )
                .into_response();
        }
    };
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    handler.write_data(GLOBAL_RATE_LIMIT_KEY.to_string(), config_bytes);
    info!(
        limit_per_second = request.limit_per_second,
        "Set global rate limit"
    );
    (
        StatusCode::OK,
        Json(json!({
            "status": "updated",
            "limit_per_second": request.limit_per_second
        })),
    )
        .into_response()
}

async fn get_global_rate_limit(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    match handler.read_data(GLOBAL_RATE_LIMIT_KEY.to_string()) {
        Some(value) => match serde_json::from_slice::<RateLimitConfig>(&value) {
            Ok(config) => (
                StatusCode::OK,
                Json(json!({"limit_per_second": config.limit_per_second})),
            )
                .into_response(),
            Err(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to deserialize rate limit config"})),
            )
                .into_response(),
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Global rate limit not configured"})),
        )
            .into_response(),
    }
}

async fn get_global_rate_limit_stats(State(app_state): State<Arc<AppState>>) -> Response {
    let sync_manager = match &app_state.mesh_sync_manager {
        Some(manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh sync manager not available"})),
            )
                .into_response();
        }
    };
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    let config = handler
        .read_data(GLOBAL_RATE_LIMIT_KEY.to_string())
        .and_then(|value| serde_json::from_slice::<RateLimitConfig>(&value).ok())
        .unwrap_or_default();
    let current_count = sync_manager
        .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
        .unwrap_or(0);

    (
        StatusCode::OK,
        Json(json!({
            "limit_per_second": config.limit_per_second,
            "current_count": current_count,
            "remaining": if config.limit_per_second > 0 {
                (config.limit_per_second as i64).saturating_sub(current_count).max(0)
            } else {
                -1
            }
        })),
    )
        .into_response()
}

async fn trigger_graceful_shutdown(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(handler) => handler.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    info!("Graceful shutdown triggered via API");
    tokio::spawn(async move {
        if let Err(error) = handler.graceful_shutdown().await {
            warn!(%error, "Mesh graceful shutdown failed");
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({"status": "shutdown initiated"})),
    )
        .into_response()
}

async fn wait_for_wasm_job_completion(
    job_queue: &crate::core::job_queue::JobQueue,
    status_key: &str,
    timeout: Duration,
) -> Result<(), String> {
    let start = std::time::Instant::now();
    let mut poll_interval = Duration::from_millis(100);
    let max_poll_interval = Duration::from_secs(2);
    let poll_backoff = Duration::from_millis(200);

    loop {
        if start.elapsed() > timeout {
            return Err(format!("Job timeout after {}s", timeout.as_secs()));
        }

        match job_queue.get_status(status_key) {
            Some(job_status) => match job_status.status.as_str() {
                "pending" | "processing" => {
                    tokio::time::sleep(poll_interval).await;
                    poll_interval = (poll_interval + poll_backoff).min(max_poll_interval);
                }
                "failed" => {
                    let error = job_status
                        .message
                        .unwrap_or_else(|| "Unknown error".to_string());
                    job_queue.remove_status(status_key);
                    return Err(error);
                }
                _ => {
                    job_queue.remove_status(status_key);
                    return Err("Unexpected job status".to_string());
                }
            },
            // Successful jobs remove their status record during completion.
            None => return Ok(()),
        }
    }
}

async fn add_wasm_module(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WasmModuleAddRequest>,
) -> Response {
    if state.context.wasm_manager.is_none() || state.context.worker_job_queue.get().is_none() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    let job_queue = state.context.worker_job_queue.get().expect("checked above");
    let mut status = StatusCode::OK;
    let mut modules = config.modules.clone();

    for module in &mut modules {
        let job = Job::AddWasmModule {
            config: Box::new(WasmModuleConfigRequest {
                descriptor: module.clone(),
            }),
        };
        let worker_url = job.worker_url().to_string();

        match job_queue.submit(job).await {
            Ok(()) => {
                match wait_for_wasm_job_completion(job_queue, &worker_url, Duration::from_secs(300))
                    .await
                {
                    Ok(()) => {
                        let result = state
                            .context
                            .wasm_manager
                            .as_ref()
                            .and_then(|manager| manager.get_modules().ok())
                            .and_then(|modules| {
                                modules
                                    .iter()
                                    .find(|registered| registered.module_meta.name == module.name)
                                    .map(|registered| registered.module_uuid)
                            });
                        match result {
                            Some(module_uuid) => {
                                module.add_result = Some(WasmModuleAddResult::Success(module_uuid));
                            }
                            None => {
                                module.add_result = Some(WasmModuleAddResult::Error(
                                    "Module registered but UUID not found".to_string(),
                                ));
                                status = StatusCode::BAD_REQUEST;
                            }
                        }
                    }
                    Err(error) => {
                        module.add_result = Some(WasmModuleAddResult::Error(error));
                        status = StatusCode::BAD_REQUEST;
                    }
                }
            }
            Err(error) => {
                module.add_result = Some(WasmModuleAddResult::Error(format!(
                    "Failed to submit job: {error}"
                )));
                status = StatusCode::BAD_REQUEST;
            }
        }
    }

    (status, Json(WasmModuleAddResponse { modules })).into_response()
}

async fn remove_wasm_module(
    State(state): State<Arc<AppState>>,
    Path(module_uuid): Path<String>,
) -> Response {
    let Ok(module_uuid) = Uuid::parse_str(&module_uuid) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if state.context.wasm_manager.is_none() || state.context.worker_job_queue.get().is_none() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    let job_queue = state.context.worker_job_queue.get().expect("checked above");
    let job = Job::RemoveWasmModule {
        request: Box::new(WasmModuleRemovalRequest::new(module_uuid)),
    };
    let worker_url = job.worker_url().to_string();

    match job_queue.submit(job).await {
        Ok(()) => {
            match wait_for_wasm_job_completion(job_queue, &worker_url, Duration::from_secs(60))
                .await
            {
                Ok(()) => (StatusCode::OK, "Module removed successfully").into_response(),
                Err(error) => (StatusCode::BAD_REQUEST, error).into_response(),
            }
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to submit job: {error}"),
        )
            .into_response(),
    }
}

async fn list_wasm_modules(State(state): State<Arc<AppState>>) -> Response {
    let Some(wasm_manager) = state.context.wasm_manager.as_ref() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    match wasm_manager.get_modules() {
        Ok(modules) => {
            let (total, successful, failed, total_time_ms, max_time_ms) =
                wasm_manager.get_metrics();
            let average_execution_time_ms =
                (total > 0).then_some(total_time_ms as f64 / total as f64);
            (
                StatusCode::OK,
                Json(WasmModuleListResponse {
                    modules,
                    metrics: WasmMetrics {
                        total_executions: total,
                        successful_executions: successful,
                        failed_executions: failed,
                        total_execution_time_ms: total_time_ms,
                        max_execution_time_ms: max_time_ms,
                        average_execution_time_ms,
                    },
                }),
            )
                .into_response()
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
