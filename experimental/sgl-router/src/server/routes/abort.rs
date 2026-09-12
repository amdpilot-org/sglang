// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Request-abortion admin endpoint.

use crate::server::app_context::AppContext;
use crate::server::header_utils::should_forward_request_header;
use crate::workers::worker::Worker;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

/// Bound the number of simultaneous abort calls for large worker fleets.
const MAX_CONCURRENT_ABORTS: usize = 32;

#[derive(Serialize)]
pub struct FailedWorker {
    pub worker: String,
    pub error: String,
}

/// Per-worker result for an abort fan-out. The registry is snapshotted before
/// dispatch, so every worker counted in `total_workers` appears exactly once
/// in either `successful` or `failed`.
#[derive(Serialize)]
pub struct AbortRequestResult {
    pub successful: Vec<String>,
    pub failed: Vec<FailedWorker>,
    pub total_workers: usize,
    pub message: String,
}

impl AbortRequestResult {
    fn from_outcomes(
        total_workers: usize,
        successful: Vec<String>,
        failed: Vec<FailedWorker>,
    ) -> Self {
        let message = if total_workers == 0 {
            "No workers registered; nothing to abort".to_string()
        } else if failed.is_empty() {
            format!("Abort request reached all {total_workers} workers")
        } else {
            format!(
                "Abort request: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };
        Self {
            successful,
            failed,
            total_workers,
            message,
        }
    }
}

/// Fan SGLang's `POST /abort_request` out to every registered worker.
///
/// The router cannot know which worker currently owns a request ID, while an
/// abort for an unknown ID is a safe no-op on an SGLang worker. Broadcasting
/// therefore covers ordinary, prefill, and decode workers without maintaining
/// a stale-prone request ownership table. This administrative call bypasses
/// circuit breakers and does not alter their routing-health state.
///
/// The body is intentionally forwarded byte-for-byte. Besides preserving the
/// current `rid` and `abort_all` fields, this keeps the router compatible with
/// newer worker-side AbortReq fields without requiring a lockstep rollout.
pub async fn abort_request(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let workers = ctx.registry.all();
    let total_workers = workers.len();

    if workers.is_empty() {
        tracing::warn!("abort_request called but no workers are registered");
        return (
            StatusCode::OK,
            Json(AbortRequestResult::from_outcomes(0, Vec::new(), Vec::new())),
        )
            .into_response();
    }

    let (successful, failed) = fan_out_abort(
        &workers,
        &ctx.proxy.client,
        ctx.proxy.request_timeout,
        &headers,
        body,
    )
    .await;

    if failed.is_empty() {
        tracing::info!(total_workers, "abort_request: all workers reached");
    } else {
        tracing::warn!(
            total_workers,
            succeeded = successful.len(),
            failed = failed.len(),
            "abort_request: some workers failed",
        );
    }

    let status = if failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::BAD_GATEWAY
    };
    (
        status,
        Json(AbortRequestResult::from_outcomes(
            total_workers,
            successful,
            failed,
        )),
    )
        .into_response()
}

async fn fan_out_abort(
    workers: &[Arc<Worker>],
    client: &Client,
    timeout: Duration,
    headers: &HeaderMap,
    body: Bytes,
) -> (Vec<String>, Vec<FailedWorker>) {
    let urls: Vec<String> = workers.iter().map(|worker| worker.url.clone()).collect();
    let forwarded_headers: Vec<_> = headers
        .iter()
        .filter(|(name, _)| should_forward_request_header(name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();

    let outcomes = stream::iter(urls)
        .map(|url| {
            let client = client.clone();
            let body = body.clone();
            let forwarded_headers = forwarded_headers.clone();
            async move {
                let abort_url = format!("{}/abort_request", url.trim_end_matches('/'));
                let mut request = client.post(&abort_url).body(body).timeout(timeout);
                for (name, value) in forwarded_headers {
                    request = request.header(name, value);
                }
                let result = request
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .send()
                    .await;
                (url, result)
            }
        })
        .buffer_unordered(MAX_CONCURRENT_ABORTS)
        .collect::<Vec<_>>()
        .await;

    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for (url, result) in outcomes {
        match result {
            Ok(response) if response.status().is_success() => successful.push(url),
            Ok(response) => failed.push(FailedWorker {
                worker: url,
                error: format!("HTTP {}", response.status()),
            }),
            Err(error) => failed.push(FailedWorker {
                worker: url,
                error: format!("{:#}", anyhow::Error::new(error)),
            }),
        }
    }
    (successful, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::server::app_context::AppContext;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio::sync::{mpsc, oneshot};
    use tower::ServiceExt;

    async fn spawn_status_worker(status: StatusCode) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route("/abort_request", post(move || async move { status }));
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    async fn spawn_recording_worker() -> (
        String,
        oneshot::Sender<()>,
        mpsc::UnboundedReceiver<(HeaderMap, Bytes)>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (record_tx, record_rx) = mpsc::unbounded_channel();
        let app = Router::new().route(
            "/abort_request",
            post(move |headers: HeaderMap, body: Bytes| {
                let record_tx = record_tx.clone();
                async move {
                    let _ = record_tx.send((headers, body));
                    StatusCode::OK
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), shutdown_tx, record_rx)
    }

    fn unused_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    }

    fn add_worker(ctx: &AppContext, id: &str, url: &str, mode: WorkerMode) {
        let bootstrap_port = (mode == WorkerMode::Prefill).then_some(8998);
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: url.into(),
                mode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port,
            })
            .expect("worker accepted");
    }

    fn ctx_with_workers(urls: &[&str]) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        for (index, url) in urls.iter().enumerate() {
            add_worker(&ctx, &format!("w-{index}"), url, WorkerMode::Plain);
        }
        Arc::new(ctx)
    }

    async fn post_abort(
        ctx: Arc<AppContext>,
        body: &'static str,
        authorization: Option<&str>,
    ) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let mut request = Request::builder()
            .method("POST")
            .uri("/abort_request")
            .header("content-type", "application/json");
        if let Some(value) = authorization {
            request = request.header("authorization", value);
        }
        let response = app
            .oneshot(request.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn all_workers_succeed() {
        let (u1, _s1) = spawn_status_worker(StatusCode::OK).await;
        let (u2, _s2) = spawn_status_worker(StatusCode::NO_CONTENT).await;
        let (status, body) = post_abort(
            ctx_with_workers(&[&u1, &u2]),
            r#"{"rid":"request-1"}"#,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 2);
        assert_eq!(body["successful"].as_array().unwrap().len(), 2);
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn preserves_body_and_forwardable_headers() {
        let (url, _shutdown, mut records) = spawn_recording_worker().await;
        let payload = r#"{"rid":"request-1","abort_all":false,"future_field":{"x":1}}"#;
        let (status, _) =
            post_abort(ctx_with_workers(&[&url]), payload, Some("Bearer secret")).await;
        assert_eq!(status, StatusCode::OK);

        let (headers, body) = records.recv().await.unwrap();
        assert_eq!(body, Bytes::from_static(payload.as_bytes()));
        assert_eq!(headers["content-type"], "application/json");
        assert_eq!(headers["authorization"], "Bearer secret");
    }

    #[tokio::test]
    async fn opaque_malformed_body_is_fanned_out_for_worker_validation() {
        let (url, _shutdown, mut records) = spawn_recording_worker().await;
        let payload = "not-json";
        let (status, _) = post_abort(ctx_with_workers(&[&url]), payload, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(records.recv().await.unwrap().1, payload);
    }

    #[tokio::test]
    async fn partial_failure_returns_bad_gateway_with_breakdown() {
        let (ok_url, _s1) = spawn_status_worker(StatusCode::OK).await;
        let (error_url, _s2) = spawn_status_worker(StatusCode::UNPROCESSABLE_ENTITY).await;
        let (status, body) = post_abort(
            ctx_with_workers(&[&ok_url, &error_url]),
            r#"{"rid":"request-1"}"#,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["successful"], serde_json::json!([ok_url]));
        assert_eq!(body["failed"][0]["worker"], error_url);
        assert!(body["failed"][0]["error"].as_str().unwrap().contains("422"));
    }

    #[tokio::test]
    async fn unreachable_worker_is_reported() {
        let url = format!("http://127.0.0.1:{}", unused_port());
        let (status, body) =
            post_abort(ctx_with_workers(&[&url]), r#"{"abort_all":true}"#, None).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["failed"][0]["worker"], url);
    }

    #[tokio::test]
    async fn empty_registry_is_a_successful_noop() {
        let (status, body) =
            post_abort(Arc::new(AppContext::stub()), r#"{"rid":"unknown"}"#, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 0);
    }

    #[tokio::test]
    async fn trailing_slash_worker_url_is_normalized() {
        let (url, _shutdown) = spawn_status_worker(StatusCode::OK).await;
        let url = format!("{url}/");
        let (status, body) =
            post_abort(ctx_with_workers(&[&url]), r#"{"rid":"request-1"}"#, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["successful"], serde_json::json!([url]));
    }

    #[tokio::test]
    async fn reaches_prefill_and_decode_workers() {
        let (prefill, _s1) = spawn_status_worker(StatusCode::OK).await;
        let (decode, _s2) = spawn_status_worker(StatusCode::OK).await;
        let ctx = AppContext::stub();
        add_worker(&ctx, "prefill", &prefill, WorkerMode::Prefill);
        add_worker(&ctx, "decode", &decode, WorkerMode::Decode);

        let (status, body) = post_abort(Arc::new(ctx), r#"{"rid":"request-1"}"#, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 2);
        assert_eq!(body["successful"].as_array().unwrap().len(), 2);
    }
}
