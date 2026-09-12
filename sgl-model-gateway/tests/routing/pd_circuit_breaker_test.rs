//! Regression coverage for PD circuit-breaker dispatch (issue #31206).

use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::{
    config::{CircuitBreakerConfig, RetryConfig, RouterConfig},
    core::{CircuitState, Worker},
};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestWorkerConfig};

fn pd_config(port: u16, prefill_port: u16, decode_port: u16) -> RouterConfig {
    RouterConfig::builder()
        .prefill_decode_mode(
            vec![(format!("http://127.0.0.1:{prefill_port}"), None)],
            vec![format!("http://127.0.0.1:{decode_port}")],
        )
        .round_robin_policy()
        .host("127.0.0.1")
        .port(port)
        .request_timeout_secs(30)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .retry_config(RetryConfig {
            max_retries: 0,
            ..Default::default()
        })
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_duration_secs: 1,
            window_duration_secs: 60,
        })
        .build_unchecked()
}

async fn generate_status(app: &axum::Router) -> StatusCode {
    let request = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"text": "hello", "stream": false})).unwrap(),
        ))
        .unwrap();

    tokio::time::timeout(Duration::from_secs(3), app.clone().oneshot(request))
        .await
        .expect("PD request must fail fast rather than wait for missing KV")
        .unwrap()
        .status()
}

fn only_worker(workers: Vec<std::sync::Arc<dyn Worker>>, kind: &str) -> std::sync::Arc<dyn Worker> {
    assert_eq!(workers.len(), 1, "test must have exactly one {kind} worker");
    workers.into_iter().next().unwrap()
}

#[tokio::test]
async fn open_prefill_breaker_fails_whole_request_fast() {
    let ctx = AppTestContext::new_with_config(
        pd_config(4380, 20880, 20881),
        vec![
            TestWorkerConfig::prefill(20880),
            TestWorkerConfig::decode(20881),
        ],
    )
    .await;
    let app = ctx.create_app().await;
    assert_eq!(generate_status(&app).await, StatusCode::OK);

    let prefill = only_worker(
        ctx.app_context.worker_registry.get_prefill_workers(),
        "prefill",
    );
    assert!(prefill.is_healthy());
    prefill.circuit_breaker().force_open();

    assert_eq!(generate_status(&app).await, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(prefill.circuit_breaker().state(), CircuitState::Open);

    ctx.shutdown().await;
}

#[tokio::test]
async fn recovered_prefill_half_open_probe_closes_breaker() {
    let ctx = AppTestContext::new_with_config(
        pd_config(4381, 20882, 20883),
        vec![
            TestWorkerConfig::prefill(20882),
            TestWorkerConfig::decode(20883),
        ],
    )
    .await;
    let app = ctx.create_app().await;
    let prefill = only_worker(
        ctx.app_context.worker_registry.get_prefill_workers(),
        "prefill",
    );
    prefill.circuit_breaker().force_open();

    assert_eq!(generate_status(&app).await, StatusCode::SERVICE_UNAVAILABLE);
    tokio::time::sleep(Duration::from_millis(1100)).await;
    assert_eq!(prefill.circuit_breaker().state(), CircuitState::HalfOpen);

    assert_eq!(generate_status(&app).await, StatusCode::OK);
    assert_eq!(prefill.circuit_breaker().state(), CircuitState::HalfOpen);
    assert_eq!(generate_status(&app).await, StatusCode::OK);
    assert_eq!(prefill.circuit_breaker().state(), CircuitState::Closed);

    ctx.shutdown().await;
}

#[tokio::test]
async fn open_decode_breaker_also_fails_whole_request() {
    let ctx = AppTestContext::new_with_config(
        pd_config(4382, 20884, 20885),
        vec![
            TestWorkerConfig::prefill(20884),
            TestWorkerConfig::decode(20885),
        ],
    )
    .await;
    let app = ctx.create_app().await;
    let decode = only_worker(
        ctx.app_context.worker_registry.get_decode_workers(),
        "decode",
    );
    decode.circuit_breaker().force_open();

    assert_eq!(generate_status(&app).await, StatusCode::SERVICE_UNAVAILABLE);
    assert!(decode.is_healthy());
    assert_eq!(decode.circuit_breaker().state(), CircuitState::Open);

    ctx.shutdown().await;
}
