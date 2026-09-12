use super::{
    DEFAULT_GRPC_MAX_MESSAGE_SIZE, openai_status_code, resolve_max_message_size,
    terminal_error_status,
};
use crate::bridge::TerminalError;
use std::collections::HashMap;
use std::time::Duration;
use tokio_stream::StreamExt;
use tonic::Code;
use tonic::Request;
use tonic_health::ServingStatus;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus as WireServingStatus;
use tonic_health::pb::health_server::Health;

#[test]
fn standard_health_status_matches_native_health() {
    assert_eq!(super::serving_status(true), ServingStatus::Serving);
    assert_eq!(super::serving_status(false), ServingStatus::NotServing);
}

#[tokio::test]
async fn standard_health_watch_ignores_unchanged_updates() {
    let (mut reporter, service) = super::health_pair();
    reporter
        .set_service_status("test.Service", ServingStatus::NotServing)
        .await;
    let mut stream = service
        .watch(Request::new(HealthCheckRequest {
            service: "test.Service".into(),
        }))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(
        stream.next().await.unwrap().unwrap().status,
        WireServingStatus::NotServing as i32
    );
    reporter
        .set_service_status("test.Service", ServingStatus::NotServing)
        .await;
    assert!(
        tokio::time::timeout(Duration::from_millis(50), stream.next())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn standard_health_watch_observes_late_registration() {
    let (mut reporter, service) = super::health_pair();
    let mut stream = service
        .watch(Request::new(HealthCheckRequest {
            service: "late.Service".into(),
        }))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(
        stream.next().await.unwrap().unwrap().status,
        WireServingStatus::ServiceUnknown as i32
    );
    reporter
        .set_service_status("late.Service", ServingStatus::Serving)
        .await;
    assert_eq!(
        stream.next().await.unwrap().unwrap().status,
        WireServingStatus::Serving as i32
    );
}

#[test]
fn openai_status_code_uses_forwarded_status_when_present() {
    let meta_info = HashMap::from([(String::from("status_code"), String::from("429"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 429);
}

#[test]
fn openai_status_code_falls_back_when_missing_or_invalid() {
    assert_eq!(openai_status_code(&HashMap::new(), 200), 200);

    let meta_info = HashMap::from([(String::from("status_code"), String::from("not-an-int"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 200);
}

#[test]
fn terminal_error_status_maps_channel_full_to_resource_exhausted() {
    let status = terminal_error_status(TerminalError::ChannelFull {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::ResourceExhausted);
}

#[test]
fn terminal_error_status_maps_abort_to_cancelled() {
    let status = terminal_error_status(TerminalError::Aborted {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::Cancelled);
}

// SAFETY: env vars are process-global; bundle all SGLANG_TONIC_PAYLOAD cases into one
// serial test so they don't race each other under `cargo test`'s default parallelism.
#[test]
fn resolve_max_message_size_honors_env_var() {
    const VAR: &str = "SGLANG_TONIC_PAYLOAD";

    // Unset → default.
    // SAFETY: single-threaded test mutating process env (see note above).
    unsafe {
        std::env::remove_var(VAR);
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Valid override → honored verbatim.
    unsafe {
        std::env::set_var(VAR, "1048576");
    }
    assert_eq!(resolve_max_message_size(), 1_048_576);

    // Invalid string → warn + fall back to default.
    unsafe {
        std::env::set_var(VAR, "not-a-number");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Zero → treated as invalid, fall back to default.
    unsafe {
        std::env::set_var(VAR, "0");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    unsafe {
        std::env::remove_var(VAR);
    }
}
