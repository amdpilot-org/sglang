use std::sync::Arc;

use smg_mesh::{MeshServerHandler, MeshSyncManager};

use crate::{
    app_context::AppContext,
    middleware::QueuedRequest,
    routers::{router_manager::RouterManager, RouterTrait},
};

/// Shared state injected into Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub mesh_handler: Option<Arc<MeshServerHandler>>,
    pub mesh_sync_manager: Option<Arc<MeshSyncManager>>,
}
