use crate::api;
use crate::inference::election;
use crate::mesh;
use crate::network::affinity;

pub(super) async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    crate::network::openai::ingress::api_proxy(
        node,
        port,
        target_rx,
        control_tx,
        existing_listener,
        listen_all,
        affinity,
    )
    .await;
}

pub(super) async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    crate::network::openai::ingress::bootstrap_proxy(node, port, stop_rx, listen_all, affinity)
        .await;
}

#[cfg(test)]
pub(super) fn callable_models(targets: &election::ModelTargets) -> Vec<String> {
    crate::network::openai::ingress::callable_models(targets)
}

#[cfg(test)]
mod tests;
