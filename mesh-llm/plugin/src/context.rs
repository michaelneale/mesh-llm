use std::{marker::PhantomData, sync::Arc};

use anyhow::Result;
use serde::Serialize;
use tokio::sync::Mutex;

use crate::{
    helpers::{channel_message, json_channel_message},
    io::{send_bulk_transfer_message, send_channel_message, write_envelope, LocalStream},
    proto, PROTOCOL_VERSION,
};

#[derive(Clone)]
pub struct PluginHandle {
    stream: Arc<Mutex<LocalStream>>,
    plugin_id: Arc<str>,
}

impl PluginHandle {
    pub(crate) fn new(stream: Arc<Mutex<LocalStream>>, plugin_id: impl Into<Arc<str>>) -> Self {
        Self {
            stream,
            plugin_id: plugin_id.into(),
        }
    }

    pub(crate) async fn send_envelope(&self, envelope: proto::Envelope) -> Result<()> {
        let mut stream = self.stream.lock().await;
        write_envelope(&mut stream, &envelope).await
    }

    pub async fn send_channel(&self, message: proto::ChannelMessage) -> Result<()> {
        self.send_channel_message(message).await
    }

    pub async fn send_channel_message(&self, message: proto::ChannelMessage) -> Result<()> {
        let mut stream = self.stream.lock().await;
        send_channel_message(&mut stream, &self.plugin_id, message).await
    }

    pub async fn send_text_channel(
        &self,
        channel: impl Into<String>,
        target_peer_id: impl Into<String>,
        message_kind: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<()> {
        self.send_channel_message(channel_message(
            channel,
            target_peer_id,
            "text/plain",
            text.into().into_bytes(),
            message_kind,
        ))
        .await
    }

    pub async fn send_json_channel<T: Serialize>(
        &self,
        channel: impl Into<String>,
        target_peer_id: impl Into<String>,
        message_kind: impl Into<String>,
        payload: &T,
    ) -> Result<()> {
        self.send_channel_message(json_channel_message(
            channel,
            target_peer_id,
            message_kind,
            payload,
        )?)
        .await
    }

    pub async fn send_bulk(&self, message: proto::BulkTransferMessage) -> Result<()> {
        self.send_bulk_transfer_message(message).await
    }

    pub async fn send_bulk_transfer_message(
        &self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        let mut stream = self.stream.lock().await;
        send_bulk_transfer_message(&mut stream, &self.plugin_id, message).await
    }

    pub async fn notify_host<P>(&self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        self.send_envelope(proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.plugin_id.to_string(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::RpcNotification(
                proto::RpcNotification {
                    method: method.to_string(),
                    params_json: serde_json::to_string(&params)?,
                },
            )),
        })
        .await
    }
}

pub struct PluginContext<'a> {
    handle: PluginHandle,
    _marker: PhantomData<&'a ()>,
}

impl<'a> PluginContext<'a> {
    pub(crate) fn new(handle: PluginHandle) -> Self {
        Self {
            handle,
            _marker: PhantomData,
        }
    }

    pub fn handle(&self) -> PluginHandle {
        self.handle.clone()
    }

    pub async fn send_channel(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.handle.send_channel(message).await
    }

    pub async fn send_channel_message(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.handle.send_channel_message(message).await
    }

    pub async fn send_text_channel(
        &mut self,
        channel: impl Into<String>,
        target_peer_id: impl Into<String>,
        message_kind: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<()> {
        self.handle
            .send_text_channel(channel, target_peer_id, message_kind, text)
            .await
    }

    pub async fn send_json_channel<T: Serialize>(
        &mut self,
        channel: impl Into<String>,
        target_peer_id: impl Into<String>,
        message_kind: impl Into<String>,
        payload: &T,
    ) -> Result<()> {
        self.handle
            .send_json_channel(channel, target_peer_id, message_kind, payload)
            .await
    }

    pub async fn send_bulk(&mut self, message: proto::BulkTransferMessage) -> Result<()> {
        self.handle.send_bulk(message).await
    }

    pub async fn send_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        self.handle.send_bulk_transfer_message(message).await
    }

    pub async fn notify_host<P>(&mut self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        self.handle.notify_host(method, params).await
    }
}
