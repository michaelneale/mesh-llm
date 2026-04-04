use super::{PluginManager, BLOBSTORE_PLUGIN_ID};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const PUT_REQUEST_OBJECT_METHOD: &str = "blobstore/put_request_object";
pub const GET_REQUEST_OBJECT_METHOD: &str = "blobstore/get_request_object";
pub const COMPLETE_REQUEST_METHOD: &str = "blobstore/complete_request";
pub const ABORT_REQUEST_METHOD: &str = "blobstore/abort_request";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PutRequestObjectRequest {
    pub request_id: String,
    pub mime_type: String,
    #[serde(default)]
    pub file_name: Option<String>,
    pub bytes_base64: String,
    #[serde(default)]
    pub expires_in_secs: Option<u64>,
    #[serde(default)]
    pub uses_remaining: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PutRequestObjectResponse {
    pub token: String,
    pub request_id: String,
    pub mime_type: String,
    #[serde(default)]
    pub file_name: Option<String>,
    pub size_bytes: u64,
    pub sha256_hex: String,
    pub created_at: u64,
    pub expires_at: u64,
    pub uses_remaining: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GetRequestObjectRequest {
    pub token: String,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GetRequestObjectResponse {
    pub token: String,
    pub request_id: String,
    pub mime_type: String,
    #[serde(default)]
    pub file_name: Option<String>,
    pub bytes_base64: String,
    pub size_bytes: u64,
    pub sha256_hex: String,
    pub created_at: u64,
    pub expires_at: u64,
    pub uses_remaining: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinishRequestRequest {
    pub request_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinishRequestResponse {
    pub request_id: String,
    pub removed_tokens: usize,
    pub removed_bytes: u64,
}

#[allow(dead_code)]
pub async fn put_request_object(
    plugin_manager: &PluginManager,
    request: PutRequestObjectRequest,
) -> Result<PutRequestObjectResponse> {
    plugin_manager
        .mcp_request(BLOBSTORE_PLUGIN_ID, PUT_REQUEST_OBJECT_METHOD, request)
        .await
}

#[allow(dead_code)]
pub async fn get_request_object(
    plugin_manager: &PluginManager,
    request: GetRequestObjectRequest,
) -> Result<GetRequestObjectResponse> {
    plugin_manager
        .mcp_request(BLOBSTORE_PLUGIN_ID, GET_REQUEST_OBJECT_METHOD, request)
        .await
}

#[allow(dead_code)]
pub async fn complete_request(
    plugin_manager: &PluginManager,
    request: FinishRequestRequest,
) -> Result<FinishRequestResponse> {
    plugin_manager
        .mcp_request(BLOBSTORE_PLUGIN_ID, COMPLETE_REQUEST_METHOD, request)
        .await
}

#[allow(dead_code)]
pub async fn abort_request(
    plugin_manager: &PluginManager,
    request: FinishRequestRequest,
) -> Result<FinishRequestResponse> {
    plugin_manager
        .mcp_request(BLOBSTORE_PLUGIN_ID, ABORT_REQUEST_METHOD, request)
        .await
}
