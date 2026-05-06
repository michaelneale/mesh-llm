pub(crate) use openai_frontend::{NormalizationOutcome, ResponseAdapterMode};

pub(crate) fn normalize_openai_compat_request(
    path: &str,
    body: &mut serde_json::Value,
) -> anyhow::Result<NormalizationOutcome> {
    openai_frontend::normalize_openai_compat_request(path, body).map_err(Into::into)
}
