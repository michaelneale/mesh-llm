use super::super::http::{respond_error, respond_json};
use crate::models::{
    catalog, search_catalog_json_payload, search_catalog_models, search_huggingface,
    search_huggingface_json_payload, SearchArtifactFilter, SearchSort,
};
use url::form_urlencoded;

const DEFAULT_LIMIT: usize = 20;
const MAX_LIMIT: usize = 50;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchRequest {
    query: String,
    artifact: SearchArtifactFilter,
    catalog_only: bool,
    limit: usize,
    sort: SearchSort,
}

pub(super) async fn handle(stream: &mut tokio::net::TcpStream, path: &str) -> anyhow::Result<()> {
    let request = match parse_request(path) {
        Ok(request) => request,
        Err(message) => return respond_error(stream, 400, &message).await,
    };

    if request.catalog_only {
        let results = search_catalog_models(&request.query)
            .into_iter()
            .filter(|model| catalog_model_matches_artifact(model, request.artifact))
            .collect::<Vec<_>>();
        let response = search_catalog_json_payload(
            &request.query,
            request.artifact,
            request.sort,
            &results,
            request.limit,
        );
        return respond_json(stream, 200, &response).await;
    }

    match search_huggingface(
        &request.query,
        request.limit,
        request.artifact,
        request.sort,
        |_| {},
    )
    .await
    {
        Ok(results) => {
            let response = search_huggingface_json_payload(
                &request.query,
                request.artifact,
                request.sort,
                &results,
            );
            respond_json(stream, 200, &response).await
        }
        Err(err) => respond_error(stream, 502, &format!("Search failed: {err}")).await,
    }
}

fn parse_request(path: &str) -> Result<SearchRequest, String> {
    let mut query = None;
    let mut artifact = SearchArtifactFilter::Gguf;
    let mut catalog_only = false;
    let mut limit = DEFAULT_LIMIT;
    let mut sort = SearchSort::Trending;

    if let Some((_, raw_query)) = path.split_once('?') {
        for (key, value) in form_urlencoded::parse(raw_query.as_bytes()) {
            match key.as_ref() {
                "q" => query = Some(value),
                "artifact" => artifact = parse_artifact(&value)?,
                "catalog" => catalog_only = parse_bool(&value, "catalog")?,
                "limit" => limit = parse_limit(&value)?,
                "sort" => sort = parse_sort(&value)?,
                _ => {}
            }
        }
    }

    let query = query
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing required 'q' query parameter".to_string())?
        .to_string();

    Ok(SearchRequest {
        query,
        artifact,
        catalog_only,
        limit,
        sort,
    })
}

fn parse_artifact(value: &str) -> Result<SearchArtifactFilter, String> {
    match value {
        "gguf" => Ok(SearchArtifactFilter::Gguf),
        "mlx" => Ok(SearchArtifactFilter::Mlx),
        _ => Err(format!(
            "Invalid 'artifact' value '{value}'. Expected 'gguf' or 'mlx'"
        )),
    }
}

fn parse_bool(value: &str, field: &str) -> Result<bool, String> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "Invalid '{field}' value '{value}'. Expected true or false"
        )),
    }
}

fn parse_limit(value: &str) -> Result<usize, String> {
    let limit = value
        .parse::<usize>()
        .map_err(|_| format!("Invalid 'limit' value '{value}'. Expected a positive integer"))?;
    if limit == 0 {
        return Err("Invalid 'limit' value '0'. Expected a positive integer".to_string());
    }
    Ok(limit.min(MAX_LIMIT))
}

fn parse_sort(value: &str) -> Result<SearchSort, String> {
    match value {
        "trending" => Ok(SearchSort::Trending),
        "downloads" => Ok(SearchSort::Downloads),
        "likes" => Ok(SearchSort::Likes),
        "created" => Ok(SearchSort::Created),
        "updated" => Ok(SearchSort::Updated),
        "parameters-desc" => Ok(SearchSort::ParametersDesc),
        "parameters-asc" => Ok(SearchSort::ParametersAsc),
        _ => Err(format!(
            "Invalid 'sort' value '{value}'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        )),
    }
}

fn catalog_model_matches_artifact(
    model: &catalog::CatalogModel,
    artifact: SearchArtifactFilter,
) -> bool {
    let is_mlx = model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors");
    match artifact {
        SearchArtifactFilter::Gguf => !is_mlx,
        SearchArtifactFilter::Mlx => is_mlx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_requires_non_empty_query() {
        let err = parse_request("/api/search?artifact=gguf").unwrap_err();
        assert_eq!(err, "Missing required 'q' query parameter");

        let err = parse_request("/api/search?q=%20%20").unwrap_err();
        assert_eq!(err, "Missing required 'q' query parameter");
    }

    #[test]
    fn parse_request_accepts_canonical_sort_names_and_caps_limit() {
        let request = parse_request(
            "/api/search?q=qwen&artifact=mlx&catalog=true&limit=999&sort=parameters-desc",
        )
        .unwrap();
        assert_eq!(request.query, "qwen");
        assert_eq!(request.artifact, SearchArtifactFilter::Mlx);
        assert!(request.catalog_only);
        assert_eq!(request.limit, MAX_LIMIT);
        assert_eq!(request.sort, SearchSort::ParametersDesc);
    }

    #[test]
    fn parse_request_rejects_invalid_values() {
        let err = parse_request("/api/search?q=qwen&artifact=onnx").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'artifact' value 'onnx'. Expected 'gguf' or 'mlx'"
        );

        let err = parse_request("/api/search?q=qwen&limit=0").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'limit' value '0'. Expected a positive integer"
        );

        let err = parse_request("/api/search?q=qwen&sort=random").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'sort' value 'random'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        );

        let err = parse_request("/api/search?q=qwen&sort=most-parameters").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'sort' value 'most-parameters'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        );
    }
}
