//! Embedded web-console asset accessor.
//!
//! With the default `embed-assets` feature, `include_dir!` bundles the
//! built React console (`dist/`) into the crate at compile time and
//! [`index`] / [`asset`] serve those bytes.
//!
//! With `embed-assets` disabled, the crate compiles down to ~nothing and
//! both accessors return `None`. Callers (notably
//! `mesh-llm-host-runtime`'s console asset routes) treat that as "no UI
//! bundled" and surface 404s for the asset paths while keeping every
//! other management-API surface working. This lets lib-style consumers
//! of `mesh-llm-host-runtime` drop several MB of embedded payload by
//! opting out of `default-features`.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UiAsset {
    pub contents: &'static [u8],
    pub content_type: &'static str,
    pub cache_control: &'static str,
}

#[cfg(feature = "embed-assets")]
mod embedded {
    use include_dir::{include_dir, Dir};

    pub(super) static CONSOLE_DIST: Dir<'_> = include_dir!("$MESH_LLM_UI_DIST");
}

pub fn index() -> Option<UiAsset> {
    asset("index.html").map(|mut asset| {
        asset.cache_control = "public, max-age=3600";
        asset
    })
}

#[cfg(feature = "embed-assets")]
pub fn asset(path: &str) -> Option<UiAsset> {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return None;
    }

    let file = embedded::CONSOLE_DIST.get_file(rel)?;
    Some(UiAsset {
        contents: file.contents(),
        content_type: content_type(rel),
        cache_control: cache_control(rel),
    })
}

#[cfg(not(feature = "embed-assets"))]
pub fn asset(_path: &str) -> Option<UiAsset> {
    None
}

// Helpers below are only used when `embed-assets` is on (the stub `asset`
// returns `None` without ever needing to derive a content type or cache
// header). Tests likewise only exercise the embedded path.
#[cfg(feature = "embed-assets")]
fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

#[cfg(feature = "embed-assets")]
fn cache_control(path: &str) -> &'static str {
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    }
}

#[cfg(all(test, feature = "embed-assets"))]
mod tests {
    use super::{asset, content_type};

    #[test]
    fn rejects_parent_directory_paths() {
        assert!(asset("../index.html").is_none());
    }

    #[test]
    fn maps_common_asset_content_types() {
        assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
        assert_eq!(
            content_type("assets/app.js"),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(content_type("assets/app.css"), "text/css; charset=utf-8");
        assert_eq!(
            content_type("manifest.json"),
            "application/json; charset=utf-8"
        );
    }
}

#[cfg(all(test, not(feature = "embed-assets")))]
mod stub_tests {
    use super::{asset, index};

    #[test]
    fn returns_none_when_assets_not_embedded() {
        assert!(index().is_none());
        assert!(asset("index.html").is_none());
        assert!(asset("assets/app.js").is_none());
    }
}
