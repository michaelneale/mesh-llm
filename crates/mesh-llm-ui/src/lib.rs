use include_dir::{include_dir, Dir};

static CONSOLE_DIST: Dir<'_> = include_dir!("$MESH_LLM_UI_DIST");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UiAsset {
    pub contents: &'static [u8],
    pub content_type: &'static str,
    pub cache_control: &'static str,
}

pub fn index() -> Option<UiAsset> {
    asset("index.html").map(|mut asset| {
        asset.cache_control = "public, max-age=3600";
        asset
    })
}

pub fn asset(path: &str) -> Option<UiAsset> {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return None;
    }

    let file = CONSOLE_DIST.get_file(rel)?;
    Some(UiAsset {
        contents: file.contents(),
        content_type: content_type(rel),
        cache_control: cache_control(rel),
    })
}

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

fn cache_control(path: &str) -> &'static str {
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    }
}

#[cfg(test)]
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
