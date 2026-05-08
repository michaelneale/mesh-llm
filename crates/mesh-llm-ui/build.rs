use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=dist");
    configure_console_dist();
}

fn configure_console_dist() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir");
    let console_dist = Path::new(&manifest_dir).join("dist");

    if console_dist.is_dir() {
        println!(
            "cargo:rustc-env=MESH_LLM_UI_DIST={}",
            console_dist.display()
        );
        return;
    }

    let fallback =
        Path::new(&std::env::var("OUT_DIR").expect("cargo out dir")).join("empty-ui-dist");
    fs::create_dir_all(&fallback).expect("create fallback UI dist dir");
    println!("cargo:rustc-env=MESH_LLM_UI_DIST={}", fallback.display());
}
