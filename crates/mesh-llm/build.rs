use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=ui/dist");
    configure_console_dist();
    watch_path(Path::new("proto"));
    compile_node_proto();
}

fn configure_console_dist() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir");
    let console_dist = Path::new(&manifest_dir).join("ui/dist");

    if console_dist.is_dir() {
        println!(
            "cargo:rustc-env=MESH_LLM_CONSOLE_DIST={}",
            console_dist.display()
        );
        return;
    }

    let fallback =
        Path::new(&std::env::var("OUT_DIR").expect("cargo out dir")).join("empty-console-dist");
    fs::create_dir_all(&fallback).expect("create fallback console dist dir");
    println!(
        "cargo:rustc-env=MESH_LLM_CONSOLE_DIST={}",
        fallback.display()
    );
}

fn watch_path(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    let Ok(meta) = fs::metadata(path) else {
        return;
    };
    if meta.is_dir() {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            watch_path(&entry.path());
        }
    }
}

fn compile_node_proto() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc);

    prost_build::Config::new()
        .compile_protos(&["proto/node.proto"], &["proto"])
        .expect("compile node proto");
}
