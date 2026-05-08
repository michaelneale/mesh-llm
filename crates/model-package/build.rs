use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn main() {
    let script_path = Path::new("src/scripts/split-model-job.sh");
    println!("cargo::rerun-if-changed={}", script_path.display());

    let bytes = fs::read(script_path).expect("read embedded job script");
    let hash = Sha256::digest(&bytes);
    let hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("script_hash.rs");
    fs::write(
        &dest,
        format!(
            "pub const EMBEDDED_SCRIPT_SHA256: &str = \"{hex}\";\n\
             pub const EMBEDDED_SCRIPT_SIZE: u64 = {};\n",
            bytes.len()
        ),
    )
    .expect("write generated script hash");
}
