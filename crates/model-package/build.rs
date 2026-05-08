use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn script_consts(name: &str, path: &Path) -> String {
    let bytes = fs::read(path).expect("read embedded job script");
    let hash = Sha256::digest(&bytes);
    let hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!(
        "pub const {name}_SHA256: &str = \"{hex}\";\n\
         pub const {name}_SIZE: u64 = {};\n",
        bytes.len()
    )
}

fn main() {
    let scripts = [
        (
            "EMBEDDED_SPLIT_SCRIPT",
            Path::new("src/scripts/split-model-job.sh"),
        ),
        (
            "EMBEDDED_CERTIFY_SCRIPT",
            Path::new("src/scripts/certify-model-job.sh"),
        ),
    ];

    let mut generated = String::new();
    for (name, path) in scripts {
        println!("cargo::rerun-if-changed={}", path.display());
        generated.push_str(&script_consts(name, path));
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("script_hash.rs");
    fs::write(&dest, generated).expect("write generated script hash");
}
