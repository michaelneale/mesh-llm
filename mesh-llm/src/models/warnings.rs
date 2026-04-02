use crate::models::{huggingface_hub_cache_dir, path_is_in_legacy_models_dir};
use std::path::PathBuf;

fn print_legacy_storage_warning() {
    eprintln!("WARNING: ~/.models storage is deprecated and will be removed in a future release.");
    eprintln!(
        "Use Hugging Face repository snapshots in {} instead.",
        huggingface_hub_cache_dir().display()
    );
    eprintln!("Migration steps:");
    eprintln!("1. Run: mesh-llm models migrate");
    eprintln!("2. Migrate recognized Hugging Face-backed models: mesh-llm models migrate --apply");
    eprintln!("3. Optionally remove migrated legacy files: mesh-llm models migrate --prune");
    eprintln!("4. For custom local GGUF files, use: mesh-llm --gguf /path/to/model.gguf");
}

pub fn warn_about_legacy_model_usage(paths: &[PathBuf]) {
    if paths
        .iter()
        .any(|path| path_is_in_legacy_models_dir(path.as_path()))
    {
        print_legacy_storage_warning();
        eprintln!("No update information is available for models loaded from legacy storage.");
    }
}
