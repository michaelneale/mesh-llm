use std::path::Path;

pub use mesh_llm_types::models::topology::{ModelMoeInfo, ModelTopology};

#[allow(dead_code)]
pub fn infer_local_model_topology(_path: &Path) -> Option<ModelTopology> {
    None
}
