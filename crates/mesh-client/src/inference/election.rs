pub use crate::mesh::should_be_host_for_model;
pub use mesh_llm_routing::{total_model_bytes, InferenceTarget, ModelTargets};

#[derive(Clone, Debug)]
pub struct LocalProcessInfo {
    pub backend: String,
    pub pid: u32,
    pub port: u16,
    pub context_length: u32,
}
