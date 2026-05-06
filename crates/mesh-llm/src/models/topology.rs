use std::path::Path;

pub use mesh_client::models::topology::{ModelMoeInfo, ModelTopology};

#[allow(dead_code)]
pub fn infer_catalog_topology(_model: &super::catalog::CatalogModel) -> Option<ModelTopology> {
    None
}

#[allow(dead_code)]
pub fn infer_local_model_topology(
    _path: &Path,
    _catalog: Option<&super::catalog::CatalogModel>,
) -> Option<ModelTopology> {
    None
}
