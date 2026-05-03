use anyhow::Result;

pub(crate) async fn dispatch_download_command(name: Option<&str>, draft: bool) -> Result<()> {
    match name {
        Some(query) => {
            let model = crate::models::find_catalog_model_exact(query)
                .or_else(|| crate::models::catalog::find_model(query))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No model matching '{}' in catalog. Run `mesh-llm download` to list.",
                        query
                    )
                })?;
            crate::models::catalog::download_model(model).await?;
            if draft {
                if let Some(draft_name) = model.draft.as_deref() {
                    let draft_model = crate::models::find_catalog_model_exact(draft_name)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Draft model '{}' not found in catalog", draft_name)
                        })?;
                    crate::models::catalog::download_model(draft_model).await?;
                } else {
                    eprintln!("⚠ No draft model available for {}", model.name);
                }
            }
        }
        None => crate::models::catalog::list_models(),
    }
    Ok(())
}
