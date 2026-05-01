use crate::system::moe_planner::{MoePlanReport, RankingSource};
use anyhow::Result;
use serde_json::{json, Value};

pub(crate) trait MoePlanFormatter {
    fn render(&self, report: &MoePlanReport) -> Result<()>;
}

pub(crate) struct ConsoleFormatter;
pub(crate) struct JsonFormatter;

pub(crate) fn moe_plan_formatter(json_output: bool) -> Box<dyn MoePlanFormatter> {
    if json_output {
        Box::new(JsonFormatter)
    } else {
        Box::new(ConsoleFormatter)
    }
}

pub(crate) fn print_json(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

pub(crate) fn ranking_source_label(source: &RankingSource) -> &'static str {
    match source {
        RankingSource::Override => "override",
        RankingSource::LocalCache => "local_cache",
        RankingSource::HuggingFaceDataset => "huggingface_dataset",
    }
}

pub(crate) fn plan_json(report: &MoePlanReport) -> Value {
    json!({
        "model": {
            "input": report.model.input,
            "display_name": report.model.display_name,
            "path": report.model.path,
            "source_repo": report.model.source_repo,
            "source_revision": report.model.source_revision,
            "distribution_id": report.model.distribution_id,
            "expert_count": report.model.expert_count,
            "used_expert_count": report.model.used_expert_count,
            "min_experts_per_node": report.model.min_experts_per_node,
            "total_model_bytes": report.model.total_model_bytes,
            "total_model_gb": report.model.total_model_bytes as f64 / 1e9,
        },
        "ranking": {
            "analyzer_id": report.ranking.analyzer_id,
            "source": ranking_source_label(&report.ranking.source),
            "reason": report.ranking.reason,
            "path": report.ranking.path,
            "metadata_path": report.ranking.metadata_path,
            "analysis_path": report.ranking.analysis_path,
        },
        "target": {
            "vram_bytes": report.target_vram_bytes,
            "vram_gb": report.target_vram_bytes as f64 / 1e9,
            "recommended_nodes": report.recommended_nodes,
            "max_supported_nodes": report.max_supported_nodes,
            "feasible": report.feasible,
        },
        "mass_profile": {
            "shared_mass_pct": report.shared_mass_pct,
            "max_node_mass_pct": report.max_node_mass_pct,
            "min_node_mass_pct": report.min_node_mass_pct,
        },
        "assumptions": report.assumptions,
        "assignments": report.assignments.iter().enumerate().map(|(index, assignment)| {
            json!({
                "node": index + 1,
                "expert_count": assignment.experts.len(),
                "shared": assignment.n_shared,
                "unique": assignment.n_unique,
                "experts": assignment.experts,
            })
        }).collect::<Vec<_>>(),
    })
}
