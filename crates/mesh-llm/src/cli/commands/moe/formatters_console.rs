use super::formatters::{ConsoleFormatter, MoePlanFormatter};
use crate::system::moe_planner::{MoePlanReport, RankingSource};
use anyhow::Result;

impl MoePlanFormatter for ConsoleFormatter {
    fn render(&self, report: &MoePlanReport) -> Result<()> {
        let ranking_hint = match report.ranking.source {
            RankingSource::Override => "explicit override",
            RankingSource::LocalCache => "local cache",
            RankingSource::HuggingFaceDataset => "Hugging Face dataset",
        };
        println!("🧠 MoE plan");
        println!();
        println!("📦 {}", report.model.display_name);
        println!("   path: {}", report.model.path.display());
        if let Some(repo) = &report.model.source_repo {
            println!("   source: {repo}");
        }
        if let Some(revision) = &report.model.source_revision {
            println!("   revision: {revision}");
        }
        println!(
            "   experts: {} total, top-{} active",
            report.model.expert_count, report.model.used_expert_count
        );
        println!("   distribution: {}", report.model.distribution_id);
        println!();

        println!("📊 Ranking");
        println!(
            "   analyzer: {} ({})",
            report.ranking.analyzer_id, ranking_hint
        );
        println!("   file: {}", report.ranking.path.display());
        if let Some(path) = &report.ranking.analysis_path {
            println!("   analysis: {}", path.display());
        }
        println!("   reason: {}", report.ranking.reason);
        println!();

        println!("{} Target", if report.feasible { "✅" } else { "⚠️" });
        println!(
            "   max vram: {:.1}GB",
            report.target_vram_bytes as f64 / 1e9
        );
        println!("   recommended nodes: {}", report.recommended_nodes);
        println!("   max useful nodes: {}", report.max_supported_nodes);
        println!(
            "   feasible: {}",
            if report.feasible { "yes" } else { "no" }
        );
        if let Some(shared_mass_pct) = report.shared_mass_pct {
            println!("   shared mass: {:.1}%", shared_mass_pct);
        }
        if let (Some(min_mass), Some(max_mass)) =
            (report.min_node_mass_pct, report.max_node_mass_pct)
        {
            println!("   node mass range: {:.1}% – {:.1}%", min_mass, max_mass);
        }
        println!();

        println!("🧩 Suggested split");
        for (index, assignment) in report.assignments.iter().enumerate() {
            println!(
                "   node {}: {} experts ({} shared, {} unique)",
                index + 1,
                assignment.experts.len(),
                assignment.n_shared,
                assignment.n_unique
            );
        }
        println!();

        println!("📝 Assumptions");
        for assumption in &report.assumptions {
            println!("   • {assumption}");
        }

        if !report.feasible {
            println!();
            println!(
                "⚠️ This plan exceeds the current min-experts-per-node limit at the requested VRAM target."
            );
        } else if report.recommended_nodes > 1 {
            println!();
            println!("✅ This model looks viable as an MoE split at the requested VRAM target.");
        } else {
            println!();
            println!("✅ This model fits the requested VRAM target without needing an MoE split.");
        }
        Ok(())
    }
}
