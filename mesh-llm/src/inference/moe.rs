//! MoE expert sharding: split models across mesh nodes by expert assignment.
//!
//! Each node gets a GGUF with the full trunk (attention, norms, embeddings, head)
//! plus a subset of experts. The shared core (hottest experts by gate mass) is
//! replicated to every node. Remaining experts are distributed uniquely.
//!
//! No cross-node traffic during inference — each node runs independently.

use std::path::{Path, PathBuf};

// ── GGUF assembler: combine trunk + expert files into a shard ──

// ── Ranking cache ──

fn mesh_cache_dir() -> PathBuf {
    if let Ok(path) = std::env::var("XDG_CACHE_HOME") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("mesh-llm");
        }
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("mesh-llm")
}

fn split_cache_root() -> PathBuf {
    mesh_cache_dir().join("splits")
}

/// Path to cached ranking CSV for a model.
/// Stored in a sibling `moe-rankings/` directory next to the source model file.
pub fn ranking_cache_path(model_path: &Path) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-rankings").join(format!("{stem}.csv"))
}

/// Load a cached ranking CSV. Format: one expert_id per line, sorted by gate mass descending.
/// Also supports the full CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
pub fn load_cached_ranking(path: &Path) -> Option<Vec<u32>> {
    let content = std::fs::read_to_string(path).ok()?;
    let ranking: Vec<u32> = content
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("expert"))
        .filter_map(|l| {
            // Support both plain "42" and CSV "42,1234.5,0.03,500"
            l.split(',').next()?.trim().parse().ok()
        })
        .collect();
    if ranking.is_empty() {
        None
    } else {
        Some(ranking)
    }
}

// ── Expert assignment ──

/// Expert assignment for a single node: which expert IDs it should hold.
#[derive(Clone, Debug)]
pub struct NodeAssignment {
    /// All expert IDs for this node (shared core + unique shard), sorted.
    pub experts: Vec<u32>,
    /// How many of these are shared (replicated to every node).
    pub n_shared: usize,
    /// How many are unique to this node.
    pub n_unique: usize,
}

/// Compute expert assignments for N nodes using the overlap strategy.
///
/// - `ranking`: expert IDs sorted by gate mass descending (hottest first)
/// - `n_nodes`: number of mesh nodes to split across
/// - `min_experts`: minimum experts per node for coherent output
///
/// Returns one NodeAssignment per node. Every expert appears in at least one node.
/// Convenience wrapper for compute_assignments_with_overlap with overlap=1.
pub fn compute_assignments(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
) -> Vec<NodeAssignment> {
    compute_assignments_with_overlap(ranking, n_nodes, min_experts, 1)
}

/// Compute expert assignments with a configurable overlap factor.
///
/// - `overlap`: how many nodes each expert should live on (1 = no redundancy,
///   2 = every expert on at least 2 nodes, etc.). Capped at n_nodes.
///
/// Strategy:
/// 1. Shared core = top `min_experts` by gate mass → replicated to every node
/// 2. Remaining experts distributed with `overlap` copies each
///
/// With overlap=2, losing any single node doesn't orphan any expert —
/// at least one other node still has it.
pub fn compute_assignments_with_overlap(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
    overlap: usize,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    let min_exp = min_experts as usize;
    let overlap = overlap.min(n_nodes).max(1);

    if n_nodes <= 1 || min_exp >= n_expert {
        // Single node or core covers everything — give everyone all experts
        return vec![
            NodeAssignment {
                experts: ranking.to_vec(),
                n_shared: n_expert,
                n_unique: 0,
            };
            n_nodes.max(1)
        ];
    }

    // Shared core = top min_experts by gate mass (replicated to every node)
    let shared_core: Vec<u32> = ranking[..min_exp].to_vec();

    // Remaining experts to distribute with overlap
    let remaining: Vec<u32> = ranking[min_exp..].to_vec();

    // With overlap, each expert goes to `overlap` nodes.
    // Total expert-slots = remaining.len() * overlap, distributed round-robin.
    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
        // Assign to `overlap` consecutive nodes (wrapping)
        for j in 0..overlap {
            let node = (i + j) % n_nodes;
            node_experts[node].push(expert_id);
        }
    }

    let mut assignments = Vec::with_capacity(n_nodes);
    for node_exps in node_experts {
        let n_unique = node_exps.len();
        let mut experts = shared_core.clone();
        experts.extend_from_slice(&node_exps);
        experts.sort();
        experts.dedup(); // in case overlap wraps and duplicates with shared core

        assignments.push(NodeAssignment {
            experts,
            n_shared: min_exp,
            n_unique,
        });
    }

    assignments
}

/// Format expert list as comma-separated string for moe-split --expert-list.
pub fn expert_list_arg(assignment: &NodeAssignment) -> String {
    assignment
        .experts
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Path to the cached split GGUF for a given model + node count + node index.
pub fn split_path(model_path: &Path, n_nodes: usize, node_index: usize) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let new_path = split_cache_root()
        .join(format!("{stem}"))
        .join(format!("{n_nodes}-nodes"))
        .join(format!("node-{node_index}.gguf"));
    migrate_legacy_split_if_present(model_path, n_nodes, node_index, &new_path);
    new_path
}

fn legacy_split_path(model_path: &Path, n_nodes: usize, node_index: usize) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-splits")
        .join(format!("{stem}"))
        .join(format!("{n_nodes}-nodes"))
        .join(format!("node-{node_index}.gguf"))
}

fn migrate_legacy_split_if_present(
    model_path: &Path,
    n_nodes: usize,
    node_index: usize,
    new_path: &Path,
) {
    if new_path.exists() {
        return;
    }

    let legacy_path = legacy_split_path(model_path, n_nodes, node_index);
    if !legacy_path.exists() {
        return;
    }

    if let Some(parent) = new_path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }

    if std::fs::rename(&legacy_path, new_path).is_ok() {
        cleanup_empty_legacy_split_dirs(&legacy_path);
        return;
    }

    if std::fs::copy(&legacy_path, new_path).is_ok() {
        let _ = std::fs::remove_file(&legacy_path);
        cleanup_empty_legacy_split_dirs(&legacy_path);
    }
}

fn cleanup_empty_legacy_split_dirs(legacy_path: &Path) {
    let Some(node_dir) = legacy_path.parent() else {
        return;
    };
    let Some(model_dir) = node_dir.parent() else {
        return;
    };
    let Some(root_dir) = model_dir.parent() else {
        return;
    };

    for dir in [node_dir, model_dir, root_dir] {
        let Ok(mut entries) = std::fs::read_dir(dir) else {
            break;
        };
        if entries.next().is_none() {
            let _ = std::fs::remove_dir(dir);
        } else {
            break;
        }
    }
}

fn resolve_split_binary(bin_dir: &Path) -> anyhow::Result<PathBuf> {
    let candidates = [
        bin_dir.join("llama-moe-split"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-split"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-split"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-split"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }

    anyhow::bail!(
        "llama-moe-split not found in {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    );
}

/// Run llama-moe-split to produce a split GGUF for one node.
pub fn run_split(
    bin_dir: &Path,
    model_path: &Path,
    assignment: &NodeAssignment,
    output_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let expert_list = expert_list_arg(assignment);
    let split_bin = resolve_split_binary(bin_dir)?;
    let status = std::process::Command::new(&split_bin)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--expert-list",
            &expert_list,
            "-o",
            &output_path.to_string_lossy(),
        ])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run {}: {e}", split_bin.display()))?;

    anyhow::ensure!(status.success(), "llama-moe-split exited with {status}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gguf::detect_moe;
    use serial_test::serial;

    #[test]
    fn test_assignments_2_nodes() {
        // 10 experts, min 4, 2 nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments(&ranking, 2, 4);

        assert_eq!(assignments.len(), 2);
        // Each node: 4 shared + 3 unique = 7 experts
        assert_eq!(assignments[0].experts.len(), 7);
        assert_eq!(assignments[1].experts.len(), 7);
        assert_eq!(assignments[0].n_shared, 4);
        assert_eq!(assignments[0].n_unique, 3);

        // Shared core (0-3) in both
        for e in 0..4 {
            assert!(assignments[0].experts.contains(&e));
            assert!(assignments[1].experts.contains(&e));
        }

        // Full coverage
        let mut all: Vec<u32> = assignments[0].experts.clone();
        all.extend(&assignments[1].experts);
        all.sort();
        all.dedup();
        assert_eq!(all, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_assignments_3_nodes() {
        // 128 experts, min 46, 3 nodes
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments(&ranking, 3, 46);

        assert_eq!(assignments.len(), 3);
        // 82 remaining / 3 = 27 each + 1 leftover
        // Nodes 0: 46+28=74, Node 1: 46+27=73, Node 2: 46+27=73
        assert_eq!(assignments[0].experts.len(), 74);
        assert_eq!(assignments[1].experts.len(), 73);
        assert_eq!(assignments[2].experts.len(), 73);

        // Full coverage
        let mut all: Vec<u32> = Vec::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        all.sort();
        all.dedup();
        assert_eq!(all, (0..128).collect::<Vec<u32>>());
    }

    #[test]
    fn test_ranking_cache_roundtrip() {
        let dir = std::env::temp_dir().join("moe-test-ranking");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");

        let ranking: Vec<u32> = vec![0, 26, 41, 69, 104, 3, 7, 99];
        let content: String = ranking
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, ranking);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_moe_analyze_csv() {
        // The CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
        let dir = std::env::temp_dir().join("moe-test-csv");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ranking.csv");
        std::fs::write(
            &path,
            "expert_id,total_mass,mass_fraction,selection_count\n\
            0,8365.69,0.250,15680\n\
            26,267.43,0.008,4800\n\
            41,250.11,0.007,4600\n",
        )
        .unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, vec![0, 26, 41]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[serial]
    fn split_path_uses_mesh_cache_root() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-cache-root");

        let model = Path::new("/tmp/models/Qwen3-30B-A3B-Q4_K_M.gguf");
        let path = split_path(model, 3, 1);

        assert_eq!(
            path,
            PathBuf::from("/tmp/mesh-llm-cache-root")
                .join("mesh-llm")
                .join("splits")
                .join("Qwen3-30B-A3B-Q4_K_M")
                .join("3-nodes")
                .join("node-1.gguf")
        );

        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    #[serial]
    fn split_path_migrates_legacy_split_if_present() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        let base =
            std::env::temp_dir().join(format!("mesh-llm-moe-migrate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let cache_root = base.join("cache");
        let model_root = base.join("models");
        let model_path = model_root.join("demo.gguf");
        let legacy = model_root
            .join("moe-splits")
            .join("demo")
            .join("2-nodes")
            .join("node-0.gguf");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&model_root).unwrap();
        std::fs::write(&model_path, b"model").unwrap();
        std::fs::write(&legacy, b"legacy-split").unwrap();
        std::env::set_var("XDG_CACHE_HOME", &cache_root);

        let new_path = split_path(&model_path, 2, 0);

        assert!(new_path.exists());
        assert_eq!(std::fs::read(&new_path).unwrap(), b"legacy-split");
        assert!(!legacy.exists());

        restore_env("XDG_CACHE_HOME", prev_xdg);
        let _ = std::fs::remove_dir_all(&base);
    }

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn test_detect_moe_qwen3() {
        let hf_cache = crate::models::huggingface_hub_cache_dir();
        let path = hf_cache.join("Qwen3-30B-A3B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: model file not found");
            return;
        }
        let info = detect_moe(&path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 128);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_olmoe() {
        let hf_cache = crate::models::huggingface_hub_cache_dir();
        let path = hf_cache.join("olmoe-1b-7b-0924-instruct-q4_k_m.gguf");
        if !path.exists() {
            eprintln!("Skipping: OLMoE model file not found");
            return;
        }
        let info = detect_moe(&path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 64);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_dense_model() {
        // Qwen2.5-3B is dense (no experts) — should return None
        let hf_cache = crate::models::huggingface_hub_cache_dir();
        let path = hf_cache.join("Qwen2.5-3B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: dense model file not found");
            return;
        }
        assert!(
            detect_moe(&path).is_none(),
            "Dense model should not be detected as MoE"
        );
    }

    #[test]
    fn test_single_node() {
        let ranking: Vec<u32> = (0..8).collect();
        let assignments = compute_assignments(&ranking, 1, 4);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].experts.len(), 8); // gets everything
    }

    // ── Overlap tests ──

    #[test]
    fn test_overlap_2x_3_nodes() {
        // 128 experts, min 46, 3 nodes, 2× overlap
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 46, 2);

        assert_eq!(assignments.len(), 3);

        // Every expert should appear in at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }

        // Shared core (0..46) in all 3 nodes
        for e in 0..46 {
            assert!(
                *expert_count.get(&e).unwrap() >= 3,
                "Shared expert {e} should be in all nodes"
            );
        }
        // Remaining experts (46..128) in at least 2 nodes
        for e in 46..128 {
            assert!(
                *expert_count.get(&e).unwrap() >= 2,
                "Expert {e} should be in at least 2 nodes, got {}",
                expert_count[&e]
            );
        }
        // Full coverage
        assert_eq!(expert_count.len(), 128);
    }

    #[test]
    fn test_overlap_2x_2_nodes() {
        // With 2 nodes and 2× overlap, every remaining expert is on both nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 2, 4, 2);

        assert_eq!(assignments.len(), 2);
        // Both nodes should have all 10 experts (4 shared + 6 remaining × 2× = both)
        assert_eq!(assignments[0].experts.len(), 10);
        assert_eq!(assignments[1].experts.len(), 10);
    }

    #[test]
    fn test_overlap_1x_same_as_original() {
        // overlap=1 should give same results as compute_assignments
        let ranking: Vec<u32> = (0..128).collect();
        let a1 = compute_assignments(&ranking, 3, 46);
        let a2 = compute_assignments_with_overlap(&ranking, 3, 46, 1);

        for i in 0..3 {
            assert_eq!(a1[i].experts, a2[i].experts);
        }
    }

    #[test]
    fn test_overlap_capped_at_n_nodes() {
        // overlap=10 with 3 nodes should cap to 3 (every expert on every node)
        let ranking: Vec<u32> = (0..20).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 5, 10);

        // All 3 nodes should have all 20 experts
        for a in &assignments {
            assert_eq!(a.experts.len(), 20);
        }
    }

    #[test]
    fn test_overlap_glm5_10_nodes() {
        // GLM-5: 256 experts, min 96, 10 nodes, 2× overlap
        let ranking: Vec<u32> = (0..256).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 10, 96, 2);

        assert_eq!(assignments.len(), 10);

        // Full coverage
        let mut all: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        assert_eq!(all.len(), 256);

        // Every remaining expert on at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }
        for e in 96..256 {
            assert!(*expert_count.get(&e).unwrap() >= 2);
        }

        // Print sizes for verification
        for (i, a) in assignments.iter().enumerate() {
            eprintln!(
                "  Node {i}: {} experts ({} shared + {} unique)",
                a.experts.len(),
                a.n_shared,
                a.n_unique
            );
        }
    }
}
