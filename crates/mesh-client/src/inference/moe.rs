#[cfg(feature = "host-io")]
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoeRankingStrategy {
    #[default]
    Auto,
    Analyze,
    MicroAnalyze,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoeMicroLayerScope {
    First,
    #[default]
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedRankingKind {
    Analyze,
    MicroAnalyze,
}

impl SharedRankingKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Analyze => "analyze",
            Self::MicroAnalyze => "micro-analyze",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedRankingOrigin {
    LocalFullAnalyze,
    LocalMicroAnalyze,
    PeerImport,
    LegacyCache,
}

impl SharedRankingOrigin {
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalFullAnalyze => "local-full-analyze",
            Self::LocalMicroAnalyze => "local-micro-analyze",
            Self::PeerImport => "peer-import",
            Self::LegacyCache => "legacy-cache",
        }
    }

    pub fn from_label(value: &str) -> Option<Self> {
        match value {
            "local-full-analyze" => Some(Self::LocalFullAnalyze),
            "local-micro-analyze" => Some(Self::LocalMicroAnalyze),
            "peer-import" => Some(Self::PeerImport),
            "legacy-cache" => Some(Self::LegacyCache),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRankingArtifact {
    pub kind: SharedRankingKind,
    pub origin: SharedRankingOrigin,
    pub ranking: Vec<u32>,
    pub micro_prompt_count: Option<usize>,
    pub micro_tokens: Option<u32>,
    pub micro_layer_scope: Option<MoeMicroLayerScope>,
}

#[derive(Clone, Debug)]
pub struct MoeRuntimeOptions {
    pub ranking_strategy: MoeRankingStrategy,
    pub micro_prompt_count: usize,
    pub micro_tokens: u32,
    pub micro_layer_scope: MoeMicroLayerScope,
}

impl Default for MoeRuntimeOptions {
    fn default() -> Self {
        Self {
            ranking_strategy: MoeRankingStrategy::Auto,
            micro_prompt_count: 1,
            micro_tokens: 8,
            micro_layer_scope: MoeMicroLayerScope::All,
        }
    }
}

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

pub fn ranking_strength_key(artifact: &SharedRankingArtifact) -> (u8, u8, usize, u32) {
    match artifact.kind {
        SharedRankingKind::Analyze => (2, 0, 0, 0),
        SharedRankingKind::MicroAnalyze => (
            1,
            match artifact
                .micro_layer_scope
                .unwrap_or(MoeMicroLayerScope::First)
            {
                MoeMicroLayerScope::All => 1,
                MoeMicroLayerScope::First => 0,
            },
            artifact.micro_prompt_count.unwrap_or(0),
            artifact.micro_tokens.unwrap_or(0),
        ),
    }
}

pub fn better_shared_ranking(
    candidate: &SharedRankingArtifact,
    current: &SharedRankingArtifact,
) -> bool {
    ranking_strength_key(candidate) > ranking_strength_key(current)
}

/// Compute expert assignments with a configurable overlap factor.
///
/// - `overlap`: how many nodes each expert should live on (1 = no redundancy,
///   2 = every expert on at least 2 nodes, etc.). Capped at n_nodes.
///
/// Strategy:
/// 1. Shared core = top `min_experts` by gate mass → replicated to every node
/// 2. Remaining experts distributed with `overlap` copies each
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
        return vec![
            NodeAssignment {
                experts: ranking.to_vec(),
                n_shared: n_expert,
                n_unique: 0,
            };
            n_nodes.max(1)
        ];
    }

    let shared_core: Vec<u32> = ranking[..min_exp].to_vec();
    let remaining: Vec<u32> = ranking[min_exp..].to_vec();

    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
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
        experts.dedup();

        assignments.push(NodeAssignment {
            experts,
            n_shared: min_exp,
            n_unique,
        });
    }

    assignments
}

/// Compute expert assignments by snake-drafting the ranking across nodes.
///
/// The first `replicate` experts are replicated to every node. Remaining experts
/// are assigned in snake order to balance hot and cold experts across nodes.
pub fn compute_snake_draft_assignments(
    ranking: &[u32],
    n_nodes: usize,
    replicate: usize,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    if n_nodes <= 1 || replicate >= n_expert {
        return vec![
            NodeAssignment {
                experts: ranking.to_vec(),
                n_shared: n_expert,
                n_unique: 0,
            };
            n_nodes.max(1)
        ];
    }

    let shared_core: Vec<u32> = ranking[..replicate].to_vec();
    let remaining = &ranking[replicate..];
    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
        let round = i / n_nodes;
        let pos = i % n_nodes;
        let node = if round.is_multiple_of(2) {
            pos
        } else {
            n_nodes - 1 - pos
        };
        node_experts[node].push(expert_id);
    }

    node_experts
        .into_iter()
        .map(|node_unique| {
            let n_unique = node_unique.len();
            let mut experts = shared_core.clone();
            experts.extend(node_unique);
            experts.sort();
            NodeAssignment {
                experts,
                n_shared: shared_core.len(),
                n_unique,
            }
        })
        .collect()
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

#[cfg(feature = "host-io")]
#[derive(Default)]
struct CachedRankingMetadata {
    ranking_origin: Option<SharedRankingOrigin>,
    micro_prompt_count: Option<usize>,
    micro_tokens: Option<u32>,
    micro_layer_scope: Option<MoeMicroLayerScope>,
}

#[cfg(feature = "host-io")]
struct CachedRankingFile {
    ranking: Vec<u32>,
    metadata: CachedRankingMetadata,
}

#[cfg(feature = "host-io")]
fn parse_cached_ranking_metadata(line: &str, metadata: &mut CachedRankingMetadata) {
    let Some(rest) = line.strip_prefix('#') else {
        return;
    };
    let rest = rest.trim();
    let Some((key, value)) = rest.split_once('=') else {
        return;
    };
    let value = value.trim();
    match key.trim() {
        "ranking_origin" => metadata.ranking_origin = SharedRankingOrigin::from_label(value),
        "micro_prompt_count" => metadata.micro_prompt_count = value.parse().ok(),
        "micro_tokens" => metadata.micro_tokens = value.parse().ok(),
        "micro_layer_scope" => {
            metadata.micro_layer_scope = match value {
                "all" => Some(MoeMicroLayerScope::All),
                "first" => Some(MoeMicroLayerScope::First),
                _ => None,
            }
        }
        _ => {}
    }
}

#[cfg(feature = "host-io")]
fn load_cached_ranking_file(path: &Path) -> Option<CachedRankingFile> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut metadata = CachedRankingMetadata::default();
    let ranking: Vec<u32> = content
        .lines()
        .filter_map(|line| {
            if line.is_empty() {
                return None;
            }
            if line.starts_with('#') {
                parse_cached_ranking_metadata(line, &mut metadata);
                return None;
            }
            if line.starts_with("expert") {
                return None;
            }
            line.split(',').next()?.trim().parse().ok()
        })
        .collect();
    if ranking.is_empty() {
        None
    } else {
        Some(CachedRankingFile { ranking, metadata })
    }
}

#[cfg(feature = "host-io")]
pub fn load_cached_ranking(path: &Path) -> Option<Vec<u32>> {
    load_cached_ranking_file(path).map(|file| file.ranking)
}

#[cfg(feature = "host-io")]
pub fn load_shared_ranking_artifact(
    path: &Path,
    kind: SharedRankingKind,
    fallback_origin: SharedRankingOrigin,
    micro_prompt_count: Option<usize>,
    micro_tokens: Option<u32>,
    micro_layer_scope: Option<MoeMicroLayerScope>,
) -> Option<SharedRankingArtifact> {
    let file = load_cached_ranking_file(path)?;
    Some(SharedRankingArtifact {
        kind,
        origin: file.metadata.ranking_origin.unwrap_or(fallback_origin),
        ranking: file.ranking,
        micro_prompt_count: file.metadata.micro_prompt_count.or(micro_prompt_count),
        micro_tokens: file.metadata.micro_tokens.or(micro_tokens),
        micro_layer_scope: file.metadata.micro_layer_scope.or(micro_layer_scope),
    })
}

#[cfg(feature = "host-io")]
pub fn write_shared_ranking_artifact(
    path: &Path,
    artifact: &SharedRankingArtifact,
) -> anyhow::Result<()> {
    if artifact.ranking.is_empty() {
        anyhow::bail!("cannot write empty ranking to {}", path.display());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut lines = vec![
        "# mesh-llm-moe-ranking=v1".to_string(),
        format!("# ranking_kind={}", artifact.kind.label()),
        format!("# ranking_origin={}", artifact.origin.label()),
    ];
    if let Some(prompt_count) = artifact.micro_prompt_count {
        lines.push(format!("# micro_prompt_count={prompt_count}"));
    }
    if let Some(tokens) = artifact.micro_tokens {
        lines.push(format!("# micro_tokens={tokens}"));
    }
    if let Some(layer_scope) = artifact.micro_layer_scope {
        let scope = match layer_scope {
            MoeMicroLayerScope::First => "first",
            MoeMicroLayerScope::All => "all",
        };
        lines.push(format!("# micro_layer_scope={scope}"));
    }
    lines.extend(artifact.ranking.iter().map(u32::to_string));
    std::fs::write(path, format!("{}\n", lines.join("\n")))?;
    Ok(())
}
