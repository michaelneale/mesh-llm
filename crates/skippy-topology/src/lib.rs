use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TopologyPlanRequest {
    pub topology_id: String,
    pub model_id: String,
    pub layers: Vec<LayerSpec>,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub family: Option<FamilyCapabilityRecord>,
    #[serde(default)]
    pub policy: PlannerPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LayerSpec {
    pub index: u32,
    #[serde(default)]
    pub attention: bool,
    #[serde(default)]
    pub recurrent: bool,
    #[serde(default)]
    pub parameter_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct NodeSpec {
    pub node_id: String,
    #[serde(default)]
    pub cached_slice_bytes: u64,
    #[serde(default)]
    pub vram_bytes: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PlannerPolicy {
    pub allow_recurrent_state_transfer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TopologyPlan {
    pub topology_id: String,
    pub model_id: String,
    #[serde(default)]
    pub family_id: Option<String>,
    pub stages: Vec<StagePlan>,
    #[serde(default)]
    pub boundaries: Vec<BoundaryPlan>,
    pub diagnostics: Vec<PlanDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StagePlan {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub layer_count: u32,
    pub parameter_bytes: u64,
    pub state_affinity: StateAffinity,
    pub migration_policy: MigrationPolicy,
    #[serde(default)]
    pub reason_codes: Vec<PlanReasonCode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct BoundaryPlan {
    pub producer_stage_index: u32,
    pub consumer_stage_index: u32,
    pub layer_boundary: u32,
    pub decision: BoundaryDecision,
    pub wire_dtype: WireDType,
    pub raw_activation_bytes_per_token: u64,
    pub wire_payload_bytes_per_token: u64,
    #[serde(default)]
    pub reason_codes: Vec<PlanReasonCode>,
    #[serde(default)]
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryDecision {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAffinity {
    Stateless,
    AttentionKv,
    Recurrent,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationPolicy {
    FreelyMovable,
    CostedKv,
    StickyRecurrentOwner,
    RecurrentStateTransferAllowed,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PlanDiagnostic {
    pub severity: DiagnosticSeverity,
    pub code: PlanReasonCode,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReasonCode {
    ActivationOnlyBoundary,
    AttentionKvCosted,
    RecurrentOwnerSticky,
    RecurrentStateTransferAllowed,
    RecurrentStateTransferRejected,
    SharedKvRegionCut,
    TokenSidebandRequired,
    DefaultWireDtypeF16,
    Q8WireValidated,
    Q8WireRejected,
    ExactStateMobilityAccepted,
    ExactStateMobilityRejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct FamilyCapabilityRecord {
    pub family_id: String,
    pub layer_count: u32,
    pub activation_width: u32,
    pub default_wire_dtype: WireDType,
    pub q8_wire_validation: WireValidation,
    pub exact_state_mobility: ExactStateMobility,
    #[serde(default)]
    pub recurrent_ranges: Vec<LayerRange>,
    #[serde(default)]
    pub split_constraints: Vec<SplitConstraint>,
    #[serde(default)]
    pub sidebands: Vec<SidebandRequirement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireDType {
    F32,
    F16,
    Q8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireValidation {
    Untested,
    Validated,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactStateMobility {
    Untested,
    Accepted,
    RejectedTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn contains_layer(self, layer: u32) -> bool {
        self.start <= layer && layer < self.end
    }

    pub fn contains_boundary(self, boundary: u32) -> bool {
        self.start < boundary && boundary < self.end
    }

    pub fn intersects(self, start: u32, end: u32) -> bool {
        self.start < end && start < self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SplitConstraint {
    pub kind: SplitConstraintKind,
    pub range: LayerRange,
    #[serde(default)]
    pub forbidden_boundaries: Vec<u32>,
    pub reject_boundary_inside: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitConstraintKind {
    SharedKvProducerConsumer,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SidebandRequirement {
    pub kind: SidebandKind,
    pub first_required_layer: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidebandKind {
    TokenIds,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ReviewedCapabilityRecord {
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub source_repo: Option<String>,
    #[serde(default)]
    pub source_revision: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub canonical_ref: Option<String>,
    #[serde(default)]
    pub distribution_id: Option<String>,
    #[serde(default)]
    pub selector: Option<String>,
    pub capability: FamilyCapabilityRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    EmptyLayers,
    EmptyNodes,
    NonContiguousLayers {
        expected: u32,
        found: u32,
    },
    InvalidSplitBoundary {
        boundary: u32,
        layer_start: u32,
        layer_end: u32,
    },
    NonAscendingSplitBoundary {
        previous: u32,
        boundary: u32,
    },
    NotEnoughNodesForSplits {
        stages: usize,
        nodes: usize,
    },
    FamilyLayerCountMismatch {
        family_id: String,
        expected: u32,
        found: u32,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLayers => write!(f, "topology plan requires at least one layer"),
            Self::EmptyNodes => write!(f, "topology plan requires at least one node"),
            Self::NonContiguousLayers { expected, found } => write!(
                f,
                "layers must be sorted and contiguous: expected layer {expected}, found {found}"
            ),
            Self::InvalidSplitBoundary {
                boundary,
                layer_start,
                layer_end,
            } => write!(
                f,
                "invalid split boundary {boundary}; expected {layer_start} < boundary < {layer_end}"
            ),
            Self::NonAscendingSplitBoundary { previous, boundary } => write!(
                f,
                "split boundaries must be strictly ascending: previous {previous}, found {boundary}"
            ),
            Self::NotEnoughNodesForSplits { stages, nodes } => write!(
                f,
                "split plan requires {stages} nodes but only {nodes} were provided"
            ),
            Self::FamilyLayerCountMismatch {
                family_id,
                expected,
                found,
            } => write!(
                f,
                "family capability {family_id} expects {expected} layers, found {found}"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

pub fn plan_even_contiguous(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());
    let base = request.layers.len() / stage_count;
    let remainder = request.layers.len() % stage_count;
    let mut next_layer = 0usize;
    let mut ranges = Vec::with_capacity(stage_count);

    for stage_index in 0..stage_count {
        let layer_count = base + usize::from(stage_index < remainder);
        ranges.push((next_layer, next_layer + layer_count));
        next_layer += layer_count;
    }

    plan_ranges(request, &ranges)
}

pub fn plan_weighted_contiguous(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());
    let nodes = &request.nodes[..stage_count];
    let total_weight: u64 = nodes.iter().map(|node| node.vram_bytes).sum();
    if total_weight == 0 {
        return plan_even_contiguous(request);
    }

    let mut ranges = Vec::with_capacity(stage_count);
    let mut layer_start = 0usize;
    for (stage_index, node) in nodes.iter().enumerate() {
        let remaining_stages = stage_count - stage_index;
        let remaining_layers = request.layers.len() - layer_start;
        let mut span = if remaining_stages == 1 {
            remaining_layers
        } else {
            (((request.layers.len() as u128) * (node.vram_bytes as u128)) / (total_weight as u128))
                .try_into()
                .unwrap_or(usize::MAX)
        };
        span = span.max(1).min(remaining_layers - (remaining_stages - 1));
        let layer_end = layer_start + span;
        ranges.push((layer_start, layer_end));
        layer_start = layer_end;
    }

    plan_ranges(request, &ranges)
}

pub fn plan_contiguous_with_splits(
    request: &TopologyPlanRequest,
    splits: &[u32],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let layer_start = request
        .layers
        .first()
        .expect("validated non-empty layers")
        .index;
    let layer_end = request
        .layers
        .last()
        .expect("validated non-empty layers")
        .index
        + 1;
    let mut previous = layer_start;
    let mut boundaries = Vec::with_capacity(splits.len() + 2);
    boundaries.push(layer_start);
    for &boundary in splits {
        if boundary <= layer_start || boundary >= layer_end {
            return Err(PlanError::InvalidSplitBoundary {
                boundary,
                layer_start,
                layer_end,
            });
        }
        if boundary <= previous {
            return Err(PlanError::NonAscendingSplitBoundary { previous, boundary });
        }
        boundaries.push(boundary);
        previous = boundary;
    }
    boundaries.push(layer_end);

    let stage_count = boundaries.len() - 1;
    if request.nodes.len() < stage_count {
        return Err(PlanError::NotEnoughNodesForSplits {
            stages: stage_count,
            nodes: request.nodes.len(),
        });
    }

    let first_layer = request.layers[0].index;
    let ranges = boundaries
        .windows(2)
        .map(|window| {
            (
                (window[0] - first_layer) as usize,
                (window[1] - first_layer) as usize,
            )
        })
        .collect::<Vec<_>>();

    plan_ranges(request, &ranges)
}

fn plan_ranges(
    request: &TopologyPlanRequest,
    ranges: &[(usize, usize)],
) -> Result<TopologyPlan, PlanError> {
    let mut stages = Vec::with_capacity(ranges.len());

    for (stage_index, &(start, end)) in ranges.iter().enumerate() {
        let layers = &request.layers[start..end];
        let layer_start = layers.first().expect("validated non-empty range").index;
        let layer_end = layers.last().expect("validated non-empty range").index + 1;
        let state_affinity = classify_layers_with_family(layers, request.family.as_ref());
        let migration_policy = migration_policy(state_affinity, request.policy);
        let parameter_bytes = layers.iter().map(|layer| layer.parameter_bytes).sum();
        let node_id = request.nodes[stage_index].node_id.clone();
        let reason_codes =
            stage_reason_codes(state_affinity, migration_policy, request.family.as_ref());

        stages.push(StagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id,
            layer_start,
            layer_end,
            layer_count: (end - start) as u32,
            parameter_bytes,
            state_affinity,
            migration_policy,
            reason_codes,
        });
    }

    let boundaries = boundaries_for(&stages, request.family.as_ref());
    let diagnostics = diagnostics_for(
        &stages,
        &boundaries,
        request.family.as_ref(),
        request.policy,
    );

    Ok(TopologyPlan {
        topology_id: request.topology_id.clone(),
        model_id: request.model_id.clone(),
        family_id: request
            .family
            .as_ref()
            .map(|family| family.family_id.clone()),
        stages,
        boundaries,
        diagnostics,
    })
}

pub fn classify_layers(layers: &[LayerSpec]) -> StateAffinity {
    classify_layers_with_family(layers, None)
}

fn classify_layers_with_family(
    layers: &[LayerSpec],
    family: Option<&FamilyCapabilityRecord>,
) -> StateAffinity {
    let has_attention = layers.iter().any(|layer| layer.attention);
    let has_recurrent = layers.iter().any(|layer| {
        layer.recurrent
            || family.is_some_and(|family| {
                family
                    .recurrent_ranges
                    .iter()
                    .any(|range| range.contains_layer(layer.index))
            })
    });

    match (has_attention, has_recurrent) {
        (false, false) => StateAffinity::Stateless,
        (true, false) => StateAffinity::AttentionKv,
        (false, true) => StateAffinity::Recurrent,
        (true, true) => StateAffinity::Mixed,
    }
}

fn migration_policy(affinity: StateAffinity, policy: PlannerPolicy) -> MigrationPolicy {
    match affinity {
        StateAffinity::Stateless => MigrationPolicy::FreelyMovable,
        StateAffinity::AttentionKv => MigrationPolicy::CostedKv,
        StateAffinity::Recurrent | StateAffinity::Mixed => {
            if policy.allow_recurrent_state_transfer {
                MigrationPolicy::RecurrentStateTransferAllowed
            } else {
                MigrationPolicy::StickyRecurrentOwner
            }
        }
    }
}

fn stage_reason_codes(
    affinity: StateAffinity,
    migration_policy: MigrationPolicy,
    family: Option<&FamilyCapabilityRecord>,
) -> Vec<PlanReasonCode> {
    let mut codes = Vec::new();
    match migration_policy {
        MigrationPolicy::FreelyMovable => {}
        MigrationPolicy::CostedKv => codes.push(PlanReasonCode::AttentionKvCosted),
        MigrationPolicy::StickyRecurrentOwner => codes.push(PlanReasonCode::RecurrentOwnerSticky),
        MigrationPolicy::RecurrentStateTransferAllowed => {
            codes.push(PlanReasonCode::RecurrentStateTransferAllowed)
        }
    }
    if matches!(affinity, StateAffinity::Recurrent | StateAffinity::Mixed)
        && !codes.contains(&PlanReasonCode::RecurrentOwnerSticky)
        && !codes.contains(&PlanReasonCode::RecurrentStateTransferAllowed)
    {
        codes.push(PlanReasonCode::RecurrentOwnerSticky);
    }
    if let Some(family) = family {
        match family.exact_state_mobility {
            ExactStateMobility::Accepted => codes.push(PlanReasonCode::ExactStateMobilityAccepted),
            ExactStateMobility::RejectedTooLarge => {
                codes.push(PlanReasonCode::ExactStateMobilityRejected)
            }
            ExactStateMobility::Untested => {}
        }
    }
    codes
}

fn boundaries_for(
    stages: &[StagePlan],
    family: Option<&FamilyCapabilityRecord>,
) -> Vec<BoundaryPlan> {
    stages
        .windows(2)
        .map(|window| {
            let producer = &window[0];
            let consumer = &window[1];
            let layer_boundary = producer.layer_end;
            let mut decision = BoundaryDecision::Accepted;
            let mut reason_codes = vec![PlanReasonCode::ActivationOnlyBoundary];
            let mut messages = vec![format!(
                "activation boundary after layer {}; send activation frame from {} to {}",
                layer_boundary, producer.stage_id, consumer.stage_id
            )];

            if matches!(
                producer.migration_policy,
                MigrationPolicy::StickyRecurrentOwner
            ) || matches!(
                consumer.migration_policy,
                MigrationPolicy::StickyRecurrentOwner
            ) {
                reason_codes.push(PlanReasonCode::RecurrentOwnerSticky);
                messages.push(
                    "recurrent state remains with the owning stage; only activation crosses this boundary"
                        .to_string(),
                );
            }

            let (wire_dtype, raw_activation_bytes_per_token, wire_payload_bytes_per_token) =
                if let Some(family) = family {
                    apply_family_boundary_rules(
                        family,
                        layer_boundary,
                        &mut decision,
                        &mut reason_codes,
                        &mut messages,
                    );
                    let raw = u64::from(family.activation_width) * 4;
                    let wire = wire_payload_bytes_per_token(
                        family.activation_width,
                        family.default_wire_dtype,
                    );
                    (family.default_wire_dtype, raw, wire)
                } else {
                    (WireDType::F16, 0, 0)
                };

            BoundaryPlan {
                producer_stage_index: producer.stage_index,
                consumer_stage_index: consumer.stage_index,
                layer_boundary,
                decision,
                wire_dtype,
                raw_activation_bytes_per_token,
                wire_payload_bytes_per_token,
                reason_codes,
                messages,
            }
        })
        .collect()
}

fn apply_family_boundary_rules(
    family: &FamilyCapabilityRecord,
    layer_boundary: u32,
    decision: &mut BoundaryDecision,
    reason_codes: &mut Vec<PlanReasonCode>,
    messages: &mut Vec<String>,
) {
    if family.default_wire_dtype == WireDType::F16 {
        reason_codes.push(PlanReasonCode::DefaultWireDtypeF16);
    }

    match family.q8_wire_validation {
        WireValidation::Validated => reason_codes.push(PlanReasonCode::Q8WireValidated),
        WireValidation::Rejected => reason_codes.push(PlanReasonCode::Q8WireRejected),
        WireValidation::Untested => {}
    }

    for constraint in &family.split_constraints {
        if constraint.forbidden_boundaries.contains(&layer_boundary)
            || (constraint.reject_boundary_inside
                && constraint.range.contains_boundary(layer_boundary))
        {
            *decision = BoundaryDecision::Rejected;
            reason_codes.push(match constraint.kind {
                SplitConstraintKind::SharedKvProducerConsumer => PlanReasonCode::SharedKvRegionCut,
            });
            messages.push(constraint.reason.clone());
        }
    }

    for sideband in &family.sidebands {
        if layer_boundary <= sideband.first_required_layer {
            reason_codes.push(match sideband.kind {
                SidebandKind::TokenIds => PlanReasonCode::TokenSidebandRequired,
            });
            messages.push(sideband.reason.clone());
        }
    }
}

pub fn wire_payload_bytes_per_token(activation_width: u32, dtype: WireDType) -> u64 {
    match dtype {
        WireDType::F32 => u64::from(activation_width) * 4,
        WireDType::F16 => u64::from(activation_width) * 2,
        WireDType::Q8 => u64::from(activation_width) + 4,
    }
}

fn diagnostics_for(
    stages: &[StagePlan],
    boundaries: &[BoundaryPlan],
    family: Option<&FamilyCapabilityRecord>,
    policy: PlannerPolicy,
) -> Vec<PlanDiagnostic> {
    let mut diagnostics = Vec::new();
    for stage in stages {
        if matches!(
            stage.migration_policy,
            MigrationPolicy::StickyRecurrentOwner
        ) {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::RecurrentOwnerSticky,
                message: format!(
                    "{} owns recurrent state for layers {}..{}; route future tokens back to {} and only transfer activations across stage boundaries",
                    stage.stage_id, stage.layer_start, stage.layer_end, stage.node_id
                ),
            });
        }
    }

    if policy.allow_recurrent_state_transfer {
        diagnostics.push(PlanDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: PlanReasonCode::RecurrentStateTransferAllowed,
            message: "recurrent state transfer is enabled; this should be reserved for explicit recompute-or-transfer flows, not normal routing".to_string(),
        });
    }

    if let Some(family) = family {
        match family.exact_state_mobility {
            ExactStateMobility::Accepted => diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::ExactStateMobilityAccepted,
                message: format!(
                    "{} exact state mobility is within current payload policy",
                    family.family_id
                ),
            }),
            ExactStateMobility::RejectedTooLarge => diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: PlanReasonCode::ExactStateMobilityRejected,
                message: format!(
                    "{} exact state mobility is rejected for normal routing; route activations and keep live state sticky",
                    family.family_id
                ),
            }),
            ExactStateMobility::Untested => {}
        }
    }

    for boundary in boundaries {
        if boundary.decision == BoundaryDecision::Rejected {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Error,
                code: boundary
                    .reason_codes
                    .iter()
                    .copied()
                    .find(|code| *code == PlanReasonCode::SharedKvRegionCut)
                    .unwrap_or(PlanReasonCode::RecurrentStateTransferRejected),
                message: format!(
                    "boundary at layer {} is rejected: {}",
                    boundary.layer_boundary,
                    boundary.messages.join("; ")
                ),
            });
        }
    }

    diagnostics
}

fn validate_request(request: &TopologyPlanRequest) -> Result<(), PlanError> {
    if request.layers.is_empty() {
        return Err(PlanError::EmptyLayers);
    }
    if request.nodes.is_empty() {
        return Err(PlanError::EmptyNodes);
    }
    if let Some(family) = &request.family {
        let found = request.layers.len() as u32;
        if family.layer_count != found {
            return Err(PlanError::FamilyLayerCountMismatch {
                family_id: family.family_id.clone(),
                expected: family.layer_count,
                found,
            });
        }
    }

    for (expected, layer) in (request.layers[0].index..).zip(request.layers.iter()) {
        if layer.index != expected {
            return Err(PlanError::NonContiguousLayers {
                expected,
                found: layer.index,
            });
        }
    }

    Ok(())
}

pub fn qwen3_dense_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "qwen3_dense",
        layer_count,
        activation_width,
        WireValidation::Rejected,
        ExactStateMobility::Accepted,
    )
}

pub fn dense_family_capability(
    family_id: impl Into<String>,
    layer_count: u32,
    activation_width: u32,
    q8_wire_validation: WireValidation,
    exact_state_mobility: ExactStateMobility,
) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: family_id.into(),
        layer_count,
        activation_width,
        default_wire_dtype: WireDType::F16,
        q8_wire_validation,
        exact_state_mobility,
        recurrent_ranges: Vec::new(),
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn llama_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "llama",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Untested,
    )
}

pub fn deepseek2_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "deepseek2",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Untested,
    )
}

pub fn deepseek3_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "deepseek3",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Untested,
    )
}

pub fn glm47_flash_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "glm47_flash",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Untested,
    )
}

pub fn glm4_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "glm4",
        layer_count,
        activation_width,
        WireValidation::Rejected,
        ExactStateMobility::Accepted,
    )
}

pub fn gemma2_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "gemma2",
        layer_count,
        activation_width,
        WireValidation::Validated,
        ExactStateMobility::Accepted,
    )
}

pub fn gemma3_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "gemma3",
        layer_count,
        activation_width,
        WireValidation::Rejected,
        ExactStateMobility::Accepted,
    )
}

pub fn gemma4_a4b_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "gemma4_a4b",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Untested,
    )
}

pub fn olmo_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "olmo",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Untested,
    )
}

pub fn minimax_m27_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "minimax_m27",
        layer_count,
        activation_width,
        WireValidation::Untested,
        ExactStateMobility::Accepted,
    )
}

pub fn falcon_h1_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "falcon_h1".to_string(),
        layer_count,
        activation_width,
        default_wire_dtype: WireDType::F16,
        q8_wire_validation: WireValidation::Untested,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn qwen3next_capability(
    layer_count: u32,
    activation_width: u32,
    recurrent_ranges: Vec<LayerRange>,
) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "qwen3next".to_string(),
        layer_count,
        activation_width,
        default_wire_dtype: WireDType::F16,
        q8_wire_validation: WireValidation::Untested,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges,
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn rwkv6_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "rwkv6".to_string(),
        layer_count,
        activation_width,
        default_wire_dtype: WireDType::F16,
        q8_wire_validation: WireValidation::Untested,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn gemma4_e4b_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "gemma4_e4b".to_string(),
        layer_count,
        activation_width,
        default_wire_dtype: WireDType::F16,
        q8_wire_validation: WireValidation::Rejected,
        exact_state_mobility: ExactStateMobility::Untested,
        recurrent_ranges: Vec::new(),
        split_constraints: vec![SplitConstraint {
            kind: SplitConstraintKind::SharedKvProducerConsumer,
            range: LayerRange { start: 0, end: 0 },
            forbidden_boundaries: vec![12, 14, 24, 28],
            reject_boundary_inside: false,
            reason: "known-bad Gemma4 E4B shared-KV producer/consumer boundary; keep this cut rejected unless KV replay or KV transfer is added".to_string(),
        }],
        sidebands: vec![SidebandRequirement {
            kind: SidebandKind::TokenIds,
            first_required_layer: layer_count,
            reason: "Gemma4 E4B downstream slices require token-id sideband to rebuild the auxiliary per-layer input path".to_string(),
        }],
    }
}

pub fn reviewed_capability_records() -> Vec<ReviewedCapabilityRecord> {
    serde_json::from_str(include_str!(
        "../capabilities/reviewed-family-capabilities.json"
    ))
    .expect("reviewed family capability registry must be valid JSON")
}

pub fn reviewed_capability_for_identity(
    model_identity: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    let normalized = model_identity.to_ascii_lowercase();
    reviewed_capability_records()
        .into_iter()
        .find(|record| reviewed_record_matches(record, &normalized))
        .map(|record| capability_for_request(record.capability, layer_count, activation_width))
}

pub fn infer_family_capability(
    model_identity: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if let Some(capability) =
        reviewed_capability_for_identity(model_identity, layer_count, activation_width)
    {
        return Some(capability);
    }

    let normalized = model_identity.to_ascii_lowercase();
    let compact = normalized.replace(['_', '-', '/', ' '], "");

    if compact.contains("gemma4") && compact.contains("e4b") {
        return Some(gemma4_e4b_capability(layer_count, activation_width));
    }
    if compact.contains("gemma4") && compact.contains("a4b") {
        return Some(gemma4_a4b_capability(layer_count, activation_width));
    }
    if compact.contains("gemma3") {
        return Some(gemma3_capability(layer_count, activation_width));
    }
    if compact.contains("gemma2") {
        return Some(gemma2_capability(layer_count, activation_width));
    }
    if compact.contains("falconh1") {
        return Some(falcon_h1_capability(layer_count, activation_width));
    }
    if compact.contains("minimaxm27") || compact.contains("minimaxm2.7") {
        return Some(minimax_m27_capability(layer_count, activation_width));
    }
    if compact.contains("glm47flash") || compact.contains("glm4.7flash") {
        return Some(glm47_flash_capability(layer_count, activation_width));
    }
    if compact.contains("glm4") {
        return Some(glm4_capability(layer_count, activation_width));
    }
    if compact.contains("deepseekcoderv2")
        || compact.contains("deepseekv2")
        || compact.contains("deepseek2")
    {
        return Some(deepseek2_capability(layer_count, activation_width));
    }
    if compact.contains("deepseekv3") || compact.contains("deepseek3") {
        return Some(deepseek3_capability(layer_count, activation_width));
    }
    if compact.contains("olmo") {
        return Some(olmo_capability(layer_count, activation_width));
    }
    if compact.contains("llama") {
        return Some(llama_capability(layer_count, activation_width));
    }
    if compact.contains("qwen3next") || compact.contains("qwen3codernext") {
        return Some(qwen3next_capability(
            layer_count,
            activation_width,
            vec![LayerRange {
                start: 0,
                end: layer_count,
            }],
        ));
    }
    if compact.contains("rwkv6") {
        return Some(rwkv6_capability(layer_count, activation_width));
    }
    if compact.contains("qwen3") {
        return Some(qwen3_dense_capability(layer_count, activation_width));
    }

    None
}

fn reviewed_record_matches(record: &ReviewedCapabilityRecord, normalized_identity: &str) -> bool {
    [
        record.model_id.as_deref(),
        record.canonical_ref.as_deref(),
        record.distribution_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|value| !value.is_empty() && normalized_identity.contains(&value.to_ascii_lowercase()))
        || match (
            record.source_repo.as_deref(),
            record.source_revision.as_deref(),
            record.source_file.as_deref(),
        ) {
            (Some(repo), Some(revision), Some(file)) => {
                normalized_identity.contains(&repo.to_ascii_lowercase())
                    && normalized_identity.contains(&revision.to_ascii_lowercase())
                    && normalized_identity.contains(&file.to_ascii_lowercase())
            }
            (Some(repo), _, Some(file)) => {
                normalized_identity.contains(&repo.to_ascii_lowercase())
                    && normalized_identity.contains(&file.to_ascii_lowercase())
            }
            _ => false,
        }
}

fn capability_for_request(
    mut capability: FamilyCapabilityRecord,
    layer_count: u32,
    activation_width: u32,
) -> FamilyCapabilityRecord {
    let stored_layer_count = capability.layer_count;
    capability.layer_count = layer_count;
    if activation_width != 0 {
        capability.activation_width = activation_width;
    }
    for range in &mut capability.recurrent_ranges {
        if range.start == 0 && range.end == stored_layer_count {
            range.end = layer_count;
        }
    }
    for sideband in &mut capability.sidebands {
        if sideband.first_required_layer == stored_layer_count {
            sideband.first_required_layer = layer_count;
        }
    }
    capability
}

pub fn dense_attention_layers(count: u32, parameter_bytes: u64) -> Vec<LayerSpec> {
    (0..count)
        .map(|index| LayerSpec {
            index,
            attention: true,
            recurrent: false,
            parameter_bytes,
        })
        .collect()
}

pub fn falcon_h1_layers(count: u32, parameter_bytes: u64) -> Vec<LayerSpec> {
    (0..count)
        .map(|index| LayerSpec {
            index,
            attention: true,
            recurrent: true,
            parameter_bytes,
        })
        .collect()
}

pub fn qwen3next_layers(
    count: u32,
    recurrent_layers: impl IntoIterator<Item = u32>,
    parameter_bytes: u64,
) -> Vec<LayerSpec> {
    let recurrent_layers: std::collections::BTreeSet<u32> = recurrent_layers.into_iter().collect();
    (0..count)
        .map(|index| {
            let recurrent = recurrent_layers.contains(&index);
            LayerSpec {
                index,
                attention: !recurrent,
                recurrent,
                parameter_bytes,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
