fn parse_stage_ranges(splits: &str, layer_end: u32) -> Result<Vec<(u32, u32)>> {
    if layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    let mut boundaries = Vec::new();
    for split in splits.split(',') {
        let split = split.trim();
        if split.is_empty() {
            continue;
        }
        boundaries.push(
            split
                .parse::<u32>()
                .with_context(|| format!("parse split boundary {split:?}"))?,
        );
    }
    let mut ranges = Vec::with_capacity(boundaries.len() + 1);
    let mut start = 0;
    for boundary in boundaries {
        if boundary <= start || boundary >= layer_end {
            bail!("invalid split boundary {boundary} for layer_end {layer_end}");
        }
        ranges.push((start, boundary));
        start = boundary;
    }
    ranges.push((start, layer_end));
    Ok(ranges)
}

fn validate_prompt_topology_plan(
    args: &PromptArgs,
    layer_end: u32,
    ranges: &[(u32, u32)],
) -> Result<()> {
    let identity = format!("{} {}", args.model_id, args.model_path.display());
    let activation_width =
        u32::try_from(args.activation_width).context("activation_width must be non-negative")?;
    let family = infer_family_capability(&identity, layer_end, activation_width);
    let nodes = if args.hosts.is_empty() {
        (0..ranges.len())
            .map(|index| NodeSpec {
                node_id: format!("local-stage-{index}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect()
    } else {
        args.hosts
            .iter()
            .map(|host| NodeSpec {
                node_id: host.clone(),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect()
    };
    let request = TopologyPlanRequest {
        topology_id: "local-binary-kv-repl".to_string(),
        model_id: args.model_id.clone(),
        layers: dense_attention_layers(layer_end, 0),
        nodes,
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let splits = split_boundaries_from_ranges(ranges);
    let plan = plan_contiguous_with_splits(&request, &splits).context("topology planner failed")?;

    if args.activation_wire_dtype.eq_ignore_ascii_case("q8") {
        match family.as_ref().map(|family| family.q8_wire_validation) {
            Some(WireValidation::Validated) => {}
            Some(WireValidation::Rejected) => {
                bail!("topology planner rejected q8 activation wire dtype for {}; use f16 or add a passing q8 correctness record", args.model_id);
            }
            Some(WireValidation::Untested) => {
                bail!("topology planner has no q8 validation for {}; use f16 until this family/split passes correctness", args.model_id);
            }
            None => {}
        }
    }

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn resolve_stage_ranges(
    single_stage: bool,
    splits: Option<&str>,
    default_stage_count: usize,
    layer_end: u32,
) -> Result<Vec<(u32, u32)>> {
    if layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    if single_stage {
        return Ok(vec![(0, layer_end)]);
    }
    match splits {
        Some(splits) => parse_stage_ranges(splits, layer_end),
        None => even_stage_ranges(default_stage_count, layer_end),
    }
}

fn split_boundaries_from_ranges(ranges: &[(u32, u32)]) -> Vec<u32> {
    ranges
        .iter()
        .take(ranges.len().saturating_sub(1))
        .map(|(_, end)| *end)
        .collect()
}

fn even_stage_ranges(stage_count: usize, layer_end: u32) -> Result<Vec<(u32, u32)>> {
    if stage_count == 0 {
        bail!("stage count must be greater than zero");
    }
    if u32::try_from(stage_count).context("stage count exceeds u32")? > layer_end {
        bail!("stage count {stage_count} exceeds layer_end {layer_end}");
    }

    let layer_end = usize::try_from(layer_end).context("layer_end exceeds usize")?;
    let base = layer_end / stage_count;
    let remainder = layer_end % stage_count;
    let mut ranges = Vec::with_capacity(stage_count);
    let mut start = 0usize;
    for index in 0..stage_count {
        let width = base + usize::from(index < remainder);
        let end = start + width;
        ranges.push((
            u32::try_from(start).context("layer range start exceeds u32")?,
            u32::try_from(end).context("layer range end exceeds u32")?,
        ));
        start = end;
    }
    Ok(ranges)
}

fn wait_for_socket(socket_path: &Path, timeout_secs: u64) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    loop {
        if socket_path.exists() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for socket {}", socket_path.display());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}
