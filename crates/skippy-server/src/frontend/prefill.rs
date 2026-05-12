use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PrefillChunkSchedule {
    pub(super) sizes: Vec<usize>,
}

impl PrefillChunkSchedule {
    pub(super) fn parse(spec: Option<&str>) -> Result<Option<Self>> {
        let Some(spec) = spec else {
            return Ok(None);
        };
        let spec = spec.trim();
        if spec.is_empty() {
            return Ok(None);
        }
        let mut sizes = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                bail!("empty chunk size in schedule");
            }
            let size = part
                .parse::<usize>()
                .with_context(|| format!("invalid chunk size '{part}'"))?;
            if size == 0 {
                bail!("chunk sizes must be greater than zero");
            }
            sizes.push(size);
        }
        Ok(Some(Self { sizes }))
    }

    pub(super) fn chunk_size_for(&self, chunk_index: usize) -> usize {
        self.sizes
            .get(chunk_index)
            .copied()
            .or_else(|| self.sizes.last().copied())
            .expect("schedule has at least one size")
    }

    pub(super) fn label(&self) -> String {
        self.sizes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PrefillChunkPolicy {
    Fixed {
        chunk_size: usize,
    },
    Schedule {
        fixed_chunk_size: usize,
        schedule: PrefillChunkSchedule,
    },
    AdaptiveRamp {
        fixed_chunk_size: usize,
        start: usize,
        step: usize,
        max: usize,
    },
}

pub(super) struct PrefillChunkPolicyArgs<'a> {
    pub(super) policy: &'a str,
    pub(super) schedule: Option<&'a str>,
    pub(super) fixed_chunk_size: usize,
    pub(super) adaptive_start: usize,
    pub(super) adaptive_step: usize,
    pub(super) adaptive_max: usize,
    pub(super) schedule_arg: &'static str,
    pub(super) policy_arg: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PrefillChunkObservation {
    pub(super) compute_ms: f64,
    pub(super) forward_write_ms: f64,
    pub(super) downstream_wait_ms: f64,
}

#[derive(Clone, Debug)]
pub(super) struct PrefillChunkPlanner {
    pub(super) policy: PrefillChunkPolicy,
    pub(super) next_adaptive_size: usize,
}

impl PrefillChunkPolicy {
    pub(super) fn parse(args: PrefillChunkPolicyArgs<'_>) -> Result<Self> {
        if args.fixed_chunk_size == 0 {
            bail!("prefill chunk size must be greater than zero");
        }
        let normalized = args.policy.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "fixed" => {
                if let Some(schedule) = PrefillChunkSchedule::parse(args.schedule)
                    .with_context(|| format!("invalid {} value", args.schedule_arg))?
                {
                    return Ok(Self::Schedule {
                        fixed_chunk_size: args.fixed_chunk_size,
                        schedule,
                    });
                }
                Ok(Self::Fixed {
                    chunk_size: args.fixed_chunk_size,
                })
            }
            "schedule" => {
                let schedule = PrefillChunkSchedule::parse(args.schedule)
                    .with_context(|| format!("invalid {} value", args.schedule_arg))?
                    .ok_or_else(|| anyhow!("{} requires {}", args.policy_arg, args.schedule_arg))?;
                Ok(Self::Schedule {
                    fixed_chunk_size: args.fixed_chunk_size,
                    schedule,
                })
            }
            "adaptive" | "adaptive-ramp" => {
                if args.adaptive_start == 0
                    || args.adaptive_step == 0
                    || args.adaptive_max == 0
                    || args.adaptive_start > args.adaptive_max
                {
                    bail!(
                        "{} adaptive-ramp requires positive start/step/max with start <= max",
                        args.policy_arg
                    );
                }
                Ok(Self::AdaptiveRamp {
                    fixed_chunk_size: args.fixed_chunk_size,
                    start: args.adaptive_start,
                    step: args.adaptive_step,
                    max: args.adaptive_max,
                })
            }
            other => bail!(
                "invalid {} '{}'; expected fixed, schedule, or adaptive-ramp",
                args.policy_arg,
                other
            ),
        }
    }

    pub(super) fn planner(&self) -> PrefillChunkPlanner {
        let next_adaptive_size = match self {
            Self::AdaptiveRamp { start, .. } => *start,
            _ => 0,
        };
        PrefillChunkPlanner {
            policy: self.clone(),
            next_adaptive_size,
        }
    }

    pub(super) fn policy_label(&self) -> &'static str {
        match self {
            Self::Fixed { .. } => "fixed",
            Self::Schedule { .. } => "schedule",
            Self::AdaptiveRamp { .. } => "adaptive-ramp",
        }
    }

    pub(super) fn fixed_chunk_size(&self) -> usize {
        match self {
            Self::Fixed { chunk_size } => *chunk_size,
            Self::Schedule {
                fixed_chunk_size, ..
            }
            | Self::AdaptiveRamp {
                fixed_chunk_size, ..
            } => *fixed_chunk_size,
        }
    }

    pub(super) fn schedule(&self) -> Option<&PrefillChunkSchedule> {
        match self {
            Self::Schedule { schedule, .. } => Some(schedule),
            _ => None,
        }
    }

    pub(super) fn adaptive_params(&self) -> Option<(usize, usize, usize)> {
        match self {
            Self::AdaptiveRamp {
                start, step, max, ..
            } => Some((*start, *step, *max)),
            _ => None,
        }
    }
}

impl PrefillChunkPlanner {
    pub(super) fn chunk_size_for(&mut self, chunk_index: usize) -> usize {
        match &self.policy {
            PrefillChunkPolicy::Fixed { chunk_size } => *chunk_size,
            PrefillChunkPolicy::Schedule { schedule, .. } => schedule.chunk_size_for(chunk_index),
            PrefillChunkPolicy::AdaptiveRamp { .. } => self.next_adaptive_size,
        }
    }

    pub(super) fn observe(&mut self, observation: PrefillChunkObservation) {
        let PrefillChunkPolicy::AdaptiveRamp {
            start, step, max, ..
        } = &self.policy
        else {
            return;
        };
        let compute_ms = observation.compute_ms.max(0.001);
        let downstream_hidden = observation.downstream_wait_ms <= compute_ms * 0.75
            && observation.forward_write_ms <= compute_ms * 0.25;
        let downstream_exposed = observation.downstream_wait_ms > compute_ms * 1.25;
        if downstream_hidden {
            self.next_adaptive_size = self.next_adaptive_size.saturating_add(*step).min(*max);
        } else if downstream_exposed {
            self.next_adaptive_size = self.next_adaptive_size.saturating_sub(*step).max(*start);
        }
    }

    pub(super) fn advance_without_observation(&mut self) {
        let PrefillChunkPolicy::AdaptiveRamp { step, max, .. } = &self.policy else {
            return;
        };
        self.next_adaptive_size = self.next_adaptive_size.saturating_add(*step).min(*max);
    }
}
