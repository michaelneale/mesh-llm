use std::{collections::BTreeMap, fmt};

#[derive(Clone, Copy)]
pub struct NgramConfig {
    pub n: usize,
    pub min_hits: u32,
    pub max_tokens_per_pool: usize,
}

pub struct NgramStore {
    config: NgramConfig,
    pools: BTreeMap<String, NgramPool>,
}

#[derive(Default)]
struct NgramPool {
    tokens: Vec<i32>,
    policy: NgramPolicyState,
}

#[derive(Clone, Copy)]
struct NgramPolicyState {
    enabled: bool,
    cooldown_remaining: usize,
    current_window: usize,
    windows: usize,
    skipped_single_token_windows: usize,
    policy_window_cap_probes: usize,
    auto_repeated_suffix_hits: usize,
    proposed_tokens: usize,
    accepted_tokens: usize,
    rejected_windows: usize,
    early_reject_windows: usize,
    unproductive_windows: usize,
    enable_count: usize,
    disable_count: usize,
    grow_count: usize,
    shrink_count: usize,
    verify_elapsed_ms: f64,
    repair_elapsed_ms: f64,
    non_spec_decode_elapsed_ms: f64,
    non_spec_decode_tokens: usize,
    last_reason: &'static str,
}

impl Default for NgramPolicyState {
    fn default() -> Self {
        Self {
            enabled: false,
            cooldown_remaining: 0,
            current_window: 1,
            windows: 0,
            skipped_single_token_windows: 0,
            policy_window_cap_probes: 0,
            auto_repeated_suffix_hits: 0,
            proposed_tokens: 0,
            accepted_tokens: 0,
            rejected_windows: 0,
            early_reject_windows: 0,
            unproductive_windows: 0,
            enable_count: 0,
            disable_count: 0,
            grow_count: 0,
            shrink_count: 0,
            verify_elapsed_ms: 0.0,
            repair_elapsed_ms: 0.0,
            non_spec_decode_elapsed_ms: 0.0,
            non_spec_decode_tokens: 0,
            last_reason: "new_pool",
        }
    }
}

impl NgramPolicyState {
    fn accept_rate(&self) -> f64 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.proposed_tokens as f64
        }
    }

    fn quality_label(&self) -> &'static str {
        if !self.enabled || self.cooldown_remaining > 0 {
            "disabled"
        } else if self.proposed_tokens >= 16 && self.accept_rate() >= 0.8 {
            "high_accept"
        } else if self.proposed_tokens >= 8 && self.accept_rate() < 0.45 {
            "low_accept"
        } else {
            "probing"
        }
    }

    fn policy_cap_probe_period(&self) -> usize {
        if self.quality_label() == "high_accept" {
            2
        } else {
            4
        }
    }
}

#[derive(Clone)]
pub struct NgramProposal {
    pub tokens: Vec<i32>,
    pub policy: NgramPolicySnapshot,
    pub stop_reason: &'static str,
    pub match_order_min: usize,
    pub match_order_max: usize,
    pub configured_order: usize,
}

#[derive(Clone, Copy)]
pub enum NgramActivation {
    Manual {
        stable_user: bool,
    },
    Auto {
        prompt_candidate: bool,
        min_repeated_suffix_hits: usize,
    },
}

struct NgramPoolProposal {
    tokens: Vec<i32>,
    stop_reason: &'static str,
    match_order_min: usize,
    match_order_max: usize,
}

struct NgramNextToken {
    token: i32,
    order: usize,
}

enum NgramMatchResult {
    Match(NgramNextToken),
    NoSuffixMatch,
    MinHitsFiltered,
}

#[derive(Clone, Copy)]
pub struct NgramPolicySnapshot {
    pub enabled: bool,
    pub cooldown_remaining: usize,
    pub current_window: usize,
    pub windows: usize,
    pub skipped_single_token_windows: usize,
    pub policy_window_cap_probes: usize,
    pub auto_repeated_suffix_hits: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_windows: usize,
    pub early_reject_windows: usize,
    pub unproductive_windows: usize,
    pub enable_count: usize,
    pub disable_count: usize,
    pub grow_count: usize,
    pub shrink_count: usize,
    pub accept_rate: f64,
    pub verify_ms_per_accepted: f64,
    pub repair_ms_per_accepted: f64,
    pub non_spec_decode_ms_per_token: f64,
    pub quality: &'static str,
    pub reason: &'static str,
}

impl NgramPolicySnapshot {
    fn from_state(state: &NgramPolicyState) -> Self {
        Self {
            enabled: state.enabled,
            cooldown_remaining: state.cooldown_remaining,
            current_window: state.current_window,
            windows: state.windows,
            skipped_single_token_windows: state.skipped_single_token_windows,
            policy_window_cap_probes: state.policy_window_cap_probes,
            auto_repeated_suffix_hits: state.auto_repeated_suffix_hits,
            proposed_tokens: state.proposed_tokens,
            accepted_tokens: state.accepted_tokens,
            rejected_windows: state.rejected_windows,
            early_reject_windows: state.early_reject_windows,
            unproductive_windows: state.unproductive_windows,
            enable_count: state.enable_count,
            disable_count: state.disable_count,
            grow_count: state.grow_count,
            shrink_count: state.shrink_count,
            accept_rate: state.accept_rate(),
            verify_ms_per_accepted: if state.accepted_tokens == 0 {
                0.0
            } else {
                state.verify_elapsed_ms / state.accepted_tokens as f64
            },
            repair_ms_per_accepted: if state.accepted_tokens == 0 {
                0.0
            } else {
                state.repair_elapsed_ms / state.accepted_tokens as f64
            },
            non_spec_decode_ms_per_token: if state.non_spec_decode_tokens == 0 {
                0.0
            } else {
                state.non_spec_decode_elapsed_ms / state.non_spec_decode_tokens as f64
            },
            quality: state.quality_label(),
            reason: state.last_reason,
        }
    }

    pub fn should_skip_single_token_verify(
        self,
        match_order_max: usize,
        configured_order: usize,
    ) -> bool {
        if self.current_window > 1 || self.windows == 0 {
            return false;
        }
        if match_order_max < configured_order {
            return true;
        }
        self.skipped_single_token_windows % 16 != 15
    }
}

impl NgramStore {
    pub fn new(config: NgramConfig) -> Self {
        Self {
            config,
            pools: BTreeMap::new(),
        }
    }

    pub fn observe_sequence(&mut self, pool_key: &str, tokens: &[i32]) {
        if tokens.is_empty() {
            return;
        }
        let pool = self.pools.entry(pool_key.to_string()).or_default();
        pool.tokens.extend_from_slice(tokens);
        if pool.tokens.len() > self.config.max_tokens_per_pool {
            let excess = pool.tokens.len() - self.config.max_tokens_per_pool;
            pool.tokens.drain(..excess);
        }
    }

    pub fn observe_decode_step(
        &mut self,
        pool_key: &str,
        elapsed_ms: f64,
    ) -> Option<NgramPolicySnapshot> {
        let pool = self.pools.get_mut(pool_key)?;
        pool.policy.non_spec_decode_elapsed_ms += elapsed_ms.max(0.0);
        pool.policy.non_spec_decode_tokens += 1;
        Some(NgramPolicySnapshot::from_state(&pool.policy))
    }

    pub fn propose(
        &mut self,
        pool_key: &str,
        context_tokens: &[i32],
        max_tokens: usize,
        has_stable_user: bool,
    ) -> NgramProposal {
        self.propose_with_activation(
            pool_key,
            context_tokens,
            max_tokens,
            NgramActivation::Manual {
                stable_user: has_stable_user,
            },
        )
    }

    pub fn propose_with_activation(
        &mut self,
        pool_key: &str,
        context_tokens: &[i32],
        max_tokens: usize,
        activation: NgramActivation,
    ) -> NgramProposal {
        let Some(pool_view) = self.pools.get(pool_key) else {
            return NgramProposal {
                tokens: Vec::new(),
                policy: NgramPolicySnapshot::from_state(&NgramPolicyState {
                    last_reason: "no_pool",
                    ..NgramPolicyState::default()
                }),
                stop_reason: "no_pool",
                match_order_min: 0,
                match_order_max: 0,
                configured_order: self.config.n,
            };
        };
        let proposal = self.propose_from_pool(pool_view, context_tokens, max_tokens);
        let mut tokens = proposal.tokens;
        let mut stop_reason = proposal.stop_reason;
        let Some(pool) = self.pools.get_mut(pool_key) else {
            unreachable!("pool was just read");
        };
        if pool.policy.cooldown_remaining > 0 {
            pool.policy.cooldown_remaining -= 1;
            pool.policy.last_reason = "cooldown";
            return NgramProposal {
                tokens: Vec::new(),
                policy: NgramPolicySnapshot::from_state(&pool.policy),
                stop_reason: "cooldown",
                match_order_min: 0,
                match_order_max: 0,
                configured_order: self.config.n,
            };
        }
        if !pool.policy.enabled {
            match activation {
                NgramActivation::Manual { stable_user } => {
                    if stable_user {
                        pool.policy.enabled = true;
                        pool.policy.enable_count += 1;
                        pool.policy.last_reason = "enabled_stable_user";
                    } else if tokens.is_empty() {
                        pool.policy.last_reason = "probing_no_repeated_suffix";
                        return NgramProposal {
                            tokens,
                            policy: NgramPolicySnapshot::from_state(&pool.policy),
                            stop_reason,
                            match_order_min: proposal.match_order_min,
                            match_order_max: proposal.match_order_max,
                            configured_order: self.config.n,
                        };
                    } else {
                        pool.policy.enabled = true;
                        pool.policy.enable_count += 1;
                        pool.policy.last_reason = "enabled_repeated_suffix";
                    }
                }
                NgramActivation::Auto {
                    prompt_candidate,
                    min_repeated_suffix_hits,
                } => {
                    let min_auto_hits = min_repeated_suffix_hits.max(1);
                    if prompt_candidate && tokens.len() >= 2 {
                        pool.policy.enabled = true;
                        pool.policy.enable_count += 1;
                        pool.policy.last_reason = "enabled_prompt_shape";
                    } else if tokens.is_empty() {
                        pool.policy.last_reason = if prompt_candidate {
                            "cold_observe_prompt_shape"
                        } else {
                            "cold_observe"
                        };
                        return NgramProposal {
                            tokens,
                            policy: NgramPolicySnapshot::from_state(&pool.policy),
                            stop_reason,
                            match_order_min: proposal.match_order_min,
                            match_order_max: proposal.match_order_max,
                            configured_order: self.config.n,
                        };
                    } else {
                        pool.policy.auto_repeated_suffix_hits += 1;
                        let required_hits = if prompt_candidate && tokens.len() == 1 {
                            min_auto_hits.max(2)
                        } else {
                            min_auto_hits
                        };
                        if pool.policy.auto_repeated_suffix_hits < required_hits {
                            pool.policy.last_reason = if prompt_candidate {
                                "cold_observe_prompt_shape"
                            } else {
                                "cold_observe_repeated_suffix"
                            };
                            return NgramProposal {
                                tokens: Vec::new(),
                                policy: NgramPolicySnapshot::from_state(&pool.policy),
                                stop_reason: "auto_warmup",
                                match_order_min: proposal.match_order_min,
                                match_order_max: proposal.match_order_max,
                                configured_order: self.config.n,
                            };
                        }
                        pool.policy.enabled = true;
                        pool.policy.enable_count += 1;
                        pool.policy.last_reason = "enabled_repeated_suffix";
                    }
                }
            }
        }
        if let NgramActivation::Auto { .. } = activation {
            if pool.policy.enabled && !tokens.is_empty() {
                pool.policy.auto_repeated_suffix_hits += 1;
            }
        }
        let policy_window = pool.policy.current_window.max(1);
        if tokens.len() > policy_window {
            let cap_probe_window = if policy_window == 1
                && pool.policy.windows > 0
                && proposal.match_order_max >= self.config.n
                && tokens.len() >= 2
                && pool.policy.policy_window_cap_probes % pool.policy.policy_cap_probe_period() == 0
            {
                2
            } else {
                policy_window
            };
            if cap_probe_window > policy_window {
                pool.policy.policy_window_cap_probes += 1;
                pool.policy.last_reason = "probe_policy_window_cap";
                tokens.truncate(cap_probe_window);
                stop_reason = "policy_window_cap_probe";
            } else {
                if pool.policy.windows > 0 {
                    pool.policy.policy_window_cap_probes += 1;
                }
                tokens.truncate(policy_window);
                stop_reason = "policy_window_cap";
            }
        }
        if let NgramActivation::Auto { .. } = activation {
            let allow_single_token_auto = pool.policy.auto_repeated_suffix_hits >= 4
                || (pool.policy.windows >= 4 && pool.policy.accept_rate() >= 0.75);
            if tokens.len() == 1 && policy_window == 1 && !allow_single_token_auto {
                pool.policy.last_reason = "auto_single_token_hold";
                return NgramProposal {
                    tokens: Vec::new(),
                    policy: NgramPolicySnapshot::from_state(&pool.policy),
                    stop_reason: "auto_single_token_hold",
                    match_order_min: proposal.match_order_min,
                    match_order_max: proposal.match_order_max,
                    configured_order: self.config.n,
                };
            }
        }
        if tokens.is_empty() {
            pool.policy.last_reason = "enabled_no_repeated_suffix";
        }
        NgramProposal {
            tokens,
            policy: NgramPolicySnapshot::from_state(&pool.policy),
            stop_reason,
            match_order_min: proposal.match_order_min,
            match_order_max: proposal.match_order_max,
            configured_order: self.config.n,
        }
    }

    pub fn observe_window(
        &mut self,
        pool_key: &str,
        decision: VerifySpanDecision,
        proposed_tokens: usize,
        verify_elapsed_ms: f64,
        repair_elapsed_ms: f64,
        max_window: usize,
    ) -> Option<NgramPolicySnapshot> {
        let pool = self.pools.get_mut(pool_key)?;
        let policy = &mut pool.policy;
        policy.windows += 1;
        policy.proposed_tokens += proposed_tokens;
        policy.accepted_tokens += decision.accepted_before_reject;
        policy.verify_elapsed_ms += verify_elapsed_ms;
        policy.repair_elapsed_ms += repair_elapsed_ms;
        let total_spec_cost_ms = verify_elapsed_ms + repair_elapsed_ms;
        let non_spec_ms_per_token = if policy.non_spec_decode_tokens == 0 {
            None
        } else {
            Some(policy.non_spec_decode_elapsed_ms / policy.non_spec_decode_tokens as f64)
        };
        let accepted = decision.accepted_before_reject;
        let unproductive = accepted == 0;
        if unproductive {
            policy.unproductive_windows += 1;
        }
        if decision.rejected() {
            policy.rejected_windows += 1;
        }
        match decision.kind {
            VerifySpanDecisionKind::FullAccept | VerifySpanDecisionKind::TailReject => {
                if unproductive {
                    let next_window = policy.current_window.saturating_sub(1).max(1);
                    if next_window < policy.current_window {
                        policy.current_window = next_window;
                        policy.shrink_count += 1;
                    }
                    policy.last_reason = "shrunk_zero_accept_tail_reject";
                } else {
                    let efficient = non_spec_ms_per_token.is_none_or(|baseline| {
                        total_spec_cost_ms / accepted as f64 <= baseline * 1.15
                    });
                    if efficient {
                        let grow_by = if decision.kind == VerifySpanDecisionKind::FullAccept
                            && accepted >= 2
                        {
                            2
                        } else {
                            1
                        };
                        let next_window = policy
                            .current_window
                            .saturating_add(grow_by)
                            .min(max_window.max(1));
                        if next_window > policy.current_window {
                            policy.current_window = next_window;
                            policy.grow_count += 1;
                            policy.last_reason = if grow_by > 1 {
                                "grown_full_accept_fast"
                            } else {
                                "grown_acceptance"
                            };
                        } else {
                            policy.last_reason = "kept_acceptance";
                        }
                    } else {
                        policy.last_reason = "kept_costly_acceptance";
                    }
                }
            }
            VerifySpanDecisionKind::AcceptedStop | VerifySpanDecisionKind::EarlyRejectStop => {
                policy.last_reason = "kept_stop";
            }
            VerifySpanDecisionKind::EarlyReject => {
                policy.early_reject_windows += 1;
                let next_window = policy
                    .current_window
                    .saturating_sub(1)
                    .max(decision.repair_input_count.unwrap_or(1))
                    .max(1);
                if next_window < policy.current_window {
                    policy.current_window = next_window;
                    policy.shrink_count += 1;
                    policy.last_reason = "shrunk_early_reject";
                } else {
                    policy.last_reason = "kept_early_reject";
                }
                if decision.accepted_before_reject == 0 {
                    policy.enabled = false;
                    policy.cooldown_remaining = 8;
                    policy.disable_count += 1;
                    policy.last_reason = "disabled_zero_accept_early_reject";
                }
            }
        }
        if let Some(baseline) = non_spec_ms_per_token {
            let accepted_or_commit = accepted.max(decision.commit_count).max(1);
            let spec_ms_per_token = total_spec_cost_ms / accepted_or_commit as f64;
            if policy.enabled && spec_ms_per_token > baseline * 1.35 {
                let next_window = policy.current_window.saturating_sub(1).max(1);
                if next_window < policy.current_window {
                    policy.current_window = next_window;
                    policy.shrink_count += 1;
                }
                if accepted == 0 {
                    policy.enabled = false;
                    policy.cooldown_remaining = 8;
                    policy.disable_count += 1;
                    policy.last_reason = "disabled_costly_zero_yield";
                } else {
                    policy.last_reason = "shrunk_costly_window";
                }
            }
        }
        let accept_rate = policy.accept_rate();
        if policy.enabled
            && policy.windows >= 3
            && policy.proposed_tokens >= 8
            && accept_rate < 0.45
        {
            policy.enabled = false;
            policy.cooldown_remaining = 12;
            policy.disable_count += 1;
            policy.last_reason = "disabled_low_accept_rate_early";
        } else if policy.enabled && policy.windows >= 4 && accept_rate < 0.35 {
            policy.enabled = false;
            policy.cooldown_remaining = 8;
            policy.disable_count += 1;
            policy.last_reason = "disabled_low_accept_rate";
        }
        Some(NgramPolicySnapshot::from_state(policy))
    }

    pub fn observe_skipped_single_token_verify(
        &mut self,
        pool_key: &str,
    ) -> Option<NgramPolicySnapshot> {
        let pool = self.pools.get_mut(pool_key)?;
        pool.policy.skipped_single_token_windows += 1;
        if pool.policy.enabled
            && pool.policy.current_window == 1
            && pool.policy.skipped_single_token_windows >= 8
            && (pool.policy.accepted_tokens == 0 || pool.policy.accept_rate() < 0.55)
        {
            pool.policy.enabled = false;
            pool.policy.cooldown_remaining = 16;
            pool.policy.disable_count += 1;
            pool.policy.last_reason = "disabled_single_token_stall";
        } else {
            pool.policy.last_reason = "skipped_single_token_verify";
        }
        Some(NgramPolicySnapshot::from_state(&pool.policy))
    }

    fn propose_from_pool(
        &self,
        pool: &NgramPool,
        context_tokens: &[i32],
        max_tokens: usize,
    ) -> NgramPoolProposal {
        if max_tokens == 0 {
            return NgramPoolProposal {
                tokens: Vec::new(),
                stop_reason: "proposal_limit",
                match_order_min: 0,
                match_order_max: 0,
            };
        }
        if context_tokens.is_empty() {
            return NgramPoolProposal {
                tokens: Vec::new(),
                stop_reason: "context_too_short",
                match_order_min: 0,
                match_order_max: 0,
            };
        }
        if pool.tokens.len() <= 1 {
            return NgramPoolProposal {
                tokens: Vec::new(),
                stop_reason: "history_too_short",
                match_order_min: 0,
                match_order_max: 0,
            };
        }

        let mut context = context_tokens.to_vec();
        let mut proposals = Vec::with_capacity(max_tokens);
        let mut min_order = usize::MAX;
        let mut max_order = 0usize;
        let mut stop_reason = "proposal_limit";
        for _ in 0..max_tokens {
            match self.best_next_token(pool, &context) {
                NgramMatchResult::Match(next) => {
                    min_order = min_order.min(next.order);
                    max_order = max_order.max(next.order);
                    proposals.push(next.token);
                    context.push(next.token);
                }
                NgramMatchResult::NoSuffixMatch => {
                    stop_reason = "no_suffix_match";
                    break;
                }
                NgramMatchResult::MinHitsFiltered => {
                    stop_reason = "min_hits_filtered";
                    break;
                }
            }
        }
        if proposals.is_empty() {
            min_order = 0;
        }
        NgramPoolProposal {
            tokens: proposals,
            stop_reason,
            match_order_min: min_order,
            match_order_max: max_order,
        }
    }

    fn best_next_token(&self, pool: &NgramPool, context_tokens: &[i32]) -> NgramMatchResult {
        let max_order = self
            .config
            .n
            .min(context_tokens.len())
            .min(pool.tokens.len().saturating_sub(1));
        if max_order == 0 {
            return NgramMatchResult::NoSuffixMatch;
        }
        let mut saw_filtered = false;
        for order in (1..=max_order).rev() {
            match self.best_next_token_for_order(pool, context_tokens, order) {
                NgramMatchResult::Match(next) => return NgramMatchResult::Match(next),
                NgramMatchResult::MinHitsFiltered => saw_filtered = true,
                NgramMatchResult::NoSuffixMatch => {}
            }
        }
        if saw_filtered {
            NgramMatchResult::MinHitsFiltered
        } else {
            NgramMatchResult::NoSuffixMatch
        }
    }

    fn best_next_token_for_order(
        &self,
        pool: &NgramPool,
        context_tokens: &[i32],
        order: usize,
    ) -> NgramMatchResult {
        let Some(suffix) = context_tokens.get(context_tokens.len().saturating_sub(order)..) else {
            return NgramMatchResult::NoSuffixMatch;
        };
        let mut counts = BTreeMap::<i32, u32>::new();
        for start in 0..pool.tokens.len().saturating_sub(order) {
            if &pool.tokens[start..start + order] == suffix {
                let next = pool.tokens[start + order];
                *counts.entry(next).or_default() += 1;
            }
        }
        if counts.is_empty() {
            return NgramMatchResult::NoSuffixMatch;
        }
        counts
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_hits)
            .max_by_key(|(token, count)| (*count, std::cmp::Reverse(*token)))
            .map(|(token, _)| NgramMatchResult::Match(NgramNextToken { token, order }))
            .unwrap_or(NgramMatchResult::MinHitsFiltered)
    }
}

pub fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len());
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    tokens
}

pub fn proposal_len_bucket(len: usize) -> &'static str {
    match len {
        0 => "0",
        1 => "1",
        2 => "2",
        3 => "3",
        _ => "4+",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifySpanDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifySpanDecision {
    pub kind: VerifySpanDecisionKind,
    pub accepted_before_reject: usize,
    pub repair_input_count: Option<usize>,
    pub commit_count: usize,
}

impl VerifySpanDecision {
    pub fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifySpanDecisionKind::TailReject
                | VerifySpanDecisionKind::EarlyReject
                | VerifySpanDecisionKind::EarlyRejectStop
        )
    }

    pub fn requires_repair(self) -> bool {
        self.kind == VerifySpanDecisionKind::EarlyReject
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifySpanRepairStrategy {
    None,
    RestoreDecodeOne,
    RestoreReverify,
}

impl VerifySpanRepairStrategy {
    pub fn for_decision(
        decision: VerifySpanDecision,
        proposal_source: &str,
        policy: Option<NgramPolicySnapshot>,
        verify_elapsed_ms: f64,
        verified_tokens: usize,
    ) -> Self {
        if !decision.requires_repair() {
            return Self::None;
        }
        let Some(repair_input_count) = decision.repair_input_count else {
            return Self::RestoreReverify;
        };
        if repair_input_count == 1 {
            return Self::RestoreDecodeOne;
        }
        if proposal_source == "ngram" {
            if let Some(policy) = policy {
                let baseline = policy.non_spec_decode_ms_per_token;
                let verify_ms_per_token = verify_elapsed_ms / verified_tokens.max(1) as f64;
                let estimated_reverify_ms = verify_ms_per_token * repair_input_count as f64;
                if baseline > 0.0 && estimated_reverify_ms > baseline * 1.15 {
                    return Self::RestoreDecodeOne;
                }
            }
        }
        Self::RestoreReverify
    }
}

pub fn repair_path_label(
    decision: VerifySpanDecision,
    strategy: VerifySpanRepairStrategy,
) -> &'static str {
    match decision.kind {
        VerifySpanDecisionKind::FullAccept => "none_full_accept",
        VerifySpanDecisionKind::AcceptedStop => "none_accepted_stop",
        VerifySpanDecisionKind::TailReject => "none_tail_reject",
        VerifySpanDecisionKind::EarlyReject => match strategy {
            VerifySpanRepairStrategy::None => "none_early_reject",
            VerifySpanRepairStrategy::RestoreDecodeOne => {
                if decision.repair_input_count == Some(1) {
                    "restore_decode"
                } else {
                    "restore_decode_one_cost"
                }
            }
            VerifySpanRepairStrategy::RestoreReverify => "restore_reverify",
        },
        VerifySpanDecisionKind::EarlyRejectStop => "none_early_reject_stop",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifySpanClassifyError<E> {
    Token(E),
    TooFewTokens { got: usize, expected: usize },
}

impl<E: fmt::Display> fmt::Display for VerifySpanClassifyError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Token(error) => write!(f, "{error}"),
            Self::TooFewTokens { got, expected } => write!(
                f,
                "verify span returned too few tokens: got {got} expected {expected}"
            ),
        }
    }
}

impl<E> std::error::Error for VerifySpanClassifyError<E> where E: fmt::Debug + fmt::Display {}

pub fn classify_verify_span<F, E>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> Result<VerifySpanDecision, VerifySpanClassifyError<E>>
where
    F: FnMut(i32) -> Result<bool, E>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        return Err(VerifySpanClassifyError::TooFewTokens {
            got: predicted_tokens.len(),
            expected: draft_tokens.len(),
        });
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted).map_err(VerifySpanClassifyError::Token)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if (reached_eog || reached_limit) && commit_count < draft_tokens.len() {
                return Ok(VerifySpanDecision {
                    kind: VerifySpanDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    repair_input_count: None,
                    commit_count,
                });
            }
            continue;
        }

        let repair_input_count = accepted_before_reject + 1;
        let kind = if repair_input_count == draft_tokens.len() {
            VerifySpanDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifySpanDecisionKind::EarlyRejectStop
        } else {
            VerifySpanDecisionKind::EarlyReject
        };
        return Ok(VerifySpanDecision {
            kind,
            accepted_before_reject,
            repair_input_count: Some(repair_input_count),
            commit_count,
        });
    }

    Ok(VerifySpanDecision {
        kind: VerifySpanDecisionKind::FullAccept,
        accepted_before_reject,
        repair_input_count: None,
        commit_count,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairedCommitError {
    TooFewTokens { expected: usize, got: Vec<i32> },
}

impl fmt::Display for RepairedCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewTokens { expected, got } => {
                write!(
                    f,
                    "recovery verify returned too few tokens: expected {expected} got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for RepairedCommitError {}

pub fn repaired_commit_tokens(
    _draft_tokens: &[i32],
    _accepted_before_reject: usize,
    repair_input_count: usize,
    repaired_predictions: &[i32],
) -> Result<Vec<i32>, RepairedCommitError> {
    if repaired_predictions.len() < repair_input_count {
        return Err(RepairedCommitError::TooFewTokens {
            expected: repair_input_count,
            got: repaired_predictions.to_vec(),
        });
    }
    Ok(repaired_predictions[..repair_input_count].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ngram_config(n: usize, min_hits: u32) -> NgramConfig {
        NgramConfig {
            n,
            min_hits,
            max_tokens_per_pool: 1024,
        }
    }

    #[test]
    fn ngram_store_proposes_repeated_suffix_continuation() {
        let mut store = NgramStore::new(ngram_config(3, 1));
        store.observe_sequence("session-a", &[10, 20, 30, 40, 50, 60]);

        assert_eq!(
            store.propose("session-a", &[1, 20, 30, 40], 2, true).tokens,
            vec![50]
        );
    }

    #[test]
    fn ngram_store_respects_min_hits() {
        let mut store = NgramStore::new(ngram_config(2, 2));
        store.observe_sequence("session-a", &[1, 2, 3, 1, 2, 3]);
        store.observe_sequence("session-b", &[1, 2, 3]);

        assert_eq!(store.propose("session-a", &[1, 2], 1, true).tokens, vec![3]);
        assert!(store
            .propose("session-b", &[1, 2], 1, true)
            .tokens
            .is_empty());
    }

    #[test]
    fn ngram_store_backs_off_to_shorter_suffix() {
        let mut store = NgramStore::new(ngram_config(3, 1));
        store.observe_sequence("session-a", &[7, 8, 9, 10, 4, 8, 9, 11]);

        let proposal = store.propose("session-a", &[1, 8, 9], 1, true);

        assert_eq!(proposal.tokens, vec![10]);
        assert_eq!(proposal.match_order_min, 2);
        assert_eq!(proposal.match_order_max, 2);
        assert_eq!(proposal.stop_reason, "proposal_limit");
    }

    #[test]
    fn ngram_store_reports_policy_window_cap() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 2, 2, 2]);

        let proposal = store.propose("session-a", &[1], 4, true);

        assert_eq!(proposal.tokens, vec![2]);
        assert_eq!(proposal.stop_reason, "policy_window_cap");

        store
            .observe_window(
                "session-a",
                VerifySpanDecision {
                    kind: VerifySpanDecisionKind::TailReject,
                    accepted_before_reject: 0,
                    repair_input_count: Some(1),
                    commit_count: 1,
                },
                1,
                1.0,
                0.0,
                8,
            )
            .expect("policy snapshot");

        let proposal = store.propose("session-a", &[1], 4, true);

        assert_eq!(proposal.tokens, vec![2, 2]);
        assert_eq!(proposal.stop_reason, "policy_window_cap_probe");
    }

    #[test]
    fn ngram_high_accept_policy_probes_caps_more_often() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 2, 2, 2]);

        for _ in 0..8 {
            store
                .observe_window(
                    "session-a",
                    VerifySpanDecision {
                        kind: VerifySpanDecisionKind::FullAccept,
                        accepted_before_reject: 2,
                        repair_input_count: None,
                        commit_count: 2,
                    },
                    2,
                    1.0,
                    0.0,
                    8,
                )
                .expect("policy snapshot");
        }

        let proposal = store.propose("session-a", &[1], 4, true);

        assert_eq!(proposal.policy.quality, "high_accept");
        assert_eq!(proposal.tokens.len(), 4);
        assert_eq!(proposal.stop_reason, "proposal_limit");
    }

    #[test]
    fn ngram_policy_probes_without_user_until_repeated_suffix() {
        let mut store = NgramStore::new(ngram_config(2, 1));
        store.observe_sequence("session-a", &[1, 2, 3]);

        let miss = store.propose("session-a", &[8, 9], 1, false);
        assert!(miss.tokens.is_empty());
        assert!(!miss.policy.enabled);
        assert_eq!(miss.policy.reason, "probing_no_repeated_suffix");

        let hit = store.propose("session-a", &[1, 2], 1, false);
        assert_eq!(hit.tokens, vec![3]);
        assert!(hit.policy.enabled);
        assert_eq!(hit.policy.reason, "enabled_repeated_suffix");
    }

    #[test]
    fn ngram_auto_warms_up_before_repeated_suffix_enable() {
        let mut store = NgramStore::new(ngram_config(2, 1));
        store.observe_sequence("session-a", &[1, 2, 3, 1, 2, 3]);

        let first = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: false,
                min_repeated_suffix_hits: 2,
            },
        );
        assert!(first.tokens.is_empty());
        assert_eq!(first.stop_reason, "auto_warmup");
        assert_eq!(first.policy.reason, "cold_observe_repeated_suffix");
        assert_eq!(first.policy.auto_repeated_suffix_hits, 1);

        let second = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: false,
                min_repeated_suffix_hits: 2,
            },
        );
        assert!(second.tokens.is_empty());
        assert_eq!(second.stop_reason, "auto_single_token_hold");
        assert_eq!(second.policy.reason, "auto_single_token_hold");
        assert_eq!(second.policy.auto_repeated_suffix_hits, 3);

        let third = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: false,
                min_repeated_suffix_hits: 2,
            },
        );
        assert_eq!(third.tokens, vec![3]);
        assert_eq!(third.policy.auto_repeated_suffix_hits, 4);
    }

    #[test]
    fn ngram_auto_prompt_shape_enables_immediately_for_multi_token_match() {
        let mut store = NgramStore::new(ngram_config(2, 1));
        store.observe_sequence("session-a", &[1, 2, 3, 4, 1, 2, 3, 4]);

        for expected_hits in 1..=3 {
            let proposal = store.propose_with_activation(
                "session-a",
                &[1, 2],
                2,
                NgramActivation::Auto {
                    prompt_candidate: true,
                    min_repeated_suffix_hits: 8,
                },
            );

            assert!(proposal.tokens.is_empty());
            assert_eq!(proposal.stop_reason, "auto_single_token_hold");
            assert_eq!(proposal.policy.auto_repeated_suffix_hits, expected_hits);
        }

        let proposal = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: true,
                min_repeated_suffix_hits: 8,
            },
        );

        assert_eq!(proposal.tokens, vec![3]);
        assert_eq!(proposal.policy.auto_repeated_suffix_hits, 4);
    }

    #[test]
    fn ngram_auto_prompt_shape_warms_up_one_token_match_before_enable() {
        let mut store = NgramStore::new(ngram_config(2, 1));
        store.observe_sequence("session-a", &[1, 2, 3]);

        let first = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: true,
                min_repeated_suffix_hits: 1,
            },
        );

        assert!(first.tokens.is_empty());
        assert_eq!(first.stop_reason, "auto_warmup");
        assert_eq!(first.policy.reason, "cold_observe_prompt_shape");
        assert_eq!(first.policy.auto_repeated_suffix_hits, 1);

        let second = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: true,
                min_repeated_suffix_hits: 1,
            },
        );

        assert!(second.tokens.is_empty());
        assert_eq!(second.stop_reason, "auto_single_token_hold");
        assert_eq!(second.policy.reason, "auto_single_token_hold");
        assert_eq!(second.policy.auto_repeated_suffix_hits, 3);

        let third = store.propose_with_activation(
            "session-a",
            &[1, 2],
            2,
            NgramActivation::Auto {
                prompt_candidate: true,
                min_repeated_suffix_hits: 1,
            },
        );

        assert_eq!(third.tokens, vec![3]);
        assert_eq!(third.policy.auto_repeated_suffix_hits, 4);
    }

    #[test]
    fn ngram_policy_disables_after_zero_accept_early_reject() {
        let mut store = NgramStore::new(ngram_config(2, 1));
        store.observe_sequence("session-a", &[1, 2, 3]);
        let proposal = store.propose("session-a", &[1, 2], 1, true);
        assert_eq!(proposal.tokens, vec![3]);

        let snapshot = store
            .observe_window(
                "session-a",
                VerifySpanDecision {
                    kind: VerifySpanDecisionKind::EarlyReject,
                    accepted_before_reject: 0,
                    repair_input_count: Some(1),
                    commit_count: 1,
                },
                1,
                5.0,
                2.0,
                8,
            )
            .expect("policy snapshot");

        assert!(!snapshot.enabled);
        assert_eq!(snapshot.cooldown_remaining, 8);
        assert_eq!(snapshot.reason, "disabled_zero_accept_early_reject");
    }

    #[test]
    fn ngram_policy_does_not_grow_zero_accept_tail_reject() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 1, 3]);
        let proposal = store.propose("session-a", &[1], 4, true);
        assert_eq!(proposal.policy.current_window, 1);
        assert_eq!(proposal.tokens.len(), 1);

        let snapshot = store
            .observe_window(
                "session-a",
                VerifySpanDecision {
                    kind: VerifySpanDecisionKind::TailReject,
                    accepted_before_reject: 0,
                    repair_input_count: Some(1),
                    commit_count: 1,
                },
                1,
                5.0,
                0.0,
                8,
            )
            .expect("policy snapshot");

        assert_eq!(snapshot.current_window, 1);
        assert_eq!(snapshot.grow_count, 0);
        assert_eq!(snapshot.unproductive_windows, 1);
        assert_eq!(snapshot.reason, "shrunk_zero_accept_tail_reject");
    }

    #[test]
    fn ngram_policy_fast_grows_full_accept_two_token_window() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 2, 2]);

        let snapshot = store
            .observe_window(
                "session-a",
                VerifySpanDecision {
                    kind: VerifySpanDecisionKind::FullAccept,
                    accepted_before_reject: 2,
                    repair_input_count: None,
                    commit_count: 2,
                },
                2,
                2.0,
                0.0,
                8,
            )
            .expect("policy snapshot");

        assert_eq!(snapshot.current_window, 3);
        assert_eq!(snapshot.reason, "grown_full_accept_fast");
    }

    #[test]
    fn ngram_policy_uses_decode_cost_to_disable_zero_yield() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 1, 3]);
        store.observe_decode_step("session-a", 2.0);
        let proposal = store.propose("session-a", &[1], 4, true);
        assert_eq!(proposal.tokens.len(), 1);

        let snapshot = store
            .observe_window(
                "session-a",
                VerifySpanDecision {
                    kind: VerifySpanDecisionKind::TailReject,
                    accepted_before_reject: 0,
                    repair_input_count: Some(1),
                    commit_count: 1,
                },
                1,
                10.0,
                0.0,
                8,
            )
            .expect("policy snapshot");

        assert!(!snapshot.enabled);
        assert_eq!(snapshot.cooldown_remaining, 8);
        assert_eq!(snapshot.reason, "disabled_costly_zero_yield");
    }

    #[test]
    fn ngram_policy_disables_low_acceptance_early() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 1, 3]);
        let proposal = store.propose("session-a", &[1], 1, true);
        assert!(proposal.policy.enabled);

        let mut snapshot = None;
        for _ in 0..3 {
            snapshot = store
                .observe_window(
                    "session-a",
                    VerifySpanDecision {
                        kind: VerifySpanDecisionKind::TailReject,
                        accepted_before_reject: 1,
                        repair_input_count: Some(2),
                        commit_count: 2,
                    },
                    4,
                    4.0,
                    0.0,
                    8,
                )
                .or(snapshot);
        }
        let snapshot = snapshot.expect("policy snapshot");

        assert!(!snapshot.enabled);
        assert_eq!(snapshot.quality, "disabled");
        assert_eq!(snapshot.cooldown_remaining, 12);
        assert_eq!(snapshot.reason, "disabled_low_accept_rate_early");
    }

    #[test]
    fn ngram_policy_skips_unproductive_single_token_verify() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 1, 3]);
        let proposal = store.propose("session-a", &[1], 4, true);
        assert_eq!(proposal.tokens.len(), 1);

        let snapshot = store
            .observe_window(
                "session-a",
                VerifySpanDecision {
                    kind: VerifySpanDecisionKind::TailReject,
                    accepted_before_reject: 0,
                    repair_input_count: Some(1),
                    commit_count: 1,
                },
                1,
                5.0,
                0.0,
                8,
            )
            .expect("policy snapshot");
        assert!(snapshot.should_skip_single_token_verify(1, 1));
        assert!(snapshot.should_skip_single_token_verify(1, 2));

        let snapshot = store
            .observe_skipped_single_token_verify("session-a")
            .expect("policy snapshot");
        assert_eq!(snapshot.skipped_single_token_windows, 1);
        assert_eq!(snapshot.reason, "skipped_single_token_verify");
    }

    #[test]
    fn ngram_policy_disables_after_repeated_single_token_stalls() {
        let mut store = NgramStore::new(ngram_config(1, 1));
        store.observe_sequence("session-a", &[1, 2, 1, 3]);
        let proposal = store.propose("session-a", &[1], 4, true);
        assert_eq!(proposal.tokens.len(), 1);

        let mut snapshot = None;
        for _ in 0..8 {
            snapshot = store.observe_skipped_single_token_verify("session-a");
        }
        let snapshot = snapshot.expect("policy snapshot");

        assert!(!snapshot.enabled);
        assert_eq!(snapshot.cooldown_remaining, 16);
        assert_eq!(snapshot.reason, "disabled_single_token_stall");
        assert_eq!(snapshot.disable_count, 1);
    }

    #[test]
    fn verify_span_repair_strategy_uses_costly_decode_one_for_ngram() {
        let decision = VerifySpanDecision {
            kind: VerifySpanDecisionKind::EarlyReject,
            accepted_before_reject: 1,
            repair_input_count: Some(2),
            commit_count: 2,
        };
        let policy = test_policy(8.0);

        assert_eq!(
            VerifySpanRepairStrategy::for_decision(decision, "ngram", Some(policy), 8.0, 2),
            VerifySpanRepairStrategy::RestoreDecodeOne
        );
        assert_eq!(
            repair_path_label(decision, VerifySpanRepairStrategy::RestoreDecodeOne),
            "restore_decode_one_cost"
        );
    }

    #[test]
    fn verify_span_repair_strategy_keeps_reverify_when_cost_is_reasonable() {
        let decision = VerifySpanDecision {
            kind: VerifySpanDecisionKind::EarlyReject,
            accepted_before_reject: 1,
            repair_input_count: Some(2),
            commit_count: 2,
        };
        let policy = test_policy_with_baseline(2.0, 4.0);

        assert_eq!(
            VerifySpanRepairStrategy::for_decision(decision, "ngram", Some(policy), 3.0, 2),
            VerifySpanRepairStrategy::RestoreReverify
        );
    }

    #[test]
    fn verify_span_repair_strategy_projects_reverify_cost_from_full_span() {
        let decision = VerifySpanDecision {
            kind: VerifySpanDecisionKind::EarlyReject,
            accepted_before_reject: 2,
            repair_input_count: Some(3),
            commit_count: 3,
        };
        let policy = test_policy_with_baseline(10.0, 10.0);

        assert_eq!(
            VerifySpanRepairStrategy::for_decision(decision, "ngram", Some(policy), 16.0, 8),
            VerifySpanRepairStrategy::RestoreReverify
        );
        assert_eq!(
            VerifySpanRepairStrategy::for_decision(decision, "ngram", Some(policy), 40.0, 8),
            VerifySpanRepairStrategy::RestoreDecodeOne
        );
    }

    #[test]
    fn classify_verify_span_reports_full_accept() {
        let decision = classify_verify_span(&[1, 2], &[1, 2], 0, 16, |token| {
            Ok::<_, &'static str>(token == 0)
        })
        .expect("decision");

        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::FullAccept,
                accepted_before_reject: 2,
                repair_input_count: None,
                commit_count: 2,
            }
        );
    }

    #[test]
    fn classify_verify_span_reports_early_reject() {
        let decision = classify_verify_span(&[1, 2, 3], &[1, 9, 3], 0, 16, |token| {
            Ok::<_, &'static str>(token == 0)
        })
        .expect("decision");

        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            }
        );
    }

    #[test]
    fn repaired_commit_tokens_returns_repair_span() {
        assert_eq!(
            repaired_commit_tokens(&[1, 2, 3], 1, 2, &[1, 9, 8]).expect("commit tokens"),
            vec![1, 9]
        );
    }

    fn test_policy(verify_ms_per_accepted: f64) -> NgramPolicySnapshot {
        test_policy_with_baseline(verify_ms_per_accepted, 2.0)
    }

    fn test_policy_with_baseline(
        verify_ms_per_accepted: f64,
        non_spec_decode_ms_per_token: f64,
    ) -> NgramPolicySnapshot {
        NgramPolicySnapshot {
            enabled: true,
            cooldown_remaining: 0,
            current_window: 1,
            windows: 1,
            skipped_single_token_windows: 0,
            policy_window_cap_probes: 0,
            auto_repeated_suffix_hits: 0,
            proposed_tokens: 2,
            accepted_tokens: 1,
            rejected_windows: 1,
            early_reject_windows: 1,
            unproductive_windows: 0,
            enable_count: 1,
            disable_count: 0,
            grow_count: 0,
            shrink_count: 0,
            accept_rate: 0.5,
            verify_ms_per_accepted,
            repair_ms_per_accepted: 0.0,
            non_spec_decode_ms_per_token,
            quality: "probing",
            reason: "test",
        }
    }
}
