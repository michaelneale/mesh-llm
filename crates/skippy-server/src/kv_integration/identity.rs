use skippy_cache::{
    prefix_identity_with_namespace, NATIVE_KV_DTYPE, NATIVE_KV_RUNTIME_ABI_VERSION,
};
use skippy_protocol::{MessageBase, StageConfig};

use crate::kv_proto::{KvCodec, PageIdentity, PageLayout};

use super::{KvStageIntegration, PrefillKvIdentity};

impl KvStageIntegration {
    pub fn prefill_identity(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> PrefillKvIdentity {
        let prefix = prefix_identity_with_namespace(
            config,
            token_start,
            token_ids,
            base.chat_template_id.as_deref(),
        );
        let identity = PageIdentity {
            model_id: config.model_id.clone(),
            model_revision: "unknown".to_string(),
            runtime_abi_version: NATIVE_KV_RUNTIME_ABI_VERSION.to_string(),
            topology_id: config.topology_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            prefix_hash: prefix.prefix_hash.clone(),
            session_id: base.session_id.clone(),
            token_start,
            token_count: prefix.token_count,
            generation: 1,
            layout: PageLayout::LayerContiguous as i32,
            codec: KvCodec::Fp16 as i32,
            tokenizer_id: base
                .tokenizer_id
                .clone()
                .unwrap_or_else(|| config.model_id.clone()),
            chat_template_id: base.chat_template_id.clone().unwrap_or_default(),
            position_config_hash: format!("ctx:{}", config.ctx_size),
            kv_dtype: NATIVE_KV_DTYPE.to_string(),
        };
        PrefillKvIdentity {
            identity,
            page_id: prefix.page_id,
        }
    }

    pub fn lookup_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.record_candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }

    pub fn record_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.record_candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }
}
