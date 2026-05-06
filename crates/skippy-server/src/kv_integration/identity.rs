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
        self.lookup_candidate_token_counts(token_ids.len() as u64)
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

#[cfg(test)]
mod tests {
    use skippy_protocol::{
        LoadMode, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
        SCHEMA_VERSION,
    };

    use super::*;

    fn test_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "org/model:Q4_K_M".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 4,
            ctx_size: 512,
            lane_count: 2,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: Default::default(),
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                payload: StageKvCachePayload::ResidentKv,
                max_entries: 8,
                max_bytes: 0,
                min_tokens: 64,
                shared_prefix_stride_tokens: 32,
                shared_prefix_record_limit: 2,
            }),
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn test_base() -> MessageBase {
        MessageBase {
            schema_version: SCHEMA_VERSION,
            run_id: "run".to_string(),
            request_id: "request".to_string(),
            session_id: "session".to_string(),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            topology_id: "topology".to_string(),
            model_id: Some("org/model:Q4_K_M".to_string()),
            tokenizer_id: None,
            chat_template_id: Some("template".to_string()),
            seq: Some(1),
        }
    }

    #[test]
    fn lookup_identities_use_full_longest_prefix_grid() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("cache enabled");
        let tokens = (0..160).collect::<Vec<_>>();

        let lookup_counts = kv
            .lookup_identities(&config, &test_base(), 0, &tokens)
            .into_iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();
        let record_counts = kv
            .record_identities(&config, &test_base(), 0, &tokens)
            .into_iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();

        assert_eq!(lookup_counts, vec![160, 159, 127, 95, 64]);
        assert_eq!(record_counts, vec![160, 64]);
    }
}
