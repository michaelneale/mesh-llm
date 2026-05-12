struct NgramSource;

impl NgramSource {
    fn open(_args: &BinaryReplArgs, _session_id: &str) -> Result<Self> {
        bail!("ngram speculative prompting is not imported into mesh-llm")
    }

    fn observe_sequence(&mut self, _session_id: &str, _tokens: &[i32]) -> Result<()> {
        Ok(())
    }

    fn observe_accepted(&mut self, _session_id: &str, _context_tokens: &[i32]) -> Result<()> {
        Ok(())
    }

    fn propose(
        &mut self,
        _session_id: &str,
        _context_tokens: &[i32],
        _remaining: usize,
    ) -> Result<Vec<i32>> {
        Ok(Vec::new())
    }
}

struct DraftRunner {
    path: PathBuf,
    window: usize,
    _model: StageModel,
    session: StageSession,
}

impl DraftRunner {
    fn open(path: &Path, ctx_size: u32, n_gpu_layers: i32, window: usize) -> Result<Self> {
        if !path.is_file() {
            bail!("draft model does not exist: {}", path.display());
        }
        let layer_count = model_layer_count(path)?;
        let model = StageModel::open(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers,
                selected_backend_device: None,
                cache_type_k: GGML_TYPE_F16,
                cache_type_v: GGML_TYPE_F16,
                flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: true,
            },
        )
        .with_context(|| format!("open draft model {}", path.display()))?;
        let session = model.create_session().context("create draft session")?;
        Ok(Self {
            path: path.to_path_buf(),
            window,
            _model: model,
            session,
        })
    }

    fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.session.reset().context("reset draft session")?;
        if context_tokens.len() > 1 {
            self.session
                .prefill_chunk(&context_tokens[..context_tokens.len() - 1])
                .context("prefill draft context")?;
        }
        Ok(())
    }

    fn propose(&mut self, mut current: i32, max_tokens: usize) -> Result<Vec<i32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            current = self
                .session
                .decode_step(current)
                .context("draft decode step")?;
            tokens.push(current);
        }
        Ok(tokens)
    }
}

fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .context("model has no layer-indexed tensors")?;
    Ok(layer_count)
}
