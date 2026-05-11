struct ChildGuard {
    child: Child,
}

impl ChildGuard {
    fn spawn(mut command: Command) -> Result<Self> {
        command.stdin(Stdio::null());
        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn {:?}", command))?;
        Ok(Self { child })
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct LocalStage {
    stage_id: String,
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    port: u16,
    bind_addr: String,
    endpoint_addr: String,
    config_path: PathBuf,
    model_path: PathBuf,
    remote: Option<RemoteStage>,
}

struct RemoteStage {
    host: String,
    stage_dir: String,
    stage_server_bin: String,
    model_path: String,
    config_path: String,
    stage_log_path: String,
    stage_exit_path: String,
}

struct PromptLogContext {
    entries: Vec<PromptLogEntry>,
    default_lines: usize,
}

struct PromptLogEntry {
    label: String,
    target: PromptLogTarget,
}

enum PromptLogTarget {
    Local(PathBuf),
    Remote { host: String, path: String },
}

pub fn prompt_repl(_args: PromptArgs) -> Result<()> {
    bail!(
        "skippy-prompt topology launch is disabled in mesh-llm; use `skippy-prompt binary` against a mesh-managed first stage"
    )
}

#[allow(dead_code)]
fn prompt_repl_launch(args: PromptArgs) -> Result<()> {
    let remote = !args.hosts.is_empty();
    if args.single_stage && args.splits.is_some() {
        bail!("--single-stage conflicts with --splits");
    }
    let hf_package_ref = args
        .model_path
        .to_str()
        .is_some_and(|s| s.starts_with("hf://"))
        || args.model_path.join("model-package.json").is_file();
    if !hf_package_ref && !args.model_path.is_file() {
        bail!("model does not exist: {}", args.model_path.display());
    }
    let effective_layer_end = if args.single_stage {
        if hf_package_ref {
            bail!("--single-stage requires a local GGUF file, not an hf:// package ref");
        }
        model_layer_count(&args.model_path).context("infer full layer count for --single-stage")?
    } else {
        args.layer_end
    };
    let default_stage_count = if args.single_stage {
        1
    } else if remote {
        args.hosts.len()
    } else {
        4
    };
    let ranges = resolve_stage_ranges(
        args.single_stage,
        args.splits.as_deref(),
        default_stage_count,
        effective_layer_end,
    )?;
    validate_prompt_topology_plan(&args, effective_layer_end, &ranges)?;
    if remote && args.hosts.len() != ranges.len() {
        bail!(
            "--hosts count must match the stage count; got {} hosts for {} stages",
            args.hosts.len(),
            ranges.len()
        );
    }
    let kv_mode = parse_stage_kv_mode(&args.kv_mode)?;

    let run_id = format!("prompt-{}", unix_millis());
    let run_dir = args.run_root.join(&run_id);
    let config_dir = run_dir.join("configs");
    fs::create_dir_all(&config_dir)
        .with_context(|| format!("create run config dir {}", config_dir.display()))?;
    let model_cache_key = if hf_package_ref {
        let mut hasher = DefaultHasher::new();
        args.model_path.to_str().unwrap_or("hf").hash(&mut hasher);
        format!("hf-{:016x}", hasher.finish())
    } else {
        model_cache_key(&args.model_path, &ranges)?
    };
    let model_cache_dir = args.run_root.join("model-cache").join(&model_cache_key);
    let model_package_cache_key = if hf_package_ref {
        model_cache_key.clone()
    } else {
        model_package_cache_key(&args.model_path)?
    };
    let model_package_dir = args
        .run_root
        .join("model-package-cache")
        .join(&model_package_cache_key);
    let binary_cache_key = binary_cache_key(&args.stage_server_bin)?;

    let mut stages = Vec::with_capacity(ranges.len());
    for (index, (layer_start, layer_end)) in ranges.iter().copied().enumerate() {
        let port = args
            .first_stage_port
            .checked_add(u16::try_from(index).context("stage index exceeds u16")?)
            .context("stage port overflow")?;
        let (bind_addr, endpoint_addr, remote_stage) = if remote {
            let host = args.hosts[index].trim().to_string();
            if host.is_empty() {
                bail!("--hosts contains an empty host at position {}", index + 1);
            }
            let stage_dir = format!("{}/runs/{run_id}/stage-{index}", args.remote_root);
            let remote_model_package_dir = if hf_package_ref {
                args.model_path.to_str().unwrap_or("").to_string()
            } else {
                format!(
                    "{}/model-package-cache/{model_package_cache_key}",
                    args.remote_root
                )
            };
            let remote_binary_cache_dir =
                format!("{}/binary-cache/{binary_cache_key}", args.remote_root);
            (
                format!("{}:{port}", args.remote_bind_host),
                format!("{host}:{port}"),
                Some(RemoteStage {
                    host,
                    stage_server_bin: format!("{remote_binary_cache_dir}/skippy-server"),
                    model_path: remote_model_package_dir,
                    config_path: format!("{stage_dir}/stage-{index}.json"),
                    stage_log_path: format!("{stage_dir}/stage-{index}.log"),
                    stage_exit_path: format!("{stage_dir}/stage.exit"),
                    stage_dir,
                }),
            )
        } else {
            (
                format!("127.0.0.1:{port}"),
                format!("127.0.0.1:{port}"),
                None,
            )
        };
        stages.push(LocalStage {
            stage_id: format!("stage-{index}"),
            stage_index: index,
            layer_start,
            layer_end,
            port,
            bind_addr,
            endpoint_addr,
            config_path: config_dir.join(format!("stage-{index}.json")),
            model_path: model_cache_dir.join(format!("stage-{index}.gguf")),
            remote: remote_stage,
        });
    }

    let metrics_otlp_url = metrics_otlp_url(&args, &stages)?;
    if hf_package_ref {
        eprintln!(
            "launch: using HF layer package {} (each stage downloads its own layers)",
            args.model_path.display()
        );
    } else if remote {
        eprintln!(
            "launch: materializing remote model package cache at {}",
            model_package_dir.display()
        );
        materialize_model_package(&args, &model_package_dir)?;
    } else {
        eprintln!(
            "launch: materializing {} local GGUF stage shards",
            stages.len()
        );
        materialize_stage_artifacts(&args, &stages)?;
    }
    eprintln!(
        "launch: writing stage and KV configs under {}",
        run_dir.display()
    );
    write_local_configs(
        &args,
        &run_id,
        &run_dir,
        kv_mode,
        &stages,
        &metrics_otlp_url,
        hf_package_ref,
    )?;

    let mut children = Vec::new();
    let metrics_db = run_dir.join("metrics.sqlite");
    let metrics_otlp_bind_addr = metrics_otlp_bind_addr(&args, remote);
    let mut metrics = Command::new(&args.metrics_server_bin);
    metrics.args([
        "serve",
        "--db",
        path_str(&metrics_db)?,
        "--http-addr",
        &args.metrics_http_addr.to_string(),
        "--otlp-grpc-addr",
        &metrics_otlp_bind_addr,
    ]);
    configure_process_log(&mut metrics, &run_dir.join("metrics-server.log"))?;
    eprintln!(
        "launch: starting metrics-server http={} otlp_bind={} log={}",
        args.metrics_http_addr,
        metrics_otlp_bind_addr,
        run_dir.join("metrics-server.log").display()
    );
    children.push(ChildGuard::spawn(metrics)?);

    let ngram_pool_uds_path = if args.ngram_speculative {
        let socket_path = args
            .ngram_pool_uds_path
            .clone()
            .unwrap_or_else(|| run_dir.join("ngram-pool.sock"));
        let mut ngram_pool = Command::new(&args.ngram_pool_server_bin);
        ngram_pool.args(["serve", "--uds-path", path_str(&socket_path)?]);
        configure_process_log(&mut ngram_pool, &run_dir.join("ngram-pool-server.log"))?;
        eprintln!(
            "launch: starting ngram-pool-server socket={} log={}",
            socket_path.display(),
            run_dir.join("ngram-pool-server.log").display()
        );
        children.push(ChildGuard::spawn(ngram_pool)?);
        eprintln!(
            "launch: waiting for ngram pool socket {}",
            socket_path.display()
        );
        wait_for_socket(&socket_path, args.startup_timeout_secs)?;
        Some(socket_path)
    } else {
        None
    };

    if remote {
        rsync_remote_stage_inputs(&args, &stages, &model_package_dir, hf_package_ref)?;
        eprintln!(
            "launch: starting {} remote stages; first_stage_addr={}",
            stages.len(),
            stages[0].endpoint_addr
        );
        start_remote_stages(&args, &stages, &metrics_otlp_url, &mut children)?;
    } else {
        eprintln!("launch: starting {} local stage servers", stages.len());
        for stage in stages.iter().rev() {
            let mut server = Command::new(&args.stage_server_bin);
            add_stage_server_args(&mut server, &args, stage, &metrics_otlp_url)?;
            configure_process_log(
                &mut server,
                &run_dir.join(format!("stage-{}.log", stage.stage_index)),
            )?;
            children.push(ChildGuard::spawn(server)?);
        }
    }

    eprintln!("prompt run dir: {}", run_dir.display());
    eprintln!(
        "metrics: otlp_bind={} otlp_url={}",
        metrics_otlp_bind_addr, metrics_otlp_url
    );
    eprintln!(
        "topology: {}",
        stages
            .iter()
            .map(|stage| format!(
                "{}:{}..{}@{} model={}",
                stage.stage_id,
                stage.layer_start,
                stage.layer_end,
                stage.endpoint_addr,
                stage_model_location(stage)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("logs: use :logs [name] [lines] inside the REPL");

    let log_context = Some(prompt_log_context(&run_dir, &stages, args.log_tail_lines));

    let tokenizer_model_path = if hf_package_ref {
        let hf_ref = args.model_path.to_str().context("model_path not utf-8")?;
        eprintln!("launch: materializing tokenizer from package (layers 0..1)");
        let tokenizer_gguf = materialize_layer_package(&PackageStageRequest {
            model_id: args.model_id.clone(),
            topology_id: "tokenizer".to_string(),
            package_ref: hf_ref.to_string(),
            stage_id: "tokenizer".to_string(),
            layer_start: 0,
            layer_end: 1,
            include_embeddings: true,
            include_output: true,
        })
        .context("materialize tokenizer from layer package")?;
        Some(tokenizer_gguf)
    } else if remote {
        None
    } else {
        stages.first().map(|stage| stage.model_path.clone())
    };
    let tokenizer_load_mode = if hf_package_ref {
        ReplLoadMode::LayerPackage
    } else if tokenizer_model_path.is_some() {
        ReplLoadMode::ArtifactSlice
    } else {
        ReplLoadMode::RuntimeSlice
    };
    let repl_result = binary_repl(BinaryReplArgs {
        model_path: args.model_path,
        tokenizer_model_path,
        tokenizer_load_mode,
        tokenizer_n_gpu_layers: 0,
        first_stage_addr: stages[0].endpoint_addr.clone(),
        tokenizer_layer_start: stages[0].layer_start as u32,
        tokenizer_layer_end: stages[0].layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        prefill_chunk_size: args.prefill_chunk_size,
        max_new_tokens: args.max_new_tokens,
        draft_model_path: args.draft_model_path,
        speculative_window: args.speculative_window,
        adaptive_speculative_window: args.adaptive_speculative_window,
        ngram_speculative: args.ngram_speculative,
        ngram_proposal_mode: args.ngram_proposal_mode,
        spec_ngram_size_n: args.spec_ngram_size_n,
        ngram_history_min_hits: args.ngram_history_min_hits,
        draft_min: args.draft_min,
        draft_max: args.draft_max,
        ngram_min_winner_count: args.ngram_min_winner_count,
        ngram_min_confidence: args.ngram_min_confidence,
        ngram_min_margin: args.ngram_min_margin,
        ngram_confidence_step: args.ngram_confidence_step,
        ngram_confidence_step_tokens: args.ngram_confidence_step_tokens,
        ngram_max_confidence: args.ngram_max_confidence,
        ngram_count_step_tokens: args.ngram_count_step_tokens,
        ngram_margin_step_tokens: args.ngram_margin_step_tokens,
        startup_timeout_secs: args.startup_timeout_secs,
        decode_timeout_secs: args.decode_timeout_secs,
        history_path: Some(
            args.history_path
                .unwrap_or_else(|| args.run_root.join("prompt-history.txt")),
        ),
        session_id: args.session_id,
        ngram_pool_uds_path,
        native_logs: args.native_logs,
        raw_prompt: args.raw_prompt,
        no_think: args.no_think,
        thinking_token_budget: args.thinking_token_budget,
        diagnostics_hint: Some(stage_diagnostics_hint(&run_dir, &stages)),
        log_context,
    });

    if remote {
        for stage in &stages {
            let _ = stop_remote_stage_listener(stage);
        }
    }
    drop(children);
    repl_result
}
