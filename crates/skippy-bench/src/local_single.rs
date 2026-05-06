use std::{
    fs,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    cli::LocalSingleArgs,
    model_identity::model_identity_for_path,
    support::{generate_run_id, retry, temp_config_path, temp_db_path, ChildGuard},
};

#[derive(Deserialize)]
struct CreateRunResponse {
    run_id: String,
}

#[derive(Deserialize)]
struct StageStatus {
    ready: bool,
    runtime_loaded: bool,
}

#[derive(Serialize)]
struct TextRequest<'a> {
    request_id: &'a str,
    session_id: &'a str,
    prompt: &'a str,
    max_new_tokens: usize,
}

pub fn local_single(args: LocalSingleArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!("layer_start must be less than layer_end");
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;
    let run_id = args.run_id.unwrap_or_else(generate_run_id);
    let metrics_http = format!("http://{}", args.metrics_http_addr);
    let metrics_otlp = format!("http://{}", args.metrics_otlp_grpc_addr);
    let stage_http = format!("http://{}", args.stage_bind_addr);
    let db = args.db.unwrap_or_else(|| temp_db_path(&run_id));
    let stage_config = temp_config_path(&run_id);
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;

    let mut metrics_command = Command::new(&args.metrics_server_bin);
    metrics_command.args([
        "serve",
        "--db",
        db.to_str().context("db path is not valid UTF-8")?,
        "--http-addr",
        &args.metrics_http_addr.to_string(),
        "--otlp-grpc-addr",
        &args.metrics_otlp_grpc_addr.to_string(),
    ]);
    if args.child_logs {
        metrics_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        metrics_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _metrics = ChildGuard::spawn(metrics_command)?;

    let run_config = json!({
        "run_id": run_id,
        "topology_id": args.topology_id,
        "model_id": model_identity.model_id,
        "model_identity": model_identity,
        "mode": "local-single",
    });
    retry(args.startup_timeout_secs, || {
        let response = client
            .post(format!("{metrics_http}/v1/runs"))
            .json(&run_config)
            .send()
            .and_then(|response| response.error_for_status())?
            .json::<CreateRunResponse>()?;
        if response.run_id == run_id {
            Ok(())
        } else {
            Err(anyhow!(
                "metrics-server returned unexpected run_id {}",
                response.run_id
            ))
        }
    })
    .context("metrics-server did not become ready")?;

    let config = json!({
        "run_id": run_id,
        "topology_id": run_config["topology_id"],
        "model_id": run_config["model_id"],
        "model_path": args.model_path,
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "filter_tensors_on_load": false,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage_bind_addr,
        "upstream": null,
        "downstream": null
    });
    fs::write(&stage_config, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", stage_config.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve",
        "--config",
        stage_config
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--metrics-otlp-grpc",
        &metrics_otlp,
    ]);
    if args.child_logs {
        stage_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        stage_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _stage = ChildGuard::spawn(stage_command)?;

    retry(args.startup_timeout_secs, || {
        let status = client
            .get(format!("{stage_http}/v1/status"))
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(anyhow::Error::new)?
            .json::<StageStatus>()?;
        if status.ready && status.runtime_loaded {
            Ok(())
        } else {
            Err(anyhow!("stage is not ready yet"))
        }
    })
    .context("stage server did not become ready")?;

    let request = TextRequest {
        request_id: "local-single-request-1",
        session_id: "local-single-session-1",
        prompt: &args.prompt,
        max_new_tokens: args.max_new_tokens,
    };
    let text_response: Value = client
        .post(format!("{stage_http}/v1/text"))
        .json(&request)
        .send()
        .context("failed to send text request")?
        .error_for_status()
        .context("text request failed")?
        .json()
        .context("failed to parse text response")?;

    thread::sleep(Duration::from_secs(1));
    client
        .post(format!("{metrics_http}/v1/runs/{run_id}/finalize"))
        .send()
        .context("failed to finalize run")?
        .error_for_status()
        .context("metrics-server rejected finalize")?;
    let report: Value = client
        .get(format!("{metrics_http}/v1/runs/{run_id}/report.json"))
        .send()
        .context("failed to fetch report")?
        .error_for_status()
        .context("metrics-server rejected report fetch")?
        .json()
        .context("failed to parse report")?;

    if let Some(output) = args.output {
        fs::write(&output, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": run_id,
            "model_identity": run_config["model_identity"],
            "text_response": text_response,
            "report_counts": report["counts"],
        }))?
    );

    Ok(())
}
