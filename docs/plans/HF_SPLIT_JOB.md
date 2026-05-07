# `mesh-llm model-prepare` — Layer Package Preparation via HF Jobs

Status: plan

## Summary

A new `mesh-llm model-prepare` CLI subcommand that takes a HuggingFace model
repo + quant, submits an HF Job to split it into a per-layer package, publishes
the result, and updates the `meshllm/catalog` dataset.

The splitting logic already exists — `skippy-model-package` (built from this
repo) runs inside the HF Job container. The job script already lives in the
`meshllm/layer-split-output` bucket. This plan brings the submission workflow
into mesh-llm as a first-class command, replacing the standalone shell scripts
in `hf-mesh-skippy-splitter`.

```bash
mesh-llm model-prepare unsloth/Qwen3-235B-A22B-GGUF --quant UD-Q4_K_XL
```

## Background

### What exists today

| Piece | Where | Status |
|---|---|---|
| `skippy-model-package` crate | `crates/skippy-model-package/` | Working binary — `write-package` and `validate-package` subcommands |
| Shell job scripts | `hf-mesh-skippy-splitter/scripts/` | `run-split-job.sh` submits HF Job, `split-model-job.sh` runs inside the container |
| Job script in bucket | `meshllm/layer-split-output/split-model-job.sh` | Deployed, public, **may be stale** relative to repo copy |
| `meshllm/catalog` dataset | HuggingFace `meshllm/catalog` | Per-source-repo JSON entries with variant → layer-package mappings |
| Layer package spec | `docs/specs/layer-package-repos.md` | Defines manifest schema, layout, validation rules |

### How it works today

```
User runs run-split-job.sh
  → uploads split-model-job.sh to HF bucket
  → submits HF Job (cpu-xl, ~$1/hr)
      → inside the container:
          1. apt-get build deps, install Rust
          2. clone mesh-llm, build skippy-model-package (~5 min)
          3. mount source GGUF repo as /source (instant, no download)
          4. split into per-layer GGUFs (~1 GB/min)
          5. validate tensor coverage
          6. upload layer package to target HF repo
          7. update meshllm/catalog dataset entry
          8. upload model card README
```

## Design

### Name: `model-prepare`

The user is preparing a model for distributed inference. The implementation
splits + publishes + catalogs, but the user-facing concept is preparation.

### No local mode

The command always submits an HF Job. It does not run `skippy-model-package`
locally. The HF Job mounts the source model directly from HF storage — no
local download needed.

### No confirmation prompt

The command submits immediately. The `--dry-run` flag prints the job spec
without submitting for users who want to inspect first.

### Script freshness checking

The job script lives in the `meshllm/layer-split-output` bucket and is
publicly readable. Everyone references it as a read-only volume mount. The
script is **not** re-uploaded on every run.

The `model-prepare` crate embeds the canonical script source via
`include_str!` and computes its SHA-256 at build time in `build.rs`. At
runtime, before submitting a job:

1. Call `HFBucket::get_paths_info(["split-model-job.sh"])` to get the bucket
   file's `size` from `BucketTreeEntry::File`.
2. Compare against the embedded script's size.
3. If sizes differ, the bucket is stale:
   - **meshllm member:** warn and suggest `mesh-llm model-prepare --update-script`
   - **Non-member:** warn that the bucket script may be out of date; proceed
     anyway since they cannot update it.
4. If sizes match and exact verification is wanted, download the ~10 KB file
   via `HFBucket::download_files` and compare SHA-256 against the compile-time
   constant.

The `--update-script` flag uploads the embedded script to the bucket via
`HFBucket::upload_files`. Only meshllm org members have write access.

### Permission-aware publish strategy

After the HF Job completes, it needs to:
1. Create/update the target layer-package repo
2. Update the `meshllm/catalog` dataset

Permission is checked **before submission** via `HFClient::whoami()`:

| User type | Target repo | Catalog update | Job namespace |
|---|---|---|---|
| meshllm org member | `meshllm/{distribution}-layers` | Direct commit to `meshllm/catalog` | `meshllm` |
| Non-member | `{username}/{distribution}-layers` | Opens PR to `meshllm/catalog` | `{username}` |

Detection: `whoami()` returns `User { orgs: Option<Vec<OrgMembership>> }`.
Check if any org has `name == Some("meshllm")`.

The permission mode is passed to the job as env var `CATALOG_CREATE_PR=true`
or `false`. The job script's Python catalog-update step uses
`create_pr=True` on `api.upload_file()` when the flag is set. Similarly,
`TARGET_REPO` is derived to the appropriate namespace before submission.

## HF API surface

### What the Rust `hf_hub` crate covers

Every operation except the Jobs API is handled natively by the Rust `hf_hub`
crate. No shelling out to the `hf` CLI anywhere.

| Operation | API |
|---|---|
| whoami (org check) | `HFClient::whoami()` → `User { orgs }` |
| List model repo files | `HFClient::model(owner, name).list_tree(params)` → `RepoTreeEntry::File { path, size }` |
| Check bucket script metadata | `HFClient::bucket("meshllm", "layer-split-output").get_paths_info(["split-model-job.sh"])` → `BucketTreeEntry::File { size, xet_hash }` |
| Download bucket file (exact hash) | `HFBucket::download_files(params)` |
| Upload bucket script | `HFBucket::upload_files([(local, remote)])` |
| Create target repo | `HFClient::create_repo(CreateRepoParams { exist_ok: true })` |
| Upload catalog entry | `HFClient::dataset(owner, name).upload_file(RepoUploadFileParams { create_pr: Some(true) })` |

### Jobs API via reqwest

The Rust `hf_hub` crate has no Jobs API. The HF Jobs REST API is 5 simple
endpoints, called directly with `reqwest` (already in the workspace):

| Operation | Method | Path |
|---|---|---|
| Submit | POST | `/api/jobs/{namespace}` |
| Inspect | GET | `/api/jobs/{namespace}/{job_id}` |
| Logs | GET (SSE) | `/api/jobs/{namespace}/{job_id}/logs` |
| Cancel | POST | `/api/jobs/{namespace}/{job_id}/cancel` |
| List | GET | `/api/jobs/{namespace}` |

Auth: `Authorization: Bearer {token}` — token resolved via `HF_TOKEN` env
var or `~/.cache/huggingface/token` (same path `hf_hub` uses internally;
`model-hf` already has `hf_token_override()` for this).

#### Submit request body

```json
{
  "dockerImage": "ubuntu:22.04",
  "command": ["bash", "/bucket/split-model-job.sh"],
  "arguments": [],
  "environment": {
    "SOURCE_REPO": "unsloth/Qwen3-235B-A22B-GGUF",
    "SOURCE_FILE": "UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf",
    "TARGET_REPO": "meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers",
    "MODEL_ID": "unsloth/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
    "SOURCE_REVISION": "main",
    "MESH_LLM_REF": "main",
    "CATALOG_CREATE_PR": "false"
  },
  "secrets": { "HF_TOKEN": true },
  "flavor": "cpu-xl",
  "timeoutSeconds": 10800,
  "volumes": [
    {
      "type": "model",
      "source": "unsloth/Qwen3-235B-A22B-GGUF",
      "mountPath": "/source",
      "readOnly": true
    },
    {
      "type": "bucket",
      "source": "meshllm/layer-split-output",
      "mountPath": "/bucket",
      "readOnly": true
    }
  ]
}
```

#### Response / inspect

```json
{
  "id": "job-abc123",
  "status": { "stage": "RUNNING", "message": null },
  "created_at": "2025-07-22T16:06:25Z",
  "dockerImage": "ubuntu:22.04",
  ...
}
```

Stages: `RUNNING` → `COMPLETED` | `ERROR` | `CANCELED` | `DELETED`.

#### Logs

SSE stream. Each line: `data: { "data": "log line text", "timestamp": "..." }`.

## Crate structure

### New workspace crate: `crates/model-prepare/`

```
crates/model-prepare/
├── Cargo.toml
├── README.md
├── build.rs                        # Computes SHA-256 of embedded script at build time
└── src/
    ├── lib.rs                      # Crate root, re-exports
    ├── jobs.rs                     # HfJobsClient: submit, inspect, logs, cancel, list
    ├── prepare.rs                  # PrepareJob: source resolution, target derivation, job spec
    ├── permissions.rs              # whoami → org membership → direct vs PR mode
    ├── script.rs                   # Embedded script, bucket freshness check, --update-script
    └── scripts/
        └── split-model-job.sh      # Canonical job script (from hf-mesh-skippy-splitter)
```

### CLI handler

`crates/mesh-llm/src/cli/commands/model_prepare.rs` — wired into the
`Command` enum and `dispatch()` in the standard way.

### Dependencies

```toml
[package]
name = "model-prepare"
edition.workspace = true
license.workspace = true
version.workspace = true

[build-dependencies]
sha2.workspace = true

[dependencies]
anyhow.workspace = true
hf_hub = { package = "huggingface-hub", ... }   # same rev as model-hf
model-hf = { path = "../model-hf" }
model-ref = { path = "../model-ref" }
reqwest = { version = "0.12", features = ["json", "stream"] }
serde.workspace = true
serde_json.workspace = true
sha2.workspace = true
tokio = { version = "1", features = ["rt"] }
```

No dependency on `skippy-ffi`, `skippy-runtime`, gossip, QUIC, or any mesh
runtime code.

### `build.rs`

```rust
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn main() {
    let script_path = Path::new("src/scripts/split-model-job.sh");
    println!("cargo::rerun-if-changed={}", script_path.display());

    let bytes = fs::read(script_path).expect("read embedded job script");
    let hash = Sha256::digest(&bytes);
    let hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("script_hash.rs");
    fs::write(
        &dest,
        format!(
            "pub const EMBEDDED_SCRIPT_SHA256: &str = \"{hex}\";\n\
             pub const EMBEDDED_SCRIPT_SIZE: u64 = {};\n",
            bytes.len()
        ),
    )
    .expect("write generated script hash");
}
```

### Module: `jobs.rs`

```rust
pub struct HfJobsClient {
    http: reqwest::Client,
    endpoint: String,       // default "https://huggingface.co"
    token: String,
}

pub struct JobSpec {
    pub docker_image: String,
    pub command: Vec<String>,
    pub environment: HashMap<String, String>,
    pub secrets: HashMap<String, bool>,
    pub flavor: String,
    pub timeout_seconds: u64,
    pub volumes: Vec<JobVolume>,
}

pub struct JobVolume {
    pub volume_type: String,    // "model", "bucket", "dataset", "space"
    pub source: String,
    pub mount_path: String,
    pub read_only: Option<bool>,
}

pub struct JobInfo {
    pub id: String,
    pub status: JobStatus,
    pub created_at: Option<String>,
}

pub struct JobStatus {
    pub stage: JobStage,
    pub message: Option<String>,
}

pub enum JobStage {
    Running,
    Completed,
    Error,
    Canceled,
    Deleted,
}

impl HfJobsClient {
    pub fn from_env() -> Result<Self>;
    pub async fn submit(&self, namespace: &str, spec: &JobSpec) -> Result<JobInfo>;
    pub async fn inspect(&self, namespace: &str, job_id: &str) -> Result<JobInfo>;
    pub async fn logs(&self, namespace: &str, job_id: &str, follow: bool)
        -> Result<impl Stream<Item = String>>;
    pub async fn cancel(&self, namespace: &str, job_id: &str) -> Result<()>;
    pub async fn list(&self, namespace: &str) -> Result<Vec<JobInfo>>;
}
```

### Module: `permissions.rs`

```rust
pub struct PermissionCheck {
    pub username: String,
    pub is_meshllm_member: bool,
    pub namespace: String,          // "meshllm" or username
    pub catalog_create_pr: bool,    // !is_meshllm_member
}

/// Call whoami, inspect org memberships, decide direct vs PR mode.
pub async fn check_permissions(client: &HFClient) -> Result<PermissionCheck>;
```

### Module: `script.rs`

```rust
const EMBEDDED_SCRIPT: &str = include_str!("scripts/split-model-job.sh");
include!(concat!(env!("OUT_DIR"), "/script_hash.rs"));
// brings in: EMBEDDED_SCRIPT_SHA256, EMBEDDED_SCRIPT_SIZE

pub struct ScriptFreshness {
    pub bucket_size: u64,
    pub expected_size: u64,
    pub is_current: bool,
}

/// Compare the bucket script's size against the embedded version.
/// If sizes match, optionally download and SHA-256 compare.
pub async fn check_bucket_script(client: &HFClient) -> Result<ScriptFreshness>;

/// Upload the embedded script to the meshllm bucket.
/// Requires meshllm org write access.
pub async fn update_bucket_script(client: &HFClient) -> Result<()>;
```

### Module: `prepare.rs`

```rust
pub struct PrepareParams {
    pub source_repo: String,
    pub quant: Option<String>,
    pub target: Option<String>,
    pub model_id: Option<String>,
    pub flavor: String,
    pub timeout: String,
    pub mesh_llm_ref: String,
}

pub struct PrepareJob {
    pub source_repo: String,
    pub source_file: String,
    pub target_repo: String,
    pub model_id: String,
    pub namespace: String,
    pub catalog_create_pr: bool,
    pub spec: JobSpec,
}

impl PrepareJob {
    /// Resolve source files, permissions, target repo, and build the job spec.
    ///
    /// Steps:
    /// 1. List files in source_repo via HF API
    /// 2. Filter for GGUFs matching --quant (using model-ref parsing)
    /// 3. Pick the first shard (or single file)
    /// 4. Call check_permissions() for namespace and PR mode
    /// 5. Derive target repo name from distribution ID
    /// 6. Build JobSpec with env vars, volumes, secrets
    pub async fn resolve(params: PrepareParams) -> Result<Self>;
}
```

Source file resolution from `--quant`:
- List all files in the repo with `list_tree(recursive: true)`
- Filter to `*.gguf` files
- Group by distribution ID using `model_ref::normalize_gguf_distribution_id`
- If `--quant` is provided, match against the quant string (e.g. `Q4_K_M`
  matches `Qwen3-8B-Q4_K_M.gguf`, `UD-Q4_K_XL` matches
  `UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf`)
- For sharded GGUFs, pick the `-00001-of-NNNNN.gguf` shard
- Uses `model_ref::split_gguf_shard_info` and
  `model_ref::normalize_gguf_distribution_id` for parsing

If `--quant` is omitted, list the available quants and exit:

```bash
mesh-llm model-prepare unsloth/Qwen3-235B-A22B-GGUF
```

```
📦 Available quants in unsloth/Qwen3-235B-A22B-GGUF:

   UD-Q4_K_XL     3 shards, 142.7 GB
   UD-IQ2_M       6 shards,  89.3 GB
   Q4_K_M         1 file,    15.6 GB
   Q8_0           2 shards,  28.1 GB

Specify one with --quant, e.g.:
   mesh-llm model-prepare unsloth/Qwen3-235B-A22B-GGUF --quant UD-Q4_K_XL
```

This replaces the old workflow of manually browsing the HF repo page to
find the exact GGUF path and copy-pasting it into the shell script.

Target repo auto-derivation:
- Extract distribution ID from the resolved source file
  (e.g. `Qwen3-235B-A22B-UD-Q4_K_XL` from the shard filename)
- Prefix with namespace: `meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers` or
  `jsmith/Qwen3-235B-A22B-UD-Q4_K_XL-layers`

## CLI surface

### Command definition

```rust
/// Prepare a model for distributed inference by splitting it into
/// per-layer files on HF compute.
///
/// Submits an HF Job that builds skippy-model-package from source,
/// splits the model, publishes the layer package, and updates the
/// meshllm/catalog.
#[command(name = "model-prepare")]
ModelPrepare {
    /// Source HuggingFace repo (e.g. unsloth/Qwen3-235B-A22B-GGUF).
    source_repo: Option<String>,

    /// Quantization variant (e.g. UD-Q4_K_XL, Q4_K_M).
    #[arg(long)]
    quant: Option<String>,

    /// Target repo for the layer package (auto-derived if omitted).
    #[arg(long)]
    target: Option<String>,

    /// Override model ID in the manifest.
    #[arg(long)]
    model_id: Option<String>,

    /// HF Job hardware flavor [default: cpu-xl].
    #[arg(long, default_value = "cpu-xl")]
    flavor: String,

    /// Job timeout [default: 3h].
    #[arg(long, default_value = "3h")]
    timeout: String,

    /// Git ref of mesh-llm to build in the job [default: main].
    #[arg(long, default_value = "main")]
    mesh_llm_ref: String,

    /// Print the job spec without submitting.
    #[arg(long)]
    dry_run: bool,

    /// Stream job logs after submission until completion.
    #[arg(long)]
    follow: bool,

    /// Check status of a previously submitted job.
    #[arg(long)]
    status: Option<String>,

    /// Fetch logs for a previously submitted job.
    #[arg(long)]
    logs: Option<String>,

    /// Cancel a running job.
    #[arg(long)]
    cancel: Option<String>,

    /// List recent model-prepare jobs.
    #[arg(long)]
    list: bool,

    /// Upload the latest job script to the meshllm bucket (requires org access).
    #[arg(long)]
    update_script: bool,
},
```

### Usage examples

Submit a split job (meshllm member):

```bash
mesh-llm model-prepare unsloth/Qwen3-235B-A22B-GGUF --quant UD-Q4_K_XL
```

```
🔍 Resolving source...
   Repo:   unsloth/Qwen3-235B-A22B-GGUF
   File:   UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf (3 shards)

🔑 Permissions: micn (meshllm org member)
   Target:  meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
   Catalog: meshllm/catalog (direct commit)

📋 Job: cpu-xl, timeout 3h, mesh-llm@main

🚀 Submitted: job-abc123def456
   Status:  mesh-llm model-prepare --status job-abc123def456
   Logs:    mesh-llm model-prepare --logs job-abc123def456
```

Submit a split job (non-member):

```bash
mesh-llm model-prepare unsloth/Qwen3-235B-A22B-GGUF --quant UD-Q4_K_XL
```

```
🔍 Resolving source...
   Repo:   unsloth/Qwen3-235B-A22B-GGUF
   File:   UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf (3 shards)

🔑 Permissions: jsmith (not in meshllm org)
   Target:  jsmith/Qwen3-235B-A22B-UD-Q4_K_XL-layers
   Catalog: meshllm/catalog (will open PR)

📋 Job: cpu-xl, timeout 3h, mesh-llm@main

🚀 Submitted: job-789xyz
   Status:  mesh-llm model-prepare --status job-789xyz
   Logs:    mesh-llm model-prepare --logs job-789xyz
```

Stale script warning (meshllm member):

```bash
mesh-llm model-prepare unsloth/Qwen3-8B-GGUF --quant Q4_K_M
```

```
⚠ Bucket script is out of date (bucket: 8341 bytes, expected: 9009 bytes)
  Run: mesh-llm model-prepare --update-script

🔍 Resolving source...
...
```

Stale script warning (non-member):

```
⚠ Bucket script may be out of date (bucket: 8341 bytes, expected: 9009 bytes).
  The meshllm team needs to update it.

🔍 Resolving source...
...
```

Dry run:

```bash
mesh-llm model-prepare unsloth/Qwen3-8B-GGUF --quant Q4_K_M --dry-run
```

Update the bucket script:

```bash
mesh-llm model-prepare --update-script
```

Check job status:

```bash
mesh-llm model-prepare --status job-abc123def456
```

Follow job logs:

```bash
mesh-llm model-prepare --logs job-abc123def456
```

Cancel a running job:

```bash
mesh-llm model-prepare --cancel job-abc123def456
```

List recent jobs:

```bash
mesh-llm model-prepare --list
```

## Job script changes

The embedded copy of `split-model-job.sh` (sourced from
`hf-mesh-skippy-splitter`, which has the latest version) gains one addition:
the catalog update step and model card upload step respect
`CATALOG_CREATE_PR`:

```python
create_pr = os.environ.get('CATALOG_CREATE_PR', 'false').lower() == 'true'

api.upload_file(
    repo_id=catalog_repo,
    path_or_fileobj=tmp_path,
    path_in_repo=entry_path,
    repo_type="dataset",
    commit_message=f"Add layer package for {model_id} ({target_repo})",
    create_pr=create_pr,
)
```

The same `create_pr` flag is also used for the model card upload and target
repo creation, in case the user's token can't write directly to a meshllm
repo.

The local repo copy (`hf-mesh-skippy-splitter`) currently has a newer script
than the bucket — it adds dict-style catalog handling and a model card step
that the bucket version lacks. The embedded copy in `model-prepare` should
start from the local repo's version (the latest).

## Implementation order

1. Create branch `micn/split-publish`
2. Create `crates/model-prepare/` with `Cargo.toml`, `build.rs`, module stubs
3. Copy `split-model-job.sh` from `hf-mesh-skippy-splitter` into
   `src/scripts/`, add `CATALOG_CREATE_PR` support
4. Implement `jobs.rs` — HF Jobs REST client
5. Implement `permissions.rs` — whoami + org check
6. Implement `script.rs` — embedded hash, freshness check, `--update-script`
7. Implement `prepare.rs` — source resolution, target derivation, job spec
8. Add CLI handler `model_prepare.rs`, wire into `Command` and `dispatch`
9. Add `model-prepare` to workspace `Cargo.toml`
10. `cargo check`, `cargo fmt`, verify CLI parses
