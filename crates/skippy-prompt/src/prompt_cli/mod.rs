use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, VecDeque},
    fs,
    hash::{Hash, Hasher},
    io::{self, BufRead, BufReader, IsTerminal, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mesh_client::models::gguf::{scan_gguf_compact_meta, GgufCompactMeta};
use openai_frontend::{normalize_reasoning_template_options, ReasoningConfig};
use rustyline::{error::ReadlineError, DefaultEditor};
use serde_json::Value;
use skippy_protocol::binary::{
    recv_reply, state_flags, write_stage_message, StageReplyStats, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind, LLAMA_TOKEN_NULL,
    READY_MAGIC,
};
use skippy_protocol::{
    FlashAttentionType as StageFlashAttentionType, LoadMode, PeerConfig, StageConfig,
    StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::{
    package::{inspect_layer_package, materialize_layer_package, PackageStageRequest},
    restore_native_logs, suppress_native_logs, ChatTemplateMessage, ChatTemplateOptions, ModelInfo,
    RuntimeConfig, RuntimeLoadMode, StageModel, StageSession, GGML_TYPE_F16,
};
use skippy_topology::{
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits, BoundaryDecision,
    NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
};

const DEFAULT_MIN_WINNER_COUNT: u32 = 2;
const DEFAULT_MIN_CONFIDENCE: f32 = 0.55;
const DEFAULT_MIN_MARGIN: u32 = 1;
const DEFAULT_CONFIDENCE_STEP: f32 = 0.0;
const DEFAULT_CONFIDENCE_STEP_TOKENS: usize = usize::MAX;
const DEFAULT_MAX_CONFIDENCE: f32 = 0.95;
const DEFAULT_COUNT_STEP_TOKENS: usize = usize::MAX;
const DEFAULT_MARGIN_STEP_TOKENS: usize = usize::MAX;
const DEFAULT_MESH_CTX_SIZE: u32 = 4096;
const DEFAULT_MESH_PROMPT_MAX_NEW_TOKENS: usize = 0;
const PROMPT_EXACT_PREFIX_RESTORE_MIN_TOKENS: usize = 512;

include!("args.rs");
include!("command.rs");
include!("launch.rs");
include!("interrupt.rs");
include!("binary_repl.rs");
include!("logs.rs");
include!("prompt_format.rs");
include!("generation.rs");
include!("live_session.rs");
include!("speculative.rs");
include!("wire_messages.rs");
include!("draft.rs");
include!("history.rs");
include!("stage_config.rs");
include!("remote_sync.rs");
include!("formatting.rs");
include!("topology.rs");
include!("tests.rs");
