export type LiveNodeState = "client" | "standby" | "loading" | "serving";

export type WakeableNodeState = "sleeping" | "waking";

export const LIVE_NODE_STATE_LABELS: Record<LiveNodeState, string> = {
  client: "Client",
  standby: "Standby",
  loading: "Loading",
  serving: "Serving",
};

export type MeshModel = {
  name: string;
  display_name?: string;
  status: "warm" | "cold" | string;
  node_count: number;
  mesh_vram_gb?: number;
  size_gb: number;
  architecture?: string;
  context_length?: number;
  quantization?: string;
  description?: string;
  multimodal?: boolean;
  multimodal_status?: "supported" | "none" | string;
  vision?: boolean;
  vision_status?: "supported" | "likely" | "none" | string;
  audio?: boolean;
  audio_status?: "supported" | "likely" | "none" | string;
  reasoning?: boolean;
  reasoning_status?: "supported" | "likely" | "none" | string;
  tool_use?: boolean;
  tool_use_status?: "supported" | "likely" | "none" | string;
  moe?: boolean;
  expert_count?: number;
  used_expert_count?: number;
  ranking_source?: string;
  ranking_origin?: string;
  ranking_prompt_count?: number;
  ranking_tokens?: number;
  ranking_layer_scope?: string;
  draft_model?: string;
  source_page_url?: string;
  fit_label?: string;
  fit_detail?: string;
  download_command?: string;
  run_command?: string;
  auto_command?: string;
  request_count?: number;
  last_active_secs_ago?: number;
  source_ref?: string;
  source_revision?: string;
  source_file?: string;
  active_nodes?: string[];
};

export type Ownership = {
  owner_id?: string;
  cert_id?: string;
  status: string;
  verified: boolean;
  expires_at_unix_ms?: number;
  node_label?: string;
  hostname_hint?: string;
};

export type Peer = {
  id: string;
  owner?: Ownership;
  role: string;
  state: LiveNodeState;
  models: string[];
  available_models?: string[];
  requested_models?: string[];
  vram_gb: number;
  serving_models?: string[];
  hosted_models?: string[];
  hosted_models_known?: boolean;
  rtt_ms?: number | null;
  hostname?: string;
  version?: string;
  is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
  first_joined_mesh_ts?: number;
};

export type LocalInstance = {
  pid: number;
  api_port: number | null;
  version: string | null;
  started_at_unix: number;
  runtime_dir: string;
  is_self: boolean;
};

export type WakeableNode = {
  logical_id: string;
  models: string[];
  vram_gb: number;
  provider?: string;
  state: WakeableNodeState;
  wake_eta_secs?: number;
};

export type StatusPayload = {
  version?: string;
  latest_version?: string | null;
  node_id: string;
  owner?: Ownership;
  token: string;
  node_state: LiveNodeState;
  node_status: string;
  is_host: boolean;
  is_client: boolean;
  llama_ready: boolean;
  model_name: string;
  models?: string[];
  available_models?: string[];
  requested_models?: string[];
  serving_models?: string[];
  hosted_models?: string[];
  api_port: number;
  my_vram_gb: number;
  model_size_gb: number;
  mesh_name?: string | null;
  peers: Peer[];
  wakeable_nodes?: WakeableNode[];
  local_instances?: LocalInstance[];
  inflight_requests: number;
  launch_pi?: string | null;
  launch_goose?: string | null;
  nostr_discovery?: boolean;
  /** Best-effort publication state per Issue #240: private | public | publish_failed */
  publication_state?: 'private' | 'public' | 'publish_failed';
  my_hostname?: string;
  my_is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
  first_joined_mesh_ts?: number;
};

export type ModelServingStat = {
  nodes: number;
  vramGb: number;
};

export type ThemeMode = "auto" | "light" | "dark";
