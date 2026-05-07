// ============================================================
// STATUS API TYPES (GET /api/status + SSE /api/events)
// ============================================================

export interface GpuInfo {
  idx: number
  name: string
  total_vram_gb: number
  vram_bytes?: number
  reserved_bytes?: number
  used_vram_gb?: number
  free_vram_gb?: number
  temperature?: number
  utilization?: number
  bandwidth_gbps?: number
  mem_bandwidth_gbps?: number
}

export interface ServingModel {
  name: string
  node_id: string
  status: 'warm' | 'loading' | 'unloading'
  loaded_at?: string
  vram_gb?: number
}

export type ServingModelEntry = ServingModel | string

export interface ModelCapabilities {
  vision: boolean
  moe: boolean
  [capability: string]: boolean | undefined
}

export interface MeshModelRaw {
  name: string
  status: 'warm' | 'cold'
  size_gb?: number
  node_count: number
  capabilities?: ModelCapabilities
  quantization: string
  context_length?: number
  family?: string
  tags?: string[]
  params_b?: number
  disk_gb?: number
  moe?: boolean
  vision?: boolean
  license?: string
}

export interface PeerInfo {
  node_id?: string
  id?: string
  hostname?: string
  region?: string
  node_state?: 'client' | 'standby' | 'loading' | 'serving'
  state?: 'client' | 'standby' | 'loading' | 'serving' | string
  role?: string
  serving_models?: string[]
  hosted_models?: string[]
  models?: string[]
  my_vram_gb?: number
  vram_gb?: number
  latency_ms?: number
  rtt_ms?: number
  load_pct?: number
  version?: string
  share_pct?: number
  tok_per_sec?: number
  hardware_label?: string
  owner?: string | { status?: string; verified?: boolean; name?: string; display_name?: string }
  gpus?: GpuInfo[]
}

export type MeshPublicationState = 'private' | 'public' | 'publish_failed'

export interface StatusPayload {
  node_id: string
  node_state: 'client' | 'standby' | 'loading' | 'serving'
  model_name: string
  peers: PeerInfo[]
  models: MeshModelRaw[]
  my_vram_gb: number
  api_port?: number
  gpus: GpuInfo[]
  serving_models: ServingModelEntry[]
  hostname?: string
  my_hostname?: string
  region?: string
  version?: string
  token?: string
  uptime_s?: number
  load_pct?: number
  tok_per_sec?: number
  inflight_requests?: number
  mesh_id?: string
  owner?: PeerInfo['owner']
  nostr_discovery?: boolean
  publication_state?: MeshPublicationState
}

// ============================================================
// RUNTIME API TYPES (GET /api/runtime/llama + SSE /api/runtime/events)
// ============================================================

export type LlamaRuntimeEndpointStatus = 'ready' | 'error' | 'unavailable' | string

export interface LlamaRuntimeMetricSample {
  name: string
  labels?: Record<string, string>
  value: number
}

export interface LlamaRuntimeSlotItem {
  index: number
  id?: number
  id_task?: number
  n_ctx?: number
  is_processing: boolean
}

export interface LlamaRuntimeMetricsPayload {
  status: LlamaRuntimeEndpointStatus
  last_attempt_unix_ms?: number
  last_success_unix_ms?: number
  error?: string
  raw_text?: string
  samples?: LlamaRuntimeMetricSample[]
}

export interface LlamaRuntimeSlotsPayload {
  status: LlamaRuntimeEndpointStatus
  last_attempt_unix_ms?: number
  last_success_unix_ms?: number
  error?: string
  slots?: {
    index?: number
    id?: number
    id_task?: number
    n_ctx?: number
    speculative?: boolean
    is_processing?: boolean
  }[]
}

export interface LlamaRuntimeItemsPayload {
  metrics: LlamaRuntimeMetricSample[]
  slots: LlamaRuntimeSlotItem[]
  slots_total: number
  slots_busy: number
}

export interface LlamaRuntimePayload {
  metrics: LlamaRuntimeMetricsPayload
  slots: LlamaRuntimeSlotsPayload
  items?: LlamaRuntimeItemsPayload
}

// ============================================================
// MODELS API TYPES (GET /api/models)
// ============================================================

export interface ModelsResponse {
  mesh_models: MeshModelRaw[]
}

// ============================================================
// CHAT STREAMING TYPES (POST /api/responses → SSE stream)
// ============================================================

export interface ResponsesInputTextBlock {
  type: 'input_text'
  text: string
}

export interface ResponsesInputImageBlock {
  type: 'input_image'
  image_url: string
}

export interface ResponsesInputAudioBlock {
  type: 'input_audio'
  audio_url: string
}

export interface ResponsesInputFileBlock {
  type: 'input_file'
  url: string
  mime_type?: string
  file_name?: string
}

export type ResponsesInputContentBlock =
  | ResponsesInputTextBlock
  | ResponsesInputImageBlock
  | ResponsesInputAudioBlock
  | ResponsesInputFileBlock

export interface ResponsesInputMessage {
  role: 'system' | 'user' | 'assistant'
  content: ResponsesInputContentBlock[] | string
}

export interface ResponsesRequest {
  model: string
  client_id: string
  request_id: string
  input: ResponsesInputMessage[]
  stream: boolean
  stream_options?: { include_usage: boolean }
}

export interface ChatSSEDeltaEvent {
  type: 'response.output_text.delta'
  delta: string
  response_id?: string
  output_index?: number
  content_index?: number
}

export interface ChatUsage {
  input_tokens: number
  output_tokens: number
  total_tokens?: number
}

export interface ChatTimings {
  decode_time_ms?: number
  ttft_ms?: number
  total_time_ms?: number
}

export interface ChatSSECompletedEvent {
  type: 'response.completed'
  response: {
    id?: string
    model?: string
    usage: ChatUsage
    timings?: ChatTimings
    served_by?: string
  }
}

export type ChatSSEEvent = ChatSSEDeltaEvent | ChatSSECompletedEvent

// ============================================================
// ATTACHMENT / OBJECTS API TYPES (POST /api/objects)
// ============================================================

export interface AttachmentUploadRequest {
  request_id: string
  mime_type: string
  file_name: string
  bytes_base64: string
}

export interface AttachmentUploadResponse {
  token: string
}
