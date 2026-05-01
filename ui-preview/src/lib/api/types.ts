// ============================================================
// STATUS API TYPES (GET /api/status + SSE /api/events)
// ============================================================

export interface GpuInfo {
  idx: number
  name: string
  total_vram_gb: number
  used_vram_gb?: number
  free_vram_gb?: number
  temperature?: number
  utilization?: number
}

export interface ServingModel {
  name: string
  node_id: string
  status: 'warm' | 'loading' | 'unloading'
  loaded_at?: string
  vram_gb?: number
}

export interface ModelCapabilities {
  vision: boolean
  moe: boolean
}

export interface MeshModelRaw {
  name: string
  status: 'warm' | 'cold'
  size_gb: number
  node_count: number
  capabilities: ModelCapabilities
  quantization: string
  context_length: number
  family?: string
  tags?: string[]
  params_b?: number
  disk_gb?: number
}

export interface PeerInfo {
  node_id: string
  hostname: string
  region?: string
  node_state: 'client' | 'standby' | 'loading' | 'serving'
  serving_models: string[]
  my_vram_gb?: number
  latency_ms?: number
  load_pct?: number
  version?: string
  share_pct?: number
  tok_per_sec?: number
  hardware_label?: string
  owner?: string
}

export interface StatusPayload {
  node_id: string
  node_state: 'client' | 'standby' | 'loading' | 'serving'
  model_name: string
  peers: PeerInfo[]
  models: MeshModelRaw[]
  my_vram_gb: number
  gpus: GpuInfo[]
  serving_models: ServingModel[]
  hostname?: string
  region?: string
  version?: string
  uptime_s?: number
  load_pct?: number
  tok_per_sec?: number
  total_requests?: number
  active_requests?: number
  mesh_id?: string
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
  url: string
}

export interface ResponsesInputAudioBlock {
  type: 'input_audio'
  url: string
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
  role: 'user' | 'assistant'
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
