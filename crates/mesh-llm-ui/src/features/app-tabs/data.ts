import { createElement } from 'react'
import { Activity, Cpu, HardDrive, Hash, Network, UserRound, type LucideIcon } from 'lucide-react'
import type {
  ChatHarnessData,
  ConfigAssign,
  ConfigModel,
  ConfigNode,
  ConfigurationHarnessData,
  Conversation,
  DashboardHarnessData,
  Decision,
  MeshNode,
  ModelSummary,
  Peer,
  PeerSummary,
  ShellHarnessData,
  StatusMetric,
  ThreadMessage,
  TomlValidationWarning,
  TransparencyMessage,
  TransparencyNode
} from '@/features/app-tabs/types'
import { env } from '@/lib/env'

export const APP_STORAGE_KEYS = {
  featureFlagOverrides: `${env.storageNamespace}:feature-flags:v1`,
  chatSystemPrompt: `${env.storageNamespace}:chat-system-prompt:v1`,
  preferences: `${env.storageNamespace}:preferences:v1`
}

const metricIcon = (Icon: LucideIcon) => createElement(Icon, { className: 'size-[11px] shrink-0', 'aria-hidden': true })

export const STATUS_METRICS: StatusMetric[] = [
  {
    id: 'node-id',
    icon: metricIcon(Hash),
    label: 'Node ID',
    value: '990232e1c1',
    variant: 'identity',
    mono: true,
    badge: { label: 'Serving', tone: 'good' }
  },
  {
    id: 'owner',
    icon: metricIcon(UserRound),
    label: 'Owner',
    value: 'Unsigned',
    variant: 'identity',
    mono: true,
    badge: { label: 'not cryptographically bound', tone: 'muted' }
  },
  { id: 'nodes', icon: metricIcon(Network), label: 'Nodes', value: 3, meta: '1 you · 2 peers' },
  { id: 'active-models', icon: metricIcon(Cpu), label: 'Active models', value: 6, meta: '2 loaded locally · 4 remote' },
  {
    id: 'mesh-vram',
    icon: metricIcon(HardDrive),
    label: 'Mesh VRAM',
    value: '160.5',
    unit: 'GB',
    meta: '57% free',
    sparkline: [12, 14, 11, 16, 13, 15, 17, 14, 18]
  },
  {
    id: 'inflight',
    icon: metricIcon(Activity),
    label: 'Inflight',
    value: 0,
    meta: '0 requests in flight',
    sparkline: [4, 8, 5, 12, 6, 14, 7, 9, 5]
  }
]
export const PEERS: Peer[] = [
  {
    id: 'p1',
    shortId: '990232e1c1',
    hostname: 'carrack',
    region: 'tor-1',
    status: 'online',
    role: 'you',
    version: '0.64.0',
    hostedModels: ['Qwen3.6-27B-UD', 'Qwen3.5-4B-UD'],
    sharePct: 38,
    latencyMs: 1,
    vramGB: 61.7,
    toksPerSec: 36.8,
    loadPct: 38,
    hardwareLabel: 'Jetson AGX Orin · 61 GB',
    ownership: 'Unsigned',
    owner: 'Unsigned'
  },
  {
    id: 'p2',
    shortId: 'e5c42cc0ad',
    hostname: 'lemony-28',
    region: 'nyc-2',
    status: 'online',
    role: 'host',
    version: '0.64.0',
    hostedModels: ['Qwen3.6-35B-A3B-UD', 'gemma-4-26B-A4B-it-UD'],
    sharePct: 31,
    latencyMs: 1,
    vramGB: 49.4,
    toksPerSec: 22.4,
    loadPct: 31,
    hardwareLabel: 'Jetson AGX Orin · 61 GB',
    ownership: 'Unsigned',
    owner: 'Unsigned'
  },
  {
    id: 'p3',
    shortId: '7d13fd27b8',
    hostname: 'lemony-29',
    region: 'sfo-1',
    status: 'online',
    role: 'host',
    version: '0.64.0',
    hostedModels: ['Qwen3.5-0.8B-UD', 'Qwen3.5-2B'],
    sharePct: 31,
    latencyMs: 1.1,
    vramGB: 49.4,
    toksPerSec: 14.2,
    loadPct: 31,
    hardwareLabel: 'Mac Studio M2 Ultra · 64 GB',
    ownership: 'Unsigned',
    owner: 'Unsigned'
  }
]
export const PEER_SUMMARY: PeerSummary = { total: 3, online: 3, capacity: 'all serving' }
export const MODELS: ModelSummary[] = [
  {
    name: 'gemma-4-26B-A4B-it-UD',
    fullId: 'gemma-4-26B-A4B-it-UD-Q4_K_XL',
    family: 'Gemma',
    familyColor: 'family-5',
    paramsB: 26,
    paramsLabel: '26B A4B',
    quant: 'Q4_K_XL',
    size: '14.8 GB',
    sizeGB: 14.8,
    diskGB: 16,
    context: '64k',
    ctxMaxK: 64,
    ctxPerGB: 0.021,
    moe: false,
    vision: false,
    status: 'warm',
    tags: ['Text'],
    nodeCount: 1,
    activitySummary: '0 requests seen · active 4 min ago'
  },
  {
    name: 'Qwen3.5-0.8B-UD',
    fullId: 'Qwen3.5-0.8B-UD-Q4_K_XL',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 0.8,
    paramsLabel: '0.8B UD',
    quant: 'Q4_K_XL',
    size: '0.6 GB',
    sizeGB: 0.6,
    diskGB: 0.8,
    context: '32k',
    ctxMaxK: 32,
    ctxPerGB: 0.004,
    moe: false,
    vision: false,
    status: 'warm',
    tags: ['Text'],
    nodeCount: 1,
    activitySummary: '0 requests seen · active 4 min ago'
  },
  {
    name: 'Qwen3.5-2B',
    fullId: 'Qwen3.5-2B-Q4_K_M',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 2,
    paramsLabel: '2B',
    quant: 'Q4_K_M',
    size: '1.3 GB',
    sizeGB: 1.3,
    diskGB: 1.6,
    context: '32k',
    ctxMaxK: 32,
    ctxPerGB: 0.008,
    moe: false,
    vision: false,
    status: 'warm',
    tags: ['Text'],
    nodeCount: 1,
    activitySummary: '0 requests seen · active 4 min ago'
  },
  {
    name: 'Qwen3.5-4B-UD',
    fullId: 'Qwen3.5-4B-UD-Q4_K_XL',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 4,
    paramsLabel: '4B UD',
    quant: 'Q4_K_XL',
    size: '2.9 GB',
    sizeGB: 2.9,
    diskGB: 3.2,
    context: '32k',
    ctxMaxK: 32,
    ctxPerGB: 0.012,
    moe: false,
    vision: false,
    status: 'warm',
    tags: ['Text'],
    nodeCount: 1,
    activitySummary: '0 requests seen · active 4 min ago'
  },
  {
    name: 'Qwen3.6-27B-UD',
    fullId: 'Qwen3.6-27B-UD-Q4_K_XL',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 27,
    paramsLabel: '27B UD',
    quant: 'Q4_K_XL',
    size: '17.8 GB',
    sizeGB: 17.8,
    diskGB: 19,
    context: '256k',
    ctxMaxK: 256,
    ctxPerGB: 0.025,
    moe: false,
    vision: false,
    status: 'warm',
    tags: ['Text'],
    nodeCount: 1,
    activitySummary: '0 requests seen · active 4 min ago'
  },
  {
    name: 'Qwen3.6-35B-A3B-UD',
    fullId: 'Qwen3.6-35B-A3B-UD-Q4_K_XL',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 35,
    paramsLabel: '35B A3B',
    quant: 'Q4_K_XL',
    size: '22.1 GB',
    sizeGB: 22.1,
    diskGB: 24,
    context: '256k',
    ctxMaxK: 256,
    ctxPerGB: 0.026,
    moe: true,
    vision: false,
    status: 'warm',
    tags: ['Text'],
    nodeCount: 1,
    activitySummary: '0 requests seen · active 4 min ago'
  }
]
export const MESH_NODES: MeshNode[] = [
  {
    id: 'self',
    peerId: 'p1',
    label: 'CARRACK',
    subLabel: 'SERVING · YOU',
    x: 58,
    y: 52,
    status: 'online',
    role: 'self'
  },
  { id: 'lemony', peerId: 'p2', label: 'LEMONY-28', subLabel: 'SERVING', x: 36, y: 26, status: 'online' },
  { id: 'lemony-29', peerId: 'p3', label: 'LEMONY-29', subLabel: 'SERVING', x: 28, y: 76, status: 'online' }
]
export const CONVERSATIONS: Conversation[] = [
  {
    id: 'c1',
    title: 'Routing latency notes',
    subtitle: 'Inspect why TTFT rose in tor-1',
    updatedAt: '09:42',
    createdAt: Date.now(),
    messages: [],
    active: true
  },
  {
    id: 'c2',
    title: 'Model capacity draft',
    subtitle: 'Plan pooled placement for coder stack',
    updatedAt: 'Yesterday',
    createdAt: Date.now(),
    messages: []
  }
]
export const TRANSPARENCY_NODES: TransparencyNode[] = [
  { id: 'desk', label: 'YOU', region: 'local', status: 'online' },
  { id: 'carrack', label: 'CARRACK', region: 'tor-1', status: 'online', isLocal: true },
  { id: 'lemony', label: 'LEMONY-28', region: 'nyc-2', status: 'online' },
  { id: 'lemony-29', label: 'LEMONY-29', region: 'sfo-1', status: 'online' }
]
export const TRANSPARENCY_MESSAGE: TransparencyMessage = {
  kind: 'assistant',
  id: 'msg-a1',
  text: 'Here are three revisions with different tones — playful, serious, and technical. Want me to expand any of them?',
  at: '14:53',
  servedBy: 'lemony-28',
  route: ['desk', 'carrack', 'lemony'],
  model: 'Qwen3.6-35B-A3B-UD',
  receipt: 'rx-92b7',
  metrics: { rttMs: 1, ttftMs: 312, throughput: '22.4 tok/s', tokens: 148 },
  decisions: [
    { id: 'fit', ok: true, label: 'Qwen3.6-35B-A3B-UD warm', detail: 'lemony-28 · 22.1 GB loaded' },
    { id: 'skip', ok: false, label: 'carrack skipped', detail: 'not enough VRAM headroom · 4.1 GB free' },
    { id: 'link', ok: true, label: 'Link healthy', detail: '0.8ms RTT · 0% loss · 1.2Gbps' },
    {
      id: 'policy',
      ok: true,
      label: 'Prompt > 20 tokens → remote',
      detail: 'policy: route big prompts to dedicated node'
    }
  ],
  trace: [
    { id: 'queue', label: 'Queue', ms: 14, tone: 'neutral' },
    { id: 'route', label: 'Route', ms: 22, tone: 'neutral' },
    { id: 'prefill', label: 'Prefill', ms: 290, tone: 'warn' },
    { id: 'decode', label: 'Decode', ms: 6607, tone: 'good' }
  ]
}
export const CFG_NODES: ConfigNode[] = [
  {
    id: 'node-a',
    hostname: 'carrack',
    region: 'tor-1',
    status: 'online',
    cpu: 'Ryzen 7950X',
    ramGB: 128,
    placement: 'separate',
    gpus: [
      { idx: 0, name: 'RTX 5090', totalGB: 34.2, reservedGB: 0.9 },
      { idx: 1, name: 'RTX 6000 Pro', totalGB: 48.0, reservedGB: 1.1 },
      { idx: 2, name: 'RTX 6000 Pro', totalGB: 48.0, reservedGB: 1.1 },
      { idx: 3, name: 'RTX 6000 Pro', totalGB: 48.0, reservedGB: 1.1 },
      { idx: 4, name: 'RTX 6000 Pro', totalGB: 48.0, reservedGB: 1.1 },
      { idx: 5, name: 'RTX 6000 Pro', totalGB: 48.0, reservedGB: 1.1 },
      { idx: 6, name: 'RTX 6000 Pro', totalGB: 48.0, reservedGB: 1.1 },
      { idx: 7, name: 'RTX 3080', totalGB: 10.7, reservedGB: 0.6 }
    ]
  },
  {
    id: 'node-b',
    hostname: 'perseus.local',
    region: 'unified',
    status: 'online',
    cpu: 'Apple M4 Pro',
    ramGB: 64,
    placement: 'pooled',
    memoryTopology: 'unified',
    gpus: [{ idx: 0, name: 'unified memory', totalGB: 51.5, reservedGB: 1.5 }]
  },
  {
    id: 'node-c',
    hostname: 'triton.lab',
    region: 'sfo-1',
    status: 'offline',
    cpu: 'Xeon W',
    ramGB: 96,
    placement: 'separate',
    gpus: [{ idx: 0, name: 'RTX 4090', totalGB: 24.0, reservedGB: 0.8 }]
  }
]
export const CFG_CATALOG: ConfigModel[] = [
  {
    id: 'glm47',
    name: 'GLM-4.7-Flash-Q4_K_M',
    family: 'GLM',
    familyColor: 'family-0',
    paramsB: 4.7,
    paramsLabel: '~70B',
    quant: 'Q4_K_M',
    sizeGB: 18.5,
    diskGB: 19,
    ctxMaxK: 128,
    ctxPerGB: 0.017,
    layers: 80,
    heads: 40,
    embed: 8192,
    tokenizer: 'glm',
    moe: false,
    vision: false,
    tags: ['chat']
  },
  {
    id: 'llama70',
    name: 'Llama-3.3-70B-Q4_K_M',
    family: 'Llama',
    familyColor: 'family-1',
    paramsB: 70,
    paramsLabel: '70B',
    quant: 'Q4_K_M',
    sizeGB: 40.3,
    diskGB: 46,
    ctxMaxK: 256,
    ctxPerGB: 0.019,
    layers: 80,
    heads: 64,
    embed: 8192,
    tokenizer: 'llama',
    moe: false,
    vision: false,
    tags: ['chat', 'tools']
  },
  {
    id: 'qwen27',
    name: 'Qwen3.5-27B-Q4_K_M',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 27,
    paramsLabel: '27B',
    quant: 'Q4_K_M',
    sizeGB: 17.4,
    diskGB: 19,
    ctxMaxK: 64,
    ctxPerGB: 0.022,
    layers: 64,
    heads: 40,
    embed: 5120,
    tokenizer: 'qwen',
    moe: false,
    vision: false,
    tags: ['chat']
  },
  {
    id: 'qwen4',
    name: 'Qwen3-4B-Q4_K_M',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 4,
    paramsLabel: '4B',
    quant: 'Q4_K_M',
    sizeGB: 2.6,
    diskGB: 3,
    ctxMaxK: 32,
    ctxPerGB: 0.018,
    layers: 36,
    heads: 28,
    embed: 3584,
    tokenizer: 'qwen',
    moe: false,
    vision: false,
    tags: ['chat']
  },
  {
    id: 'qwenud',
    name: 'Qwen3.5-27B-UD-Q4_K_XL',
    family: 'Qwen',
    familyColor: 'family-2',
    paramsB: 27,
    paramsLabel: '27B UD',
    quant: 'Q4_K_XL',
    sizeGB: 17.8,
    diskGB: 19,
    ctxMaxK: 256,
    ctxPerGB: 0.025,
    layers: 64,
    heads: 40,
    embed: 5120,
    tokenizer: 'qwen',
    moe: false,
    vision: false,
    tags: ['chat']
  },
  {
    id: 'mixtral',
    name: 'mixtral-8x22b',
    family: 'Mixtral',
    familyColor: 'family-3',
    paramsB: 176,
    paramsLabel: '8x22B MoE',
    quant: 'Q4_K_M',
    sizeGB: 86,
    diskGB: 91,
    ctxMaxK: 64,
    ctxPerGB: 0.028,
    layers: 56,
    heads: 48,
    embed: 6144,
    tokenizer: 'mixtral',
    moe: true,
    vision: false,
    tags: ['moe']
  },
  {
    id: 'llava',
    name: 'llava-next-34b',
    family: 'LLaVA',
    familyColor: 'family-4',
    paramsB: 34,
    paramsLabel: '34B',
    quant: 'Q4_K_M',
    sizeGB: 22,
    diskGB: 26,
    ctxMaxK: 32,
    ctxPerGB: 0.024,
    layers: 48,
    heads: 52,
    embed: 7168,
    tokenizer: 'llava',
    moe: false,
    vision: true,
    tags: ['vision']
  },
  {
    id: 'phi4',
    name: 'phi-4-mini',
    family: 'Phi',
    familyColor: 'family-6',
    paramsB: 3.8,
    paramsLabel: '3.8B',
    quant: 'Q8_0',
    sizeGB: 5.2,
    diskGB: 6.1,
    ctxMaxK: 32,
    ctxPerGB: 0.014,
    layers: 32,
    heads: 32,
    embed: 3072,
    tokenizer: 'phi',
    moe: false,
    vision: false,
    tags: ['small']
  }
]
export const INITIAL_ASSIGNS: ConfigAssign[] = [
  { id: 'a1', modelId: 'glm47', nodeId: 'node-a', containerIdx: 0, ctx: 16384 },
  { id: 'a2', modelId: 'llama70', nodeId: 'node-a', containerIdx: 1, ctx: 16384 },
  { id: 'a3', modelId: 'qwen27', nodeId: 'node-a', containerIdx: 2, ctx: 16384 },
  { id: 'a4', modelId: 'qwen4', nodeId: 'node-a', containerIdx: 7, ctx: 4096 },
  { id: 'a5', modelId: 'qwenud', nodeId: 'node-b', containerIdx: 0, ctx: 262144 }
]

const DRAFT_MODEL_SPECULATION_DEPENDENCY = {
  settingId: 'speculation-mode',
  condition: (value: string) => value === 'draft'
}

const THROUGHPUT_TOML_SECTION = 'defaults.throughput'

const HARDWARE_TOML_SECTION = 'defaults.hardware'

const MODEL_FIT_TOML_SECTION = 'defaults.model_fit'

const SKIPPY_TRANSPORT_TOML_SECTION = 'defaults.skippy'

const REQUEST_DEFAULTS_TOML_SECTION = 'defaults.request_defaults'

const MULTIMODAL_TOML_SECTION = 'defaults.multimodal'

const ADVANCED_SERVER_TOML_SECTION = 'defaults.advanced.server'

const MIROSTAT_MODE_DEPENDENCY = {
  settingId: 'mirostat-mode',
  condition: (value: string) => value !== 'disabled'
}

const PREFILL_CHUNKING_FIXED_DEPENDENCY = {
  settingId: 'prefill-chunking',
  condition: (value: string) => value === 'fixed'
}

const PREFILL_CHUNKING_SCHEDULE_DEPENDENCY = {
  settingId: 'prefill-chunking',
  condition: (value: string) => value === 'schedule'
}

export const CONFIGURATION_DEFAULTS = {
  categories: [
    {
      id: 'runtime',
      label: 'Runtime',
      summary: 'Model fit, hardware, and throughput defaults.',
      help: 'Load-time runtime behavior and concurrency defaults'
    },
    {
      id: 'memory',
      label: 'Memory',
      summary: 'KV cache policy and fit headroom.',
      help: 'VRAM accounting and fit headroom'
    },
    {
      id: 'speculative-decoding',
      label: 'Speculative Decoding',
      summary: 'Draft acceleration defaults.',
      help: 'Speculative draft policy defaults',
      tomlSection: 'defaults.speculative'
    },
    {
      id: 'advanced',
      label: 'Reasoning',
      summary: 'Reasoning and repetition controls.',
      help: 'Reasoning and sampling defaults'
    },
    {
      id: 'request-defaults',
      label: 'Request Defaults',
      summary: 'Sampling, reasoning, and request-time fallback defaults.',
      help: 'Request-time sampling and reasoning defaults'
    },
    {
      id: 'skippy-transport',
      label: 'Skippy Transport',
      summary: 'Activation wire dtype, prefill chunking, and lifecycle timing.',
      help: 'Stage transport, chunking, and lifecycle defaults',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION
    },
    {
      id: 'multimodal',
      label: 'Multimodal',
      summary: 'Projector and image token defaults.',
      help: 'Vision projector and image token defaults',
      tomlSection: MULTIMODAL_TOML_SECTION
    },
    {
      id: 'advanced-server',
      label: 'Advanced Server',
      summary: 'Server identity and operator overrides.',
      help: 'Advanced server defaults and identity overrides',
      tomlSection: ADVANCED_SERVER_TOML_SECTION
    }
  ],
  settings: [
    {
      id: 'threads',
      categoryId: 'runtime',
      icon: 'cpu',
      label: 'CPU threads',
      description:
        'Sets the default CPU thread count. Use 0 for auto; 256 is a safe UI ceiling for general-purpose systems.',
      inheritedLabel: 'Inherited by placements without a thread override',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'threads', value: '0', min: 0, max: 256, step: 1, unit: 'threads' }
    },
    {
      id: 'threads-batch',
      categoryId: 'runtime',
      icon: 'cpu',
      label: 'Batch threads',
      description:
        'Sets the thread count used for batching. Use 0 for auto; 256 is a safe UI ceiling for general-purpose systems.',
      inheritedLabel: 'Inherited by placements without a batch-thread override',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'threads_batch', value: '0', min: 0, max: 256, step: 1, unit: 'threads' }
    },
    {
      id: 'continuous-batching',
      categoryId: 'runtime',
      icon: 'layers',
      label: 'Continuous batching',
      description: 'Choose whether the runtime should keep batching continuously when supported.',
      inheritedLabel: 'Inherited by placements without a batching override',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'continuous_batching',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'numa',
      categoryId: 'runtime',
      icon: 'cpu',
      label: 'NUMA policy',
      description: 'Choose the NUMA policy used when launching the runtime.',
      inheritedLabel: 'Inherited by placements without a NUMA override',
      visibility: 'advanced',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'numa',
        value: 'auto',
        presentation: 'select',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'disabled', label: 'disabled' },
          { value: 'distribute', label: 'distribute' },
          { value: 'isolate', label: 'isolate' },
          { value: 'numactl', label: 'numactl' }
        ]
      }
    },
    {
      id: 'cpu-affinity',
      categoryId: 'runtime',
      icon: 'cpu',
      label: 'CPU affinity',
      description: 'Pin runtime threads to a specific CPU mask such as 0-3,8-11.',
      inheritedLabel: 'Inherited by placements without an affinity override',
      visibility: 'advanced',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'text', name: 'cpu_affinity', value: '', placeholder: 'e.g. 0-3,8-11' }
    },
    {
      id: 'priority',
      categoryId: 'runtime',
      icon: 'gauge',
      label: 'Process priority',
      description: 'Set the scheduler priority or nice value for the runtime process.',
      inheritedLabel: 'Inherited by placements without a priority override',
      visibility: 'advanced',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'text', name: 'priority', value: '', placeholder: 'e.g. 0 or normal' }
    },
    {
      id: 'poll',
      categoryId: 'runtime',
      icon: 'zap',
      label: 'Poll mode',
      description: 'Choose how the runtime polls for work when busy-waiting is available.',
      inheritedLabel: 'Inherited by placements without a poll override',
      visibility: 'advanced',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'poll',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'busy', label: 'busy' },
          { value: 'sleep', label: 'sleep' }
        ]
      }
    },
    {
      id: 'slot-prompt-similarity',
      categoryId: 'runtime',
      icon: 'gauge',
      label: 'Slot prompt similarity',
      description: 'Tune the similarity threshold used when comparing slot prompts before reuse.',
      inheritedLabel: 'Inherited by placements without a slot similarity override',
      visibility: 'advanced',
      tomlSection: THROUGHPUT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'range',
        name: 'slot_prompt_similarity',
        value: '0.50',
        min: 0,
        max: 1,
        step: 0.01
      }
    },
    {
      id: 'gpu-layers',
      categoryId: 'runtime',
      icon: 'layers',
      label: 'GPU layers',
      description: 'Set the GPU layer count, or use auto. The backend also accepts -1 to mean all layers.',
      inheritedLabel: 'Inherited by placements without a GPU layer override',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'text',
        name: 'gpu_layers',
        value: 'auto',
        placeholder: 'auto or integer layer count'
      }
    },
    {
      id: 'mmap',
      categoryId: 'runtime',
      icon: 'memory',
      label: 'Memory map',
      description: 'Choose whether model files are memory-mapped when loaded.',
      inheritedLabel: 'Inherited by placements without a memory-map override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'mmap',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'mlock',
      categoryId: 'runtime',
      icon: 'shield',
      label: 'Memory lock',
      description: 'Choose whether loaded model pages should be locked into RAM.',
      inheritedLabel: 'Inherited by placements without a memory-lock override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'mlock',
        value: 'off',
        presentation: 'toggle',
        options: [
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'warmup',
      categoryId: 'runtime',
      icon: 'zap',
      label: 'Warmup',
      description: 'Choose whether the runtime should perform a warmup pass after load.',
      inheritedLabel: 'Inherited by placements without a warmup override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'warmup',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'direct-io',
      categoryId: 'runtime',
      icon: 'folder',
      label: 'Direct I/O',
      description: 'Choose whether model files are opened with direct I/O when supported.',
      inheritedLabel: 'Inherited by placements without a direct I/O override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'direct_io',
        value: 'off',
        presentation: 'toggle',
        options: [
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'split-mode',
      categoryId: 'runtime',
      icon: 'layers',
      label: 'Split mode',
      description: 'Choose how layers are split across devices when model sharding is enabled.',
      inheritedLabel: 'Inherited by placements without a split-mode override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'split_mode',
        value: 'auto',
        presentation: 'select',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'none', label: 'none' },
          { value: 'layer', label: 'layer' },
          { value: 'row', label: 'row' }
        ]
      }
    },
    {
      id: 'main-gpu',
      categoryId: 'runtime',
      icon: 'server',
      label: 'Main GPU',
      description: 'Select the primary GPU index used for loading and dispatch.',
      inheritedLabel: 'Inherited by placements without a main-GPU override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'main_gpu', value: '0', min: 0, max: 7, step: 1, unit: 'GPU index' }
    },
    {
      id: 'tensor-split',
      categoryId: 'runtime',
      icon: 'layers',
      label: 'Tensor split',
      description: 'Set the tensor split ratios for multi-GPU placement, for example 0.5,0.5.',
      inheritedLabel: 'Inherited by placements without a tensor-split override',
      visibility: 'advanced',
      tomlSection: HARDWARE_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'text', name: 'tensor_split', value: '', placeholder: 'e.g. 0.5,0.5' }
    },
    {
      id: 'parallel-slots',
      categoryId: 'runtime',
      tomlSection: 'defaults.throughput',
      tomlKey: 'parallel',
      icon: 'cpu',
      label: 'Default slots / parallel requests',
      description:
        'Sets the default parallel slots for placements without their own value. More slots increase KV memory use.',
      inheritedLabel: 'Inherited by placements without a parallel override',
      control: { kind: 'range', name: 'parallel', value: '4', min: 1, max: 16, step: 1, unit: 'slots' }
    },
    {
      id: 'tuning-profile',
      categoryId: 'runtime',
      tomlSection: 'defaults.throughput',
      tomlKey: 'tuning_profile',
      icon: 'gauge',
      label: 'Default tuning profile',
      description: 'Choose the starting balance between throughput, batch size, and memory use.',
      inheritedLabel: 'Reset placements to default when experiments are finished',
      control: {
        kind: 'choice',
        name: 'tuning_profile',
        value: 'balanced',
        options: [
          { value: 'balanced', label: 'balanced' },
          { value: 'throughput', label: 'throughput' },
          { value: 'saver', label: 'saver' }
        ]
      }
    },
    {
      id: 'flash-attention',
      categoryId: 'runtime',
      tomlSection: 'defaults.model_fit',
      tomlKey: 'flash_attention',
      icon: 'layers',
      label: 'Flash attention policy',
      description: 'Choose the default attention kernel policy for compatible runtimes.',
      inheritedLabel: 'Inherited from Defaults unless a deployment pins kernels',
      control: {
        kind: 'choice',
        name: 'flash_attention',
        value: 'auto',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'enabled', label: 'enabled' },
          { value: 'disabled', label: 'disabled' }
        ]
      }
    },
    {
      id: 'llamacpp-flavor',
      categoryId: 'runtime',
      tomlSection: 'defaults.hardware',
      tomlKey: 'model_runtime',
      icon: 'binary',
      label: 'Model Runtime',
      description: 'Select the default runtime target for new placements.',
      inheritedLabel: 'Override when a model needs a specialized runner',
      control: {
        kind: 'choice',
        name: 'model_runtime',
        value: 'cuda',
        options: [
          { value: 'cuda', label: 'cuda' },
          { value: 'rocm', label: 'rocm' },
          { value: 'metal', label: 'metal' },
          { value: 'vulkan', label: 'vulkan' },
          { value: 'cpu', label: 'cpu' }
        ]
      }
    },
    {
      id: 'kv-cache',
      categoryId: 'memory',
      tomlSection: 'defaults.model_fit',
      tomlKey: 'kv_cache_policy',
      icon: 'filter',
      label: 'KV cache policy',
      description: 'Select how aggressively KV cache precision is reduced to fit larger contexts.',
      inheritedLabel: 'Used when the placement has no cache override',
      control: {
        kind: 'choice',
        name: 'kv_cache_policy',
        value: 'auto',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'quality', label: 'quality' },
          { value: 'balanced', label: 'balanced' },
          { value: 'saver', label: 'saver' }
        ]
      }
    },
    {
      id: 'memory-margin',
      categoryId: 'memory',
      tomlSection: 'defaults.hardware',
      tomlKey: 'safety_margin_gb',
      icon: 'memory',
      label: 'Memory / safety margin',
      description: 'Keep this much GPU memory free before placement fit checks pass.',
      inheritedLabel: 'Applied before per-model fit checks',
      control: { kind: 'range', name: 'safety_margin_gb', value: '2', min: 0, max: 8, step: 0.5, unit: 'GB' }
    },
    {
      id: 'ctx-size',
      categoryId: 'memory',
      icon: 'gauge',
      label: 'Context window size',
      description: 'Set the default context window size in tokens.',
      inheritedLabel: 'Applied when a placement does not override context size',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'ctx_size', value: '512', min: 512, max: 131072, step: 512, unit: 'tokens' }
    },
    {
      id: 'batch',
      categoryId: 'memory',
      icon: 'layers',
      label: 'Batch size',
      description: 'Set the default prefill batch size.',
      inheritedLabel: 'Applied when a placement does not override batch size',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'batch', value: '512', min: 32, max: 4096, step: 32, unit: 'tokens' }
    },
    {
      id: 'ubatch',
      categoryId: 'memory',
      icon: 'layers',
      label: 'Micro-batch size',
      description: 'Set the default decode micro-batch size.',
      inheritedLabel: 'Applied when a placement does not override micro-batch size',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'ubatch', value: '512', min: 32, max: 4096, step: 32, unit: 'tokens' }
    },
    {
      id: 'cache-type-k',
      categoryId: 'memory',
      icon: 'filter',
      label: 'KV cache type (K)',
      description: 'Choose the KV cache dtype used for keys.',
      inheritedLabel: 'Applied when a placement does not override key cache dtype',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'cache_type_k',
        value: 'f16',
        presentation: 'segmented',
        options: [
          { value: 'f16', label: 'f16' },
          { value: 'q8_0', label: 'q8_0' },
          { value: 'q4_0', label: 'q4_0' }
        ]
      }
    },
    {
      id: 'cache-type-v',
      categoryId: 'memory',
      icon: 'filter',
      label: 'KV cache type (V)',
      description: 'Choose the KV cache dtype used for values.',
      inheritedLabel: 'Applied when a placement does not override value cache dtype',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'cache_type_v',
        value: 'f16',
        presentation: 'segmented',
        options: [
          { value: 'f16', label: 'f16' },
          { value: 'q8_0', label: 'q8_0' },
          { value: 'q4_0', label: 'q4_0' }
        ]
      }
    },
    {
      id: 'kv-offload',
      categoryId: 'memory',
      icon: 'server',
      label: 'KV offload',
      description: 'Choose whether KV cache offloading stays enabled.',
      inheritedLabel: 'Applied when a placement does not override KV offload',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'kv_offload',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'kv-unified',
      categoryId: 'memory',
      icon: 'memory',
      label: 'Unified KV',
      description: 'Choose whether sequences share one unified KV buffer.',
      inheritedLabel: 'Applied when a placement does not override unified KV',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'kv_unified',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'cache-ram-mib',
      categoryId: 'memory',
      icon: 'memory',
      label: 'Prompt cache RAM',
      description: 'Set the maximum prompt cache size in MiB. Use 0 to disable the cache.',
      inheritedLabel: 'Applied when a placement does not override prompt cache RAM',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'cache_ram_mib', value: '8192', min: 0, max: 65536, step: 256, unit: 'MiB' }
    },
    {
      id: 'cache-idle-slots',
      categoryId: 'memory',
      icon: 'layers',
      label: 'Idle slot caching',
      description: 'Save and clear idle slots when a new task starts; requires unified KV and cache RAM.',
      inheritedLabel: 'Applied when a placement does not override idle slot caching',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'cache_idle_slots', value: '4', min: 0, max: 64, step: 1, unit: 'slots' }
    },
    {
      id: 'prompt-cache',
      categoryId: 'memory',
      icon: 'filter',
      label: 'Prompt cache',
      description: 'Choose whether prompt caching stays auto-managed or explicitly on or off.',
      inheritedLabel: 'Applied when a placement does not override prompt caching',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'prompt_cache',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'context-shift',
      categoryId: 'memory',
      icon: 'layers',
      label: 'Context shift',
      description: 'Allow context shifting for long-running generations when supported, or leave it on auto.',
      inheritedLabel: 'Applied when a placement does not override context shift',
      visibility: 'advanced',
      tomlSection: MODEL_FIT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'context_shift',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'speculation-mode',
      categoryId: 'speculative-decoding',
      tomlSection: 'defaults.speculative',
      tomlKey: 'mode',
      icon: 'brain',
      label: 'Default speculation mode',
      description: 'Choose the default speculation method, or leave the runtime in auto mode.',
      inheritedLabel: 'Inherited by compatible placements unless a model pins a mode',
      control: {
        kind: 'choice',
        name: 'mode',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'disabled', label: 'disabled' },
          { value: 'draft', label: 'draft' },
          { value: 'ngram', label: 'n-gram' }
        ]
      }
    },
    {
      id: 'draft-selection-policy',
      categoryId: 'speculative-decoding',
      tomlSection: 'defaults.speculative',
      tomlKey: 'draft_selection_policy',
      icon: 'filter',
      label: 'Default draft selection policy',
      description: 'Choose how draft models are selected when draft-model speculation is active.',
      inheritedLabel: 'Controls whether Mesh chooses a draft from catalog metadata',
      dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
      control: {
        kind: 'choice',
        name: 'draft_selection_policy',
        value: 'auto',
        presentation: 'toggle',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'manual', label: 'manual' }
        ]
      }
    },
    {
      id: 'incompatible-pairing-behavior',
      categoryId: 'speculative-decoding',
      tomlSection: 'defaults.speculative',
      tomlKey: 'pairing_fault',
      icon: 'shield',
      label: 'Incompatible pairing behavior',
      description: 'Choose what happens when the draft and target models cannot pair.',
      inheritedLabel: 'Determines launch behavior when draft and target models cannot pair',
      dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
      control: {
        kind: 'choice',
        name: 'pairing_fault',
        value: 'warn_disable',
        presentation: 'toggle',
        options: [
          { value: 'warn_disable', label: 'Warn & Disable' },
          { value: 'fail_closed', label: 'Fail launch' }
        ]
      }
    },
    {
      id: 'draft-max-tokens',
      categoryId: 'speculative-decoding',
      tomlSection: 'defaults.speculative',
      tomlKey: 'draft_max_tokens',
      icon: 'gauge',
      label: 'Default draft max tokens',
      description: 'Limit how many draft tokens can be proposed before verification.',
      inheritedLabel: 'Higher values can improve throughput when acceptance stays high',
      dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
      control: { kind: 'range', name: 'draft_max_tokens', value: '16', min: 1, max: 64, step: 1, unit: 'tokens' }
    },
    {
      id: 'draft-min-tokens',
      categoryId: 'speculative-decoding',
      tomlSection: 'defaults.speculative',
      tomlKey: 'draft_min_tokens',
      icon: 'gauge',
      label: 'Default draft minimum tokens',
      description: 'Set the smallest draft batch attempted before verification.',
      inheritedLabel: '0 lets the runtime verify as soon as the draft becomes uncertain',
      mutability: 'restart-required',
      dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
      control: { kind: 'range', name: 'draft_min_tokens', value: '0', min: 0, max: 32, step: 1, unit: 'tokens' }
    },
    {
      id: 'draft-acceptance-threshold',
      categoryId: 'speculative-decoding',
      tomlSection: 'defaults.speculative',
      tomlKey: 'draft_acceptance_threshold',
      icon: 'gauge',
      label: 'Default draft acceptance threshold',
      description: 'Set the confidence needed before draft tokens are accepted.',
      inheritedLabel: 'Lower values speculate more aggressively; higher values reject earlier',
      visibility: 'advanced',
      mutability: 'restart-required',
      dependsOn: DRAFT_MODEL_SPECULATION_DEPENDENCY,
      control: { kind: 'range', name: 'draft_acceptance_threshold', value: '0.70', min: 0, max: 1, step: 0.05 }
    },
    {
      id: 'temperature',
      categoryId: 'request-defaults',
      tomlSection: 'defaults.request_defaults',
      tomlKey: 'temperature',
      icon: 'gauge',
      label: 'Temperature',
      description: 'Fallback sampling temperature for requests that do not provide one.',
      inheritedLabel: 'Request payload temperature always wins when it is present',
      control: { kind: 'range', name: 'temperature', value: '0.70', min: 0, max: 2, step: 0.05 }
    },
    {
      id: 'top-p',
      categoryId: 'request-defaults',
      tomlSection: 'defaults.request_defaults',
      tomlKey: 'top_p',
      icon: 'gauge',
      label: 'Top-p',
      description: 'Fallback nucleus sampling threshold for requests that omit one.',
      inheritedLabel: 'Request payload top-p wins over this default',
      control: { kind: 'range', name: 'top_p', value: '0.95', min: 0, max: 1, step: 0.05 }
    },
    {
      id: 'reasoning-format',
      categoryId: 'request-defaults',
      tomlSection: 'defaults.request_defaults',
      tomlKey: 'reasoning_format',
      icon: 'cog',
      label: 'Reasoning format',
      description: 'Choose how thinking tokens appear in the response stream.',
      inheritedLabel: 'Inherited by model runtimes unless disabled per placement',
      control: {
        kind: 'choice',
        name: 'reasoning_format',
        value: 'auto',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'none', label: 'none' },
          { value: 'deepseek', label: 'deepseek' },
          { value: 'deepseek-legacy', label: 'deepseek-legacy' }
        ]
      }
    },
    {
      id: 'reasoning-budget',
      categoryId: 'request-defaults',
      tomlSection: 'defaults.request_defaults',
      tomlKey: 'reasoning_budget',
      icon: 'gauge',
      label: 'Reasoning budget',
      description: 'Cap the reasoning tokens reserved before the final answer.',
      inheritedLabel: 'Used only by runtimes with reasoning enabled',
      control: { kind: 'range', name: 'reasoning_budget', value: '0', min: 0, max: 4096, step: 128, unit: 'tok' }
    },
    {
      id: 'repeat-penalty',
      categoryId: 'request-defaults',
      tomlSection: 'defaults.request_defaults',
      tomlKey: 'repeat_penalty',
      icon: 'filter',
      label: 'Repeat penalty',
      description: 'Adjust how strongly repeated tokens are discouraged.',
      inheritedLabel: 'Safe fallback unless a placement tunes sampling',
      control: { kind: 'range', name: 'repeat_penalty', value: '1.1', min: 1, max: 2, step: 0.05 }
    },
    {
      id: 'repeat-last-n',
      categoryId: 'request-defaults',
      tomlSection: 'defaults.request_defaults',
      tomlKey: 'repeat_last_n',
      icon: 'layers',
      label: 'Repeat last-n window',
      description: 'Set how much recent token history the repeat penalty checks.',
      inheritedLabel: 'Inherited by placements with default sampling',
      control: { kind: 'range', name: 'repeat_last_n', value: '256', min: 0, max: 1024, step: 32, unit: 'tok' }
    },
    {
      id: 'top-k',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Top-k',
      description: 'Limit sampling to the top-k tokens.',
      inheritedLabel: 'Applied when a request does not override top-k',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: { kind: 'range', name: 'top_k', value: '40', min: 0, max: 100, step: 1 }
    },
    {
      id: 'min-p',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Min-p',
      description: 'Filter tokens below a dynamic probability floor.',
      inheritedLabel: 'Applied when a request does not override min-p',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: { kind: 'range', name: 'min_p', value: '0.05', min: 0, max: 1, step: 0.05 }
    },
    {
      id: 'presence-penalty',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Presence penalty',
      description: 'Increase or reduce the penalty for introducing new tokens.',
      inheritedLabel: 'Applied when a request does not override presence penalty',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: { kind: 'range', name: 'presence_penalty', value: '0', min: 0, max: 2, step: 0.1 }
    },
    {
      id: 'frequency-penalty',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Frequency penalty',
      description: 'Increase or reduce the penalty for repeated tokens.',
      inheritedLabel: 'Applied when a request does not override frequency penalty',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: { kind: 'range', name: 'frequency_penalty', value: '0', min: 0, max: 2, step: 0.1 }
    },
    {
      id: 'max-tokens',
      categoryId: 'request-defaults',
      icon: 'gauge',
      label: 'Max tokens',
      description: 'Cap the number of generated tokens for a request.',
      inheritedLabel: 'Applied when a request does not override the token cap',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: { kind: 'range', name: 'max_tokens', value: '0', min: 0, max: 32768, step: 256, unit: 'tokens' }
    },
    {
      id: 'seed',
      categoryId: 'request-defaults',
      icon: 'cog',
      label: 'Seed',
      description: 'Set the RNG seed for deterministic sampling when needed.',
      inheritedLabel: 'Applied when a request does not override the seed',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: { kind: 'text', name: 'seed', value: '-1', placeholder: '-1 (random)' }
    },
    {
      id: 'ignore-eos',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Ignore EOS',
      description: 'Choose whether the model should ignore end-of-sequence tokens.',
      inheritedLabel: 'Applied when a request does not override EOS handling',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: {
        kind: 'choice',
        name: 'ignore_eos',
        value: 'off',
        presentation: 'toggle',
        options: [
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'mirostat-mode',
      categoryId: 'request-defaults',
      icon: 'brain',
      label: 'Mirostat mode',
      description: 'Choose the Mirostat sampling mode, or disable it.',
      inheritedLabel: 'Applied when a request does not override Mirostat mode',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: {
        kind: 'choice',
        name: 'mirostat_mode',
        value: 'disabled',
        presentation: 'segmented',
        options: [
          { value: 'disabled', label: 'disabled' },
          { value: '1', label: '1' },
          { value: '2', label: '2' }
        ]
      }
    },
    {
      id: 'mirostat-entropy',
      categoryId: 'request-defaults',
      icon: 'gauge',
      label: 'Mirostat entropy',
      description: 'Set the Mirostat target entropy.',
      inheritedLabel: 'Applied when Mirostat mode is enabled',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      dependsOn: MIROSTAT_MODE_DEPENDENCY,
      control: { kind: 'range', name: 'mirostat_entropy', value: '5', min: 0.1, max: 10, step: 0.1 }
    },
    {
      id: 'mirostat-learning-rate',
      categoryId: 'request-defaults',
      icon: 'gauge',
      label: 'Mirostat learning rate',
      description: 'Set the Mirostat learning rate.',
      inheritedLabel: 'Applied when Mirostat mode is enabled',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      dependsOn: MIROSTAT_MODE_DEPENDENCY,
      control: { kind: 'range', name: 'mirostat_learning_rate', value: '0.1', min: 0.01, max: 1, step: 0.01 }
    },
    {
      id: 'samplers',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Samplers',
      description: 'Set the comma-separated sampler list.',
      inheritedLabel: 'Applied when a request does not override the sampler list',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: {
        kind: 'text',
        name: 'samplers',
        value: '',
        placeholder: 'top_k,tfs_z,typical_p,top_p,min_p,temperature'
      }
    },
    {
      id: 'sampler-sequence',
      categoryId: 'request-defaults',
      icon: 'layers',
      label: 'Sampler sequence',
      description: 'Set the sampler execution order.',
      inheritedLabel: 'Applied when a request does not override sampler ordering',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: {
        kind: 'text',
        name: 'sampler_sequence',
        value: '',
        placeholder: 'e.g. top_k;top_p;temperature'
      }
    },
    {
      id: 'stop',
      categoryId: 'request-defaults',
      icon: 'shield',
      label: 'Stop sequences',
      description: 'Set comma-separated stop sequences for a request.',
      inheritedLabel: 'Applied when a request does not override stop sequences',
      visibility: 'advanced',
      tomlSection: REQUEST_DEFAULTS_TOML_SECTION,
      mutability: 'runtime',
      control: {
        kind: 'text',
        name: 'stop',
        value: '',
        placeholder: 'comma-separated stop sequences'
      }
    },
    {
      id: 'activation-wire-dtype',
      categoryId: 'skippy-transport',
      icon: 'zap',
      label: 'Activation wire dtype',
      description: 'Choose the dtype used when activation frames travel between skippy stages.',
      inheritedLabel: 'Inherited by stage chains without a wire-dtype override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'activation_wire_dtype',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'f16', label: 'f16' },
          { value: 'f32', label: 'f32' },
          { value: 'q8', label: 'q8' }
        ]
      }
    },
    {
      id: 'stage-model-path',
      categoryId: 'skippy-transport',
      icon: 'folder',
      label: 'Stage model path',
      description: 'Set the model or package path used for this skippy stage.',
      inheritedLabel: 'Inherited by stage chains without an explicit model path',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      control: {
        kind: 'text',
        name: 'stage_model_path',
        value: '',
        placeholder: 'e.g. hf://meshllm/... or /path/to/stage.gguf'
      }
    },
    {
      id: 'stage-role',
      categoryId: 'skippy-transport',
      icon: 'server',
      label: 'Stage role',
      description: 'Choose the stage-chain role when topology is not inferred automatically.',
      inheritedLabel: 'Inherited by stage chains without an explicit role override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'stage_role',
        value: 'auto',
        presentation: 'select',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'prompt', label: 'prompt' },
          { value: 'stage', label: 'stage' }
        ]
      }
    },
    {
      id: 'stage-topology',
      categoryId: 'skippy-transport',
      icon: 'layers',
      label: 'Stage topology',
      description: 'Describe the stage chain topology when it is supplied as a text override.',
      inheritedLabel: 'Inherited by stage chains without a topology override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      control: {
        kind: 'text',
        name: 'stage_topology',
        value: '',
        placeholder: 'topology name or path'
      }
    },
    {
      id: 'prefill-chunking',
      categoryId: 'skippy-transport',
      icon: 'layers',
      label: 'Prefill chunking',
      description: 'Choose how prefill chunks are scheduled across a skippy stage chain.',
      inheritedLabel: 'Inherited by stage chains without a chunking override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'prefill_chunking',
        value: 'auto',
        presentation: 'select',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'fixed', label: 'fixed' },
          { value: 'schedule', label: 'schedule' },
          { value: 'adaptive-ramp', label: 'adaptive-ramp' }
        ]
      }
    },
    {
      id: 'prefill-chunk-size',
      categoryId: 'skippy-transport',
      icon: 'gauge',
      label: 'Prefill chunk size',
      description: 'Set the fixed prefill chunk size. Use 0 to keep the backend auto sentinel.',
      inheritedLabel: 'Inherited by fixed chunking when a stage does not override the size',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      mutability: 'restart-required',
      dependsOn: PREFILL_CHUNKING_FIXED_DEPENDENCY,
      control: { kind: 'range', name: 'prefill_chunk_size', value: '0', min: 0, max: 8192, step: 64, unit: 'tokens' }
    },
    {
      id: 'prefill-chunk-schedule',
      categoryId: 'skippy-transport',
      icon: 'layers',
      label: 'Prefill chunk schedule',
      description: 'Provide a comma-separated schedule for scheduled prefill chunking.',
      inheritedLabel: 'Inherited by scheduled chunking when a stage does not override the schedule',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      dependsOn: PREFILL_CHUNKING_SCHEDULE_DEPENDENCY,
      control: {
        kind: 'text',
        name: 'prefill_chunk_schedule',
        value: '',
        placeholder: 'e.g. 512,1024,2048'
      }
    },
    {
      id: 'binary-stage-transport',
      categoryId: 'skippy-transport',
      icon: 'binary',
      label: 'Binary stage transport',
      description: 'Choose whether the binary stage transport is enabled or left to auto selection.',
      inheritedLabel: 'Inherited by stage chains without a transport override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'binary_stage_transport',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'lifecycle-startup-timeout-ms',
      categoryId: 'skippy-transport',
      icon: 'gauge',
      label: 'Lifecycle startup timeout',
      description: 'Set how long the orchestrator waits for a stage to become ready during startup.',
      inheritedLabel: 'Inherited by stage chains without a startup timeout override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      control: {
        kind: 'range',
        name: 'lifecycle_startup_timeout_ms',
        value: '30000',
        min: 1,
        max: 600000,
        step: 1000,
        unit: 'ms'
      }
    },
    {
      id: 'lifecycle-readiness-interval-ms',
      categoryId: 'skippy-transport',
      icon: 'gauge',
      label: 'Lifecycle readiness interval',
      description: 'Set how often readiness is re-checked while startup is in flight.',
      inheritedLabel: 'Inherited by stage chains without a readiness polling override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      control: {
        kind: 'range',
        name: 'lifecycle_readiness_interval_ms',
        value: '1000',
        min: 100,
        max: 60000,
        step: 100,
        unit: 'ms'
      }
    },
    {
      id: 'lifecycle-health-interval-ms',
      categoryId: 'skippy-transport',
      icon: 'shield',
      label: 'Lifecycle health interval',
      description: 'Set how often background health checks run after a stage is up.',
      inheritedLabel: 'Inherited by stage chains without a health polling override',
      tomlSection: SKIPPY_TRANSPORT_TOML_SECTION,
      visibility: 'advanced',
      mutability: 'restart-required',
      control: {
        kind: 'range',
        name: 'lifecycle_health_interval_ms',
        value: '15000',
        min: 100,
        max: 60000,
        step: 100,
        unit: 'ms'
      }
    },
    {
      id: 'mmproj-offload',
      categoryId: 'multimodal',
      icon: 'image',
      label: 'MMProj offload',
      description: 'Choose whether the multimodal projector stays auto-managed or explicitly on or off.',
      inheritedLabel: 'Inherited by placements without a projector-offload override',
      tomlSection: MULTIMODAL_TOML_SECTION,
      mutability: 'restart-required',
      control: {
        kind: 'choice',
        name: 'mmproj_offload',
        value: 'auto',
        presentation: 'segmented',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'on', label: 'on' },
          { value: 'off', label: 'off' }
        ]
      }
    },
    {
      id: 'image-min-tokens',
      categoryId: 'multimodal',
      icon: 'image',
      label: 'Image minimum tokens',
      description: 'Set the minimum token budget reserved for each image input.',
      inheritedLabel: 'Inherited by placements without an image minimum override',
      tomlSection: MULTIMODAL_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'image_min_tokens', value: '0', min: 0, max: 2048, step: 32, unit: 'tokens' }
    },
    {
      id: 'image-max-tokens',
      categoryId: 'multimodal',
      icon: 'image',
      label: 'Image maximum tokens',
      description: 'Set the maximum token budget allowed for each image input.',
      inheritedLabel: 'Inherited by placements without an image maximum override',
      tomlSection: MULTIMODAL_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'range', name: 'image_max_tokens', value: '2048', min: 0, max: 4096, step: 32, unit: 'tokens' }
    },
    {
      id: 'mmproj',
      categoryId: 'multimodal',
      icon: 'image',
      label: 'MMProj path',
      description: 'Set an explicit local path to the multimodal projector file.',
      inheritedLabel: 'Inherited by placements without an explicit projector path',
      visibility: 'advanced',
      tomlSection: MULTIMODAL_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'text', name: 'mmproj', value: '', placeholder: 'e.g. /path/to/mmproj.gguf' }
    },
    {
      id: 'mmproj-url',
      categoryId: 'multimodal',
      icon: 'image',
      label: 'MMProj URL',
      description: 'Set a URL used to download or reference the multimodal projector file.',
      inheritedLabel: 'Inherited by placements without a projector URL override',
      visibility: 'advanced',
      tomlSection: MULTIMODAL_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'text', name: 'mmproj_url', value: '', placeholder: 'e.g. https://example.com/mmproj.gguf' }
    },
    {
      id: 'server-alias',
      categoryId: 'advanced-server',
      icon: 'server',
      label: 'Server alias',
      description: 'Set a human-friendly alias for the server in advanced deployments.',
      inheritedLabel: 'Inherited by deployments without an explicit server alias',
      visibility: 'advanced',
      tomlSection: ADVANCED_SERVER_TOML_SECTION,
      mutability: 'restart-required',
      control: { kind: 'text', name: 'alias', value: '', placeholder: 'model alias' }
    }
  ],
  preview: [
    { label: 'Scope', value: 'carrack only', meta: 'remote nodes are read-only context' },
    { label: 'Config path', value: '~/.mesh-llm/config.toml' },
    { label: 'Generated defaults', value: '73 settings', meta: 'deployment overrides win' },
    { label: 'Signing', value: 'Unsigned', meta: 'attestation pending' }
  ]
} as const

const NEWSLETTER_PROMPT = 'Can you draft three short intro paragraphs for a newsletter about local AI?'
const OUTBOUND_SECURITY: Decision[] = [
  { id: 'encrypted', ok: true, label: 'Encrypted in transit', detail: 'TLS 1.3 · mesh-pki' },
  { id: 'local', ok: true, label: 'Endpoint stays local', detail: '127.0.0.1:9337/v1/chat' },
  { id: 'hops', ok: true, label: 'No third-party hops', detail: 'request never leaves your mesh' },
  { id: 'hash', ok: true, label: 'Content hash', detail: '7c02...913a' }
]

const OUTBOUND_TRANSPARENCY_MESSAGE: TransparencyMessage = {
  kind: 'user',
  id: 'msg-u2',
  text: NEWSLETTER_PROMPT,
  at: '14:53',
  requestId: '7c02...913a',
  dispatch: { picked: 'lemony', candidates: 3, bytes: 184, tokens: 22, model: 'Qwen3.6-35B-A3B-UD' },
  route: ['desk', 'carrack', 'lemony'],
  security: OUTBOUND_SECURITY
}

const HELLO_TRANSPARENCY_MESSAGE: TransparencyMessage = {
  kind: 'assistant',
  id: 'msg-a0',
  text: 'Hello! How can I help you today?',
  at: '14:52',
  servedBy: 'carrack',
  route: ['desk', 'carrack'],
  model: 'Qwen3.6-27B-UD',
  receipt: 'rx-52a1',
  metrics: { rttMs: 1, ttftMs: 170, throughput: '36.8 tok/s', tokens: 10 },
  decisions: [
    { id: 'fit', ok: true, label: 'Qwen3.6-27B-UD warm', detail: 'carrack · 17.6 GB loaded' },
    { id: 'local', ok: true, label: 'Local node selected', detail: 'lowest latency for short prompt' },
    { id: 'link', ok: true, label: 'Link healthy', detail: 'local loopback · 0% loss' },
    { id: 'policy', ok: true, label: 'Short prompt stayed local', detail: 'policy: keep small replies on your node' }
  ],
  trace: [
    { id: 'queue', label: 'Queue', ms: 6, tone: 'neutral' },
    { id: 'route', label: 'Route', ms: 8, tone: 'neutral' },
    { id: 'prefill', label: 'Prefill', ms: 170, tone: 'good' },
    { id: 'decode', label: 'Decode', ms: 260, tone: 'good' }
  ]
}

const CAPACITY_PROMPT = 'Can you sketch a pooled placement plan for the coder stack before tomorrow?'
const CAPACITY_REPLY =
  'Use pooled placement on perseus.local for the small Qwen models, then keep Llama isolated on carrack GPU 1 so context-heavy drafts do not fragment the shared pool.'

const CAPACITY_OUTBOUND_MESSAGE: TransparencyMessage = {
  kind: 'user',
  id: 'msg-c2-u1',
  text: CAPACITY_PROMPT,
  at: 'Yesterday',
  requestId: 'c2a8...41ff',
  dispatch: { picked: 'carrack', candidates: 3, bytes: 152, tokens: 16, model: 'Qwen3.6-27B-UD' },
  route: ['desk', 'carrack'],
  security: OUTBOUND_SECURITY
}

const CAPACITY_REPLY_MESSAGE: TransparencyMessage = {
  kind: 'assistant',
  id: 'msg-c2-a1',
  text: CAPACITY_REPLY,
  at: 'Yesterday',
  servedBy: 'carrack',
  route: ['desk', 'carrack'],
  model: 'Qwen3.6-27B-UD',
  receipt: 'rx-c2a8',
  metrics: { rttMs: 1, ttftMs: 184, throughput: '31.2 tok/s', tokens: 64 },
  decisions: [
    { id: 'fit', ok: true, label: 'Qwen3.6-27B-UD warm', detail: 'carrack · 17.6 GB loaded' },
    {
      id: 'capacity',
      ok: true,
      label: 'Placement data available',
      detail: 'configuration plan references pooled VRAM'
    },
    { id: 'link', ok: true, label: 'Link healthy', detail: 'local loopback · 0% loss' },
    {
      id: 'policy',
      ok: true,
      label: 'Planning reply stayed local',
      detail: 'policy: keep capacity drafts on owner node'
    }
  ],
  trace: [
    { id: 'queue', label: 'Queue', ms: 8, tone: 'neutral' },
    { id: 'route', label: 'Route', ms: 12, tone: 'neutral' },
    { id: 'prefill', label: 'Prefill', ms: 184, tone: 'good' },
    { id: 'decode', label: 'Decode', ms: 1980, tone: 'good' }
  ]
}

export const CHAT_THREADS: Record<string, ThreadMessage[]> = {
  c1: [
    {
      id: 'msg-u1',
      messageRole: 'user',
      timestamp: '14:52',
      model: 'Qwen3.6-27B-UD',
      body: 'hello',
      routeNode: 'carrack',
      inspectMessage: {
        kind: 'user',
        id: 'msg-u1',
        text: 'hello',
        at: '14:52',
        requestId: '5a18...9fd0',
        dispatch: { picked: 'carrack', candidates: 3, bytes: 6, tokens: 1, model: 'Qwen3.6-27B-UD' },
        route: ['desk', 'carrack'],
        security: OUTBOUND_SECURITY
      }
    },
    {
      id: 'msg-a0',
      messageRole: 'assistant',
      timestamp: '14:52',
      model: 'Qwen3.6-27B-UD',
      body: 'Hello! How can I help you today?',
      route: 'carrack',
      routeNode: 'carrack',
      tokens: '10 tok',
      tokPerSec: '36.8 tok/s',
      ttft: '170 ms',
      inspectMessage: HELLO_TRANSPARENCY_MESSAGE,
      inspectLabel: 'Inspect transparency'
    },
    {
      id: 'msg-u2',
      messageRole: 'user',
      timestamp: '14:53',
      model: 'Qwen3.6-35B-A3B-UD',
      body: NEWSLETTER_PROMPT,
      routeNode: 'lemony-28',
      inspectMessage: OUTBOUND_TRANSPARENCY_MESSAGE,
      inspectLabel: 'Inspect outbound route'
    },
    {
      id: 'msg-a1',
      messageRole: 'assistant',
      timestamp: '14:53',
      model: 'Qwen3.6-35B-A3B-UD',
      body: 'Here are three revisions with different tones — playful, serious, and technical. Want me to expand any of them?',
      route: 'lemony-28',
      routeNode: 'lemony-28',
      tokens: '148 tok',
      tokPerSec: '22.4 tok/s',
      ttft: '312 ms',
      inspectMessage: TRANSPARENCY_MESSAGE
    }
  ],
  c2: [
    {
      id: 'msg-c2-u1',
      messageRole: 'user',
      timestamp: 'Yesterday',
      model: 'Qwen3.6-27B-UD',
      body: CAPACITY_PROMPT,
      routeNode: 'carrack',
      inspectMessage: CAPACITY_OUTBOUND_MESSAGE,
      inspectLabel: 'Inspect capacity prompt route'
    },
    {
      id: 'msg-c2-a1',
      messageRole: 'assistant',
      timestamp: 'Yesterday',
      model: 'Qwen3.6-27B-UD',
      body: CAPACITY_REPLY,
      route: 'carrack',
      routeNode: 'carrack',
      tokens: '64 tok',
      tokPerSec: '31.2 tok/s',
      ttft: '184 ms',
      inspectMessage: CAPACITY_REPLY_MESSAGE,
      inspectLabel: 'Inspect capacity reply route'
    }
  ]
}

export const DASHBOARD_HARNESS: DashboardHarnessData = {
  hero: {
    title: 'Your private mesh',
    description:
      'Build personal AI from open models. Pool machines across your home, office, or friends — no cloud needed.',
    actions: [
      { label: 'Learn more', href: 'https://docs.meshllm.cloud/', tone: 'link' },
      { label: 'GitHub', href: 'https://github.com/Mesh-LLM/mesh-llm', tone: 'secondary' }
    ]
  },
  statusMetrics: STATUS_METRICS,
  peers: PEERS,
  peerSummary: PEER_SUMMARY,
  models: MODELS,
  meshNodeSeeds: MESH_NODES,
  meshId: 'dashboard-mesh',
  connect: {
    installHref: 'https://docs.meshllm.cloud/#install',
    apiStatus: 'configured target',
    runCommand: 'mesh-llm --auto --join <mesh-invite-token>',
    description: 'contribute compute to the mesh'
  }
}

export const SHELL_HARNESS: ShellHarnessData = {
  productName: 'mesh-llm',
  brand: { primary: 'mesh', accent: 'llm' },
  footerLinks: [{ label: 'Docs', href: 'https://docs.meshllm.cloud/' }],
  footerTrailingLink: { label: 'GitHub', href: 'https://github.com/Mesh-LLM/mesh-llm' },
  topNavApiAccessLinks: [
    { href: 'https://docs.meshllm.cloud/', label: 'Docs' },
    { href: 'https://docs.meshllm.cloud/#install', label: 'Install' }
  ],
  topNavJoinCommands: [
    {
      label: 'Invite token',
      value: '<mesh-invite-token>',
      hint: 'Paste your issued token into any join command below.',
      noWrapValue: true
    },
    {
      label: 'Auto join and serve command',
      value: 'mesh-llm --auto --join <mesh-invite-token>',
      prefix: '$',
      hint: 'Matches the Connect panel flow: join, select a model, and serve the API.'
    },
    { label: 'Client-only join command', value: 'mesh-llm client --join <mesh-invite-token>', prefix: '$' },
    { label: 'Blackboard skill command', value: 'mesh-llm blackboard install-skill', prefix: '$' }
  ],
  topNavJoinLinks: [
    { href: 'https://docs.meshllm.cloud/', label: 'Setup' },
    { href: 'https://docs.meshllm.cloud/#install', label: 'Install' },
    { href: 'https://docs.meshllm.cloud/#blackboard', label: 'Blackboard' }
  ]
}

export const CHAT_HARNESS: ChatHarnessData = {
  title: 'Chat',
  conversations: CONVERSATIONS,
  conversationGroups: [
    { title: 'Today', conversationIds: ['c1', 'c2'] },
    { title: 'Earlier', conversationIds: [] }
  ],
  transparencyNodes: TRANSPARENCY_NODES,
  threads: CHAT_THREADS,
  models: MODELS,
  actionMetrics: [
    { id: 'nodes', icon: 'cpu', label: '1 node' },
    { id: 'vram', icon: 'hard-drive', label: '61.7 GB' }
  ],
  modelLabel: 'Model'
}

export const CONFIGURATION_HARNESS: ConfigurationHarnessData = {
  title: 'Configuration',
  description:
    "Drag models from the catalog onto a node's VRAM container. Pooled nodes combine all devices into one bar.",
  nodes: CFG_NODES,
  assigns: INITIAL_ASSIGNS,
  catalog: CFG_CATALOG,
  preferredAssignId: 'a2',
  defaults: CONFIGURATION_DEFAULTS,
  configFilePath: '~/.mesh-llm/config.toml',
  validationWarnings: [
    { kind: 'ok', text: 'All pinned models have valid gpu_id targets.' },
    {
      kind: 'warn',
      text: 'carrack · GPU 0 · GLM-4.7-Flash will exceed 80% VRAM at 16K context. Consider 8K or moving to GPU 1.'
    },
    { kind: 'ok', text: 'Plugin endpoint http://localhost:8000/v1 is reachable.' },
    { kind: 'info', text: 'Flash attention is on by default, no per-model override emitted.' }
  ] satisfies TomlValidationWarning[],
  launchSummaryConfig: {
    httpBind: '0.0.0.0:9337',
    mmap: 'off'
  }
}
