import type { ReactNode } from 'react'
import type { ChatMessage } from '@/features/chat/lib/chat-types'

export type ResolvedTheme = 'light' | 'dark'
export type Theme = 'auto' | ResolvedTheme
export type Accent = 'blue' | 'cyan' | 'violet' | 'green' | 'amber' | 'pink'
export type Density = 'compact' | 'normal' | 'sparse'
export type PanelStyle = 'solid' | 'soft'
export type AppTab = 'network' | 'chat' | 'configuration'

export type StatusBadgeTone = 'good' | 'warn' | 'bad' | 'muted' | 'accent'
export type StatusMetric = {
  id: string
  icon?: ReactNode
  label: string
  value: string | number
  unit?: string
  meta?: string
  sparkline?: number[]
  badge?: { label: string; tone?: StatusBadgeTone }
  variant?: 'metric' | 'identity'
  mono?: boolean
}
export type LinkItem = { label: string; href: string }
export type HeroAction = LinkItem & { tone?: 'primary' | 'secondary' | 'link' }
export type Peer = {
  id: string
  hostname: string
  region: string
  status: 'online' | 'degraded' | 'offline'
  hostedModels: string[]
  sharePct: number
  latencyMs: number
  loadPct: number
  shortId?: string
  role?: 'you' | 'host' | 'peer' | 'client' | 'worker'
  nodeState?: 'client' | 'standby' | 'loading' | 'serving'
  version?: string
  vramGB?: number
  toksPerSec?: number
  hardwareLabel?: string
  ownership?: string
  owner?: string
}
export type PeerSummary = { total: number; online: number; capacity: string }

export type PeerHostedModel = {
  name: string
  paramsB?: number
  sizeGB?: number
}

export type PeerDTO = Omit<Peer, 'hostedModels'> & {
  hostedModels: PeerHostedModel[]
}
export type ModelFamilyColorKey =
  | 'family-0'
  | 'family-1'
  | 'family-2'
  | 'family-3'
  | 'family-4'
  | 'family-5'
  | 'family-6'
  | 'family-7'
export type ModelCapabilities = Partial<Record<string, boolean>>
export type ModelSummary = {
  name: string
  family: string
  familyColor?: ModelFamilyColorKey
  size: string
  context: string
  status: 'ready' | 'warming' | 'offline' | 'warm'
  tags: string[]
  nodeCount?: number
  fullId?: string
  paramsB?: number
  paramsLabel?: string
  quant?: string
  sizeGB?: number
  diskGB?: number
  ctxMaxK?: number
  ctxPerGB?: number
  moe?: boolean
  vision?: boolean
  capabilities?: ModelCapabilities
  license?: string
  activitySummary?: string
}
export type MeshNodeRenderKind = 'client' | 'worker' | 'active' | 'serving' | 'self'
export type MeshNodeState = 'serving' | 'loading' | 'standby' | 'client'
export type MeshNode = {
  id: string
  peerId?: string
  label: string
  x: number
  y: number
  status: 'online' | 'degraded' | 'offline'
  role?: 'self' | 'peer'
  subLabel?: string
  renderKind?: MeshNodeRenderKind
  meshState?: MeshNodeState
  host?: boolean
  client?: boolean
  servingModels?: string[]
  latencyMs?: number | null
  hostname?: string
  vramGB?: number
}
export type ModelSelectStatus = { label: string; tone?: StatusBadgeTone }
export type ModelSelectOption = { value: string; label: string; meta?: string; status?: ModelSelectStatus }
export type MessageRole = 'user' | 'assistant'
export type Conversation = {
  id: string
  title: string
  subtitle?: string
  updatedAt: string
  createdAt?: number
  messages?: ChatMessage[]
  active?: boolean
}
export type TransparencyNode = {
  id: string
  label: string
  region?: string
  status?: 'online' | 'degraded' | 'offline'
  isLocal?: boolean
}
export type TomlValidationWarning = { kind: 'ok' | 'warn' | 'info'; text: string }
export type TopNavJoinCommand = {
  label: string
  value: string
  prefix?: string
  hint?: string
  noWrapValue?: boolean
  copyValue?: string
  disabled?: boolean
}
export type Decision = { id: string; ok: boolean; label: string; detail?: string }
export type TraceSegment = { id: string; label: string; ms: number; tone?: 'neutral' | 'good' | 'warn' | 'bad' }
export type InboundTransparencyMessage = {
  kind: 'assistant'
  id: string
  text: string
  at: string
  servedBy: string
  route: string[]
  model: string
  receipt: string
  metrics: { rttMs: number; ttftMs: number; throughput: string; tokens: number }
  decisions: Decision[]
  trace: TraceSegment[]
}
export type OutboundTransparencyMessage = {
  kind: 'user'
  id: string
  text: string
  at: string
  requestId: string
  dispatch: { picked: string; candidates: number; bytes: number; tokens: number; model: string }
  security: Decision[]
  route: string[]
}
export type TransparencyMessage = InboundTransparencyMessage | OutboundTransparencyMessage
export type ThreadMessage = {
  id: string
  messageRole: MessageRole
  timestamp: string
  model?: string
  body: string
  route?: string
  routeNode?: string
  tokens?: string
  tokPerSec?: string
  ttft?: string
  inspectMessage?: TransparencyMessage
  inspectLabel?: string
}
export type DashboardConnectData = { installHref: string; apiStatus: string; runCommand: string; description: string }
export type ShellHarnessData = {
  productName: string
  brand: { primary: string; accent: string }
  footerLinks: LinkItem[]
  footerTrailingLink: LinkItem
  topNavApiAccessLinks: LinkItem[]
  topNavJoinCommands: TopNavJoinCommand[]
  topNavJoinLinks: LinkItem[]
}
export type DashboardHarnessData = {
  hero: { title: string; description: string; actions: HeroAction[] }
  statusMetrics: StatusMetric[]
  peers: Peer[]
  peerSummary: PeerSummary
  models: ModelSummary[]
  meshNodeSeeds: MeshNode[]
  meshId: string
  connect: DashboardConnectData
}
export type ChatActionMetric = { id: string; icon: 'cpu' | 'hard-drive'; label: string }
export type ConversationGroup = { title: string; conversationIds: string[] }
export type ChatHarnessData = {
  title: string
  conversations: Conversation[]
  conversationGroups: ConversationGroup[]
  transparencyNodes: TransparencyNode[]
  threads: Record<string, ThreadMessage[]>
  models: ModelSummary[]
  actionMetrics: ChatActionMetric[]
  modelLabel: string
}
export type ConfigGpu = { idx: number; name: string; totalGB: number; reservedGB?: number }
export type Placement = 'separate' | 'pooled'
export type ConfigNode = {
  id: string
  hostname: string
  region: string
  status: 'online' | 'degraded' | 'offline'
  cpu: string
  ramGB: number
  gpus: ConfigGpu[]
  placement: Placement
  memoryTopology?: 'discrete' | 'unified'
}
export type ConfigModel = {
  id: string
  name: string
  family: string
  familyColor?: ModelFamilyColorKey
  paramsB: number
  paramsLabel?: string
  quant: string
  sizeGB: number
  diskGB: number
  ctxMaxK: number
  ctxPerGB?: number
  layers?: number
  heads?: number
  embed?: number
  tokenizer?: string
  moe: boolean
  vision: boolean
  tags: string[]
}
export type ConfigAssign = { id: string; modelId: string; nodeId: string; containerIdx: number; ctx: number }
export type ConfigurationDefaultsCategoryId = 'runtime' | 'memory' | 'speculative-decoding' | 'advanced'
export type ConfigurationDefaultsCategory = {
  id: ConfigurationDefaultsCategoryId
  label: string
  summary: string
  help: string
}
export type ConfigurationDefaultsChoice = { value: string; label: string; description?: string }
export type ConfigurationDefaultsControl =
  | {
      kind: 'choice'
      name: string
      options: readonly ConfigurationDefaultsChoice[]
      value: string
      presentation?: 'segmented' | 'select' | 'toggle'
    }
  | { kind: 'text'; name: string; value: string; placeholder?: string }
  | { kind: 'range'; name: string; value: string; min: number; max: number; step: number; unit?: string }
  | { kind: 'metric'; value: string; unit?: string }
export type ConfigurationDefaultsSettingIcon =
  | 'brain'
  | 'cpu'
  | 'layers'
  | 'memory'
  | 'binary'
  | 'folder'
  | 'gauge'
  | 'shield'
  | 'cog'
  | 'filter'
export type ConfigurationDefaultsSetting = {
  id: string
  categoryId: ConfigurationDefaultsCategoryId
  icon: ConfigurationDefaultsSettingIcon
  label: string
  description: string
  inheritedLabel: string
  control: ConfigurationDefaultsControl
}
export type ConfigurationDefaultsPreviewItem = { label: string; value: string; meta?: string }
export type ConfigurationDefaultsValues = Record<string, string>
export type ConfigurationDefaultsHarnessData = {
  categories: readonly ConfigurationDefaultsCategory[]
  settings: readonly ConfigurationDefaultsSetting[]
  preview: readonly ConfigurationDefaultsPreviewItem[]
}
export type ConfigurationHarnessData = {
  title: string
  description: string
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  catalog: ConfigModel[]
  preferredAssignId?: string
  defaults: ConfigurationDefaultsHarnessData
  configFilePath?: string
  validationWarnings?: TomlValidationWarning[]
  launchSummaryConfig?: { httpBind?: string; mmap?: string }
}
