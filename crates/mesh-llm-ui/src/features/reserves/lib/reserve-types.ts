export type ReserveNodeState = 'standby' | 'waking' | 'joining' | 'online' | 'failed' | 'unreachable'

export type ReserveNode = {
  id: string
  hw: string
  vram: number
  models: string[]
  state: ReserveNodeState
  location?: string
  note?: string
  eta?: number
  progress?: number
  error?: string
  failedAt?: string
  retryable?: boolean
  lastSeen?: string
  since?: string
}

export type ReserveProvider = {
  id: string
  name: string
  kind: string
  icon?: 'server' | 'cloud' | 'lan'
  region: string
  tags?: string[]
  billing: string
  summary?: string
  nodes: ReserveNode[]
}

export type MeshProviderId = 'bare-metal' | 'digitalocean' | 'gcp' | 'aws'

export type MeshProviderOptionField = {
  id: 'name' | 'region'
  label: string
  placeholder: string
  helper: string
}

export type MeshProvider = {
  id: MeshProviderId
  name: string
  kind: string
  icon: 'server' | 'cloud'
  description: string
  availability: 'supported' | 'coming-soon'
  disabledReason?: string
  defaultName: string
  defaultRegion: string
  billing: string
  summary: string
  optionFields: MeshProviderOptionField[]
}

export type ReserveWakePolicySettings = {
  autoWakeEnabled: boolean
  thresholdPercent: number
  sustainedSeconds: number
  providerOrder: string[]
  idleMinutes: number
}

export type ReservePolicyTileData = {
  title: string
  value: string
  status?: string
  explanation: string
}
