import type { StatusPayload, MeshModelRaw, PeerInfo } from '@/lib/api/types'
import type {
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationHarnessData,
  ConfigNode,
  ConfigModel
} from '@/features/app-tabs/types'
import { CONFIGURATION_DEFAULTS, CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import { getSettingBaselineValue, isSettingDisabled } from '@/features/configuration/lib/settings-utils'
import { ApiError, parseApiErrorBody } from '@/lib/api/errors'
import { env } from '@/lib/env'

export type RuntimeControlBootstrapPayload = {
  enabled: boolean
  local_only: boolean
  requires_explicit_remote_endpoint: boolean
  endpoint?: string
  disabled_reason?: string
  message?: string
  suggested_commands?: string[]
}

export type RuntimeControlDefaultsConfig = Record<string, unknown>

export type RuntimeControlMeshConfig = {
  defaults?: RuntimeControlDefaultsConfig
  [key: string]: unknown
}

export type RuntimeControlConfigSnapshot = {
  revision: number
  config: RuntimeControlMeshConfig
  [key: string]: unknown
}

type RuntimeControlConfigResponse = {
  snapshot: RuntimeControlConfigSnapshot
}

export type RuntimeControlConfigResult = {
  bootstrap: RuntimeControlBootstrapPayload
  snapshot?: RuntimeControlConfigSnapshot
}

export type RuntimeControlApplyResponse = {
  success: boolean
  current_revision: number
  config_hash: string
  apply_mode: string
  error?: unknown
}

export function runtimeControlApplyErrorMessage(response: RuntimeControlApplyResponse | null | undefined) {
  if (!response) return undefined
  return runtimeControlErrorMessage(response.error)
}

function runtimeControlErrorMessage(error: unknown): string | undefined {
  if (typeof error === 'string') return error.trim() || undefined
  if (!error || typeof error !== 'object') return undefined

  const details = error as Record<string, unknown>
  if (typeof details['message'] === 'string') return details['message'].trim() || undefined
  if (typeof details['code'] === 'string') return details['code'].replace(/_/g, ' ')
  return undefined
}

const LEGACY_CONTROL_DEFAULTS_SECTION_PATHS: Record<string, readonly string[]> = {
  'parallel-slots': ['model_fit', 'parallel'],
  'tuning-profile': ['throughput', 'tuning_profile'],
  'flash-attention': ['model_fit', 'flash_attention'],
  'llamacpp-flavor': ['hardware', 'model_runtime'],
  'kv-cache': ['model_fit', 'kv_cache_policy'],
  'memory-margin': ['model_fit', 'safety_margin_gb'],
  'speculation-mode': ['speculative', 'mode'],
  'draft-selection-policy': ['speculative', 'draft_selection_policy'],
  'incompatible-pairing-behavior': ['speculative', 'pairing_fault'],
  'draft-max-tokens': ['speculative', 'draft_max_tokens'],
  'draft-min-tokens': ['speculative', 'draft_min_tokens'],
  'draft-acceptance-threshold': ['speculative', 'draft_acceptance_threshold'],
  'reasoning-format': ['request_defaults', 'reasoning_format'],
  'reasoning-budget': ['request_defaults', 'reasoning_budget'],
  'repeat-penalty': ['request_defaults', 'repeat_penalty'],
  'repeat-last-n': ['request_defaults', 'repeat_last_n']
}

const CONTROL_DEFAULTS_KEY_OVERRIDES: Record<string, string> = {
  'server-alias': 'alias'
}

function mapNodeState(state: string | undefined): 'online' | 'degraded' | 'offline' {
  if (state === 'client') return 'offline'
  return 'online'
}

function resolvePeerId(peer: PeerInfo, fallbackIndex: number): string {
  return peer.node_id ?? peer.id ?? peer.hostname ?? `peer-${fallbackIndex}`
}

function adaptPeerToConfigNode(peer: PeerInfo, fallbackIndex: number): ConfigNode {
  const id = resolvePeerId(peer, fallbackIndex)

  return {
    id,
    hostname: peer.hostname ?? id,
    region: peer.region ?? 'unknown',
    status: mapNodeState(peer.node_state ?? peer.state ?? peer.role?.toLowerCase()),
    cpu: peer.hardware_label ?? 'Unknown CPU',
    ramGB: 0,
    gpus: [],
    placement: 'separate'
  }
}

function adaptModelToConfigModel(model: MeshModelRaw): ConfigModel {
  return {
    id: model.name,
    name: model.name,
    family: model.family ?? model.name.split('/')[0] ?? 'unknown',
    paramsB: model.params_b ?? 0,
    quant: model.quantization ?? 'unknown',
    sizeGB: model.size_gb ?? 0,
    diskGB: model.disk_gb ?? 0,
    ctxMaxK: model.context_length == null ? 0 : Math.round(model.context_length / 1000),
    moe: model.capabilities?.moe ?? model.moe ?? false,
    vision: model.capabilities?.vision ?? model.vision ?? model.tags?.includes('vision') ?? false,
    tags: model.tags ?? []
  }
}

function cloneControlValue(
  control: ConfigurationDefaultsSetting['control'],
  value: string
): ConfigurationDefaultsSetting['control'] {
  return { ...control, value }
}

function parseDefaultsSectionPath(section: string | undefined): string[] {
  if (!section?.startsWith('defaults.')) return []
  return section.slice('defaults.'.length).split('.').filter(Boolean)
}

function resolveControlDefaultsPath(setting: ConfigurationDefaultsSetting): string[] {
  const legacyPath = LEGACY_CONTROL_DEFAULTS_SECTION_PATHS[setting.id]
  if (legacyPath) return [...legacyPath]

  const key =
    CONTROL_DEFAULTS_KEY_OVERRIDES[setting.id] ?? ('name' in setting.control ? setting.control.name : setting.id)
  const categorySection = CONFIGURATION_DEFAULTS.categories.find((category) => category.id === setting.categoryId) as
    | { tomlSection?: string }
    | undefined
  const sectionPath = parseDefaultsSectionPath(setting.tomlSection ?? categorySection?.tomlSection)

  return [...sectionPath, key]
}

function overlayDefaultsValues(
  harnessDefaults: ConfigurationDefaultsHarnessData,
  defaultsValues: ConfigurationDefaultsValues
): ConfigurationDefaultsHarnessData {
  return {
    ...harnessDefaults,
    settings: harnessDefaults.settings.map((setting) => ({
      ...setting,
      control: cloneControlValue(setting.control, defaultsValues[setting.id] ?? setting.control.value)
    }))
  }
}

function readPath(source: unknown, path: readonly string[]): unknown {
  let current = source
  for (const segment of path) {
    if (!current || typeof current !== 'object' || !(segment in current)) return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return current
}

function writePath(target: Record<string, unknown>, path: readonly string[], value: unknown) {
  let current = target
  path.forEach((segment, index) => {
    const isLeaf = index === path.length - 1
    if (isLeaf) {
      current[segment] = value
      return
    }

    const next = current[segment]
    if (!next || typeof next !== 'object' || Array.isArray(next)) {
      current[segment] = {}
    }
    current = current[segment] as Record<string, unknown>
  })
}

function deletePath(target: Record<string, unknown>, path: readonly string[]): boolean {
  const [segment, ...rest] = path
  if (!segment || !(segment in target)) return Object.keys(target).length === 0

  if (rest.length === 0) {
    delete target[segment]
    return Object.keys(target).length === 0
  }

  const next = target[segment]
  if (!next || typeof next !== 'object' || Array.isArray(next)) return Object.keys(target).length === 0

  if (deletePath(next as Record<string, unknown>, rest)) delete target[segment]
  return Object.keys(target).length === 0
}

function serializeDefaultSettingValue(setting: ConfigurationDefaultsSetting, value: unknown): string | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean' && setting.control.kind === 'choice') {
    const optionValues = new Set(setting.control.options.map((option) => option.value))
    if (optionValues.has('on') && optionValues.has('off')) {
      return value ? 'on' : 'off'
    }
  }
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (setting.control.kind === 'choice') return String(value)
  return undefined
}

function parseDefaultSettingValue(setting: ConfigurationDefaultsSetting, value: string): unknown {
  if (setting.control.kind === 'choice') {
    const optionValues = new Set(setting.control.options.map((option) => option.value))
    if (optionValues.has('on') && optionValues.has('off') && !optionValues.has('auto')) {
      return value === 'on'
    }
  }
  if (setting.control.kind !== 'range') return value
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : value
}

function cloneMeshConfig(config: RuntimeControlMeshConfig): RuntimeControlMeshConfig {
  return JSON.parse(JSON.stringify(config)) as RuntimeControlMeshConfig
}

export function createConfigurationDefaultsValuesFromMeshConfig(
  config: RuntimeControlMeshConfig
): ConfigurationDefaultsValues {
  const values: ConfigurationDefaultsValues = {}
  const defaults = config.defaults

  for (const setting of CONFIGURATION_DEFAULTS.settings) {
    if (!defaults) continue

    const path = resolveControlDefaultsPath(setting)
    const value = readPath(defaults, path)
    const serialized = serializeDefaultSettingValue(setting, value)
    if (serialized !== undefined) values[setting.id] = serialized
  }

  return values
}

export function mergeConfigurationDefaultsIntoMeshConfig(
  config: RuntimeControlMeshConfig,
  defaultsValues: ConfigurationDefaultsValues
): RuntimeControlMeshConfig {
  const nextConfig = cloneMeshConfig(config)
  const defaults =
    nextConfig.defaults && typeof nextConfig.defaults === 'object' && !Array.isArray(nextConfig.defaults)
      ? { ...nextConfig.defaults }
      : {}

  for (const setting of CONFIGURATION_DEFAULTS.settings) {
    const path = resolveControlDefaultsPath(setting)
    deletePath(defaults, path)

    const nextValue = defaultsValues[setting.id]
    if (nextValue == null) continue
    if (isSettingDisabled(setting, CONFIGURATION_DEFAULTS.settings, defaultsValues)) continue
    if (nextValue === getSettingBaselineValue(setting)) continue
    if (setting.control.kind === 'text' && nextValue.trim().length === 0) continue

    writePath(defaults, path, parseDefaultSettingValue(setting, nextValue))
  }

  if (Object.keys(defaults).length === 0) {
    delete nextConfig.defaults
  } else {
    nextConfig.defaults = defaults
  }
  return nextConfig
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await parseApiErrorBody(response)
    throw new ApiError(response.status, message, message)
  }

  return response.json() as Promise<T>
}

export async function fetchRuntimeControlBootstrap(): Promise<RuntimeControlBootstrapPayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control-bootstrap`)
  return expectJson<RuntimeControlBootstrapPayload>(response)
}

export async function fetchRuntimeControlConfigSnapshot(endpoint: string): Promise<RuntimeControlConfigSnapshot> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/get-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint })
  })
  const payload = await expectJson<RuntimeControlConfigResponse>(response)
  return payload.snapshot
}

export async function fetchRuntimeControlConfig(): Promise<RuntimeControlConfigResult> {
  const bootstrap = await fetchRuntimeControlBootstrap()
  const endpoint = bootstrap.endpoint?.trim()

  if (!bootstrap.enabled || !endpoint) return { bootstrap }

  const snapshot = await fetchRuntimeControlConfigSnapshot(endpoint)
  return { bootstrap: { ...bootstrap, endpoint }, snapshot }
}

export async function applyRuntimeControlConfig(
  endpoint: string,
  snapshot: RuntimeControlConfigSnapshot,
  defaultsValues: ConfigurationDefaultsValues
): Promise<{ response: RuntimeControlApplyResponse; snapshot: RuntimeControlConfigSnapshot }> {
  const config = mergeConfigurationDefaultsIntoMeshConfig(snapshot.config, defaultsValues)
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/apply-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      endpoint,
      expected_revision: snapshot.revision,
      config
    })
  })
  const payload = await expectJson<RuntimeControlApplyResponse>(response)

  return {
    response: payload,
    snapshot: {
      ...snapshot,
      revision: payload.current_revision,
      config
    }
  }
}

export function adaptStatusToConfiguration(
  payload: StatusPayload,
  models: MeshModelRaw[],
  defaultsValues?: ConfigurationDefaultsValues
): ConfigurationHarnessData {
  const nodes: ConfigNode[] = payload.peers.map(adaptPeerToConfigNode)
  const catalog: ConfigModel[] = models.map(adaptModelToConfigModel)
  const defaults = defaultsValues
    ? overlayDefaultsValues(CONFIGURATION_HARNESS.defaults, defaultsValues)
    : CONFIGURATION_HARNESS.defaults

  return {
    ...CONFIGURATION_HARNESS,
    nodes,
    catalog,
    defaults,
    assigns: [],
    preferredAssignId: undefined
  }
}
