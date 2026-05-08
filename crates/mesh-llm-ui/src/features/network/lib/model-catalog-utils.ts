import type { ModelSummary } from '@/features/app-tabs/types'

export type ModelFilterKey = 'provider' | 'license' | 'variant' | 'status' | 'family' | 'architecture' | 'capabilities'
export type ModelFilterState = Partial<Record<ModelFilterKey, string[]>>
export type FilterOption = { value: string; count: number }
type ModelProviderRule = { label: string; prefixes: string[] }
type ModelFamilyRule = { label: string; patterns: RegExp[] }
type ModelLicenseRule = { label: string; patterns: RegExp[] }

const MODEL_PROVIDER_RULES: ModelProviderRule[] = [
  { label: 'Cohere', prefixes: ['command', 'cohere'] },
  { label: 'Z.ai', prefixes: ['glm', 'zai', 'z.ai', 'zhipu'] },
  { label: 'OpenAI', prefixes: ['gpt-oss'] },
  { label: 'StepFun', prefixes: ['step'] },
  { label: 'Google', prefixes: ['gemma'] },
  { label: 'Alibaba', prefixes: ['qwen', 'qwq'] },
  { label: 'Nvidia', prefixes: ['nvidia', 'nemotron'] },
  { label: 'Ant Group', prefixes: ['ling'] },
  { label: 'MiniMax', prefixes: ['minimax'] },
  { label: 'Meta', prefixes: ['llama'] },
  { label: 'Mistral AI', prefixes: ['mistral', 'mixtral'] },
  { label: 'Microsoft', prefixes: ['phi'] },
  { label: 'Community', prefixes: ['llava'] }
]

const MODEL_FAMILY_RULES: ModelFamilyRule[] = [
  {
    label: 'DeepSeek',
    patterns: [/\b(?:deepseek|ds[-_\s]?r\d+)(?:[-_\s]?(?:r\d+|v\d+(?:\.\d+)*|coder|vl|ocr|distill))?\b/i]
  },
  { label: 'Qwen', patterns: [/\b(?:qwen(?:\d+(?:\.\d+)*)?|qwq)\b/i] },
  { label: 'GPT-OSS', patterns: [/\bgpt[-_\s]?oss(?:[-_\s]?\d+b?)?\b/i] },
  { label: 'GLM', patterns: [/\b(?:chat[-_\s]?)?glm(?:[-_\s]?\d+(?:\.\d+)*v?)?\b/i] },
  { label: 'Nemotron', patterns: [/\bnemotron\b/i] },
  { label: 'Llama', patterns: [/\b(?:meta[-_\s]?)?llama(?:[-_\s]?\d+(?:\.\d+)*)?\b/i, /\bllava\b/i] },
  { label: 'Mixtral', patterns: [/\bmixtral\b/i] },
  { label: 'Mistral', patterns: [/\bmistral\b/i] },
  { label: 'Gemma', patterns: [/\bgemma(?:[-_\s]?\d+(?:\.\d+)*n?)?\b/i] },
  { label: 'Phi', patterns: [/\bphi(?:[-_\s]?\d+(?:\.\d+)*)?\b/i] },
  { label: 'MiniMax', patterns: [/\bminimax\b/i] },
  { label: 'Command', patterns: [/\bcommand[-_\s]?[ar]?\b/i] },
  { label: 'Step', patterns: [/\bstep(?:[-_\s]?\d+(?:\.\d+)*)?\b/i] },
  { label: 'Ling', patterns: [/\bling(?:[-_\s]?[a-z0-9.]+)?\b/i] },
  { label: 'Granite', patterns: [/\bgranite\b/i] },
  { label: 'StarCoder', patterns: [/\bstarcoder(?:\d+(?:\.\d+)*)?\b/i] },
  { label: 'InternLM', patterns: [/\binternlm(?:\d+(?:\.\d+)*)?\b/i] },
  { label: 'Yi', patterns: [/\byi(?:[-_\s]?\d+(?:\.\d+)*)?\b/i] }
]

const MODEL_LICENSE_RULES: ModelLicenseRule[] = [
  { label: 'CC-BY-NC-4.0', patterns: [/\bcommand[-_\s]?a\b/i, /\bcohere\b/i] },
  { label: 'MIT', patterns: [/\bglm(?:[-_\s]?\d+(?:\.\d+)*v?)?\b/i, /\bling(?:[-_\s]?[a-z0-9.]+)?\b/i] },
  {
    label: 'Apache 2.0',
    patterns: [/\bgpt[-_\s]?oss\b/i, /\bstep[-_\s]?\d+/i, /\bqwq\b/i, /\bqwen(?:\d+(?:\.\d+)*)?\b/i, /\bminimax\b/i]
  },
  { label: 'Nvidia Open Model', patterns: [/\bllama[-_\s]?3\.1[-_\s]?nemotron\b/i] },
  { label: 'Nvidia Open', patterns: [/\bnvidia[-_\s]?llama[-_\s]?3\.3[-_\s]?nemotron\b/i] },
  { label: 'Gemma', patterns: [/\bgemma\b/i] },
  { label: 'Llama 3.1 Community', patterns: [/\bllama[-_\s]?3\.1[-_\s]?405b\b/i] }
]

export const modelFilterColumns = [
  { key: 'provider', label: 'Provider' },
  { key: 'license', label: 'License' },
  { key: 'variant', label: 'Variant' },
  { key: 'status', label: 'Status' },
  { key: 'family', label: 'Family' },
  { key: 'architecture', label: 'Architecture' },
  { key: 'capabilities', label: 'Capabilities' }
] satisfies Array<{ key: ModelFilterKey; label: string }>

const CAPABILITY_ORDER = ['Text', 'Vision', 'Audio', 'TTS', 'Tools', 'Reasoning']
const NON_MODAL_CAPABILITY_KEYS = new Set(['moe'])

function normalizedTags(model: ModelSummary): Set<string> {
  return new Set(model.tags.map((tag) => tag.trim().toLowerCase()).filter(Boolean))
}

function hasAnyTag(tags: Set<string>, values: string[]): boolean {
  return values.some((value) => tags.has(value))
}

function hasCapability(model: ModelSummary, keys: string[]): boolean {
  return keys.some((key) => model.capabilities?.[key] === true)
}

function capabilityKeyLabel(key: string): string {
  const normalized = key.trim().toLowerCase()

  if (normalized === 'tts' || normalized === 'text_to_speech' || normalized === 'text-to-speech') return 'TTS'
  if (normalized === 'tool_use' || normalized === 'tools' || normalized === 'function_calling') return 'Tools'

  return normalized
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function modelIdentityText(model: ModelSummary): string {
  return [model.name, model.fullId, model.family].filter(Boolean).join(' ')
}

function lastPathSegment(value: string): string {
  return value.split('/').filter(Boolean).at(-1) ?? value
}

function titleCaseToken(value: string): string {
  if (!value) return value
  const knownCase: Record<string, string> = {
    glm: 'GLM',
    'gpt-oss': 'GPT-OSS',
    ling: 'Ling',
    minimax: 'MiniMax',
    nemotron: 'Nemotron',
    qwen: 'Qwen',
    phi: 'Phi'
  }
  const lower = value.toLowerCase()
  return knownCase[lower] ?? value.charAt(0).toUpperCase() + value.slice(1).toLowerCase()
}

function fallbackFamilyLabel(model: ModelSummary): string {
  const rawCandidate = [model.family, model.name, model.fullId]
    .map((value) => value?.trim())
    .find((value) => value && value.toLowerCase() !== 'unknown')

  if (!rawCandidate) return '—'

  const segment = lastPathSegment(rawCandidate)
  const firstToken = segment.split(/[-_\s]+/).find(Boolean) ?? segment
  const familyToken = firstToken.replace(/\d+(?:\.\d+)*$/u, '') || firstToken

  return titleCaseToken(familyToken)
}

export function modelFamilyLabel(model: ModelSummary): string {
  const identity = modelIdentityText(model)
  const matchingRule = MODEL_FAMILY_RULES.find((rule) => rule.patterns.some((pattern) => pattern.test(identity)))

  return matchingRule?.label ?? fallbackFamilyLabel(model)
}

export function modelCapabilityLabels(model: ModelSummary): string[] {
  const labels = new Set<string>(['Text'])
  const tags = normalizedTags(model)

  if (model.vision || hasCapability(model, ['vision']) || hasAnyTag(tags, ['vision', 'image', 'multimodal', 'vl'])) {
    labels.add('Vision')
  }

  if (hasCapability(model, ['audio']) || hasAnyTag(tags, ['audio', 'speech', 'voice'])) {
    labels.add('Audio')
  }

  if (
    hasCapability(model, ['tts', 'text_to_speech', 'text-to-speech', 'speech_synthesis']) ||
    hasAnyTag(tags, ['tts', 'text-to-speech', 'text_to_speech', 'speech-synthesis'])
  ) {
    labels.add('TTS')
  }

  if (
    hasCapability(model, ['tool_use', 'tools', 'function_calling']) ||
    hasAnyTag(tags, ['tool_use', 'tools', 'function-calling', 'function_calling'])
  ) {
    labels.add('Tools')
  }

  if (hasCapability(model, ['reasoning']) || hasAnyTag(tags, ['reasoning'])) {
    labels.add('Reasoning')
  }

  for (const [key, enabled] of Object.entries(model.capabilities ?? {})) {
    if (enabled !== true || NON_MODAL_CAPABILITY_KEYS.has(key)) continue
    labels.add(capabilityKeyLabel(key))
  }

  return [...labels].sort((a, b) => {
    const orderA = CAPABILITY_ORDER.indexOf(a)
    const orderB = CAPABILITY_ORDER.indexOf(b)
    if (orderA >= 0 && orderB >= 0) return orderA - orderB
    if (orderA >= 0) return -1
    if (orderB >= 0) return 1
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
  })
}

function modelMatchesProvider(candidate: string, prefix: string): boolean {
  const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&')
  return new RegExp(`(^|[/\\-_\\s.:])${escapedPrefix}($|[/\\-_\\s.:])`, 'iu').test(candidate)
}

export function modelProviderLabel(model: ModelSummary): string {
  const candidates = [model.family, model.name, model.fullId, modelFamilyLabel(model)]
    .map((value) => value?.trim().toLowerCase())
    .filter((value): value is string => Boolean(value))

  return (
    MODEL_PROVIDER_RULES.find((provider) =>
      provider.prefixes.some((prefix) => candidates.some((candidate) => modelMatchesProvider(candidate, prefix)))
    )?.label ?? 'Unknown'
  )
}

export function modelLicenseLabel(model: ModelSummary): string {
  const explicitLicense = model.license?.trim()
  if (explicitLicense) return explicitLicense

  const identity = modelIdentityText(model)
  const matchingRule = MODEL_LICENSE_RULES.find((rule) => rule.patterns.some((pattern) => pattern.test(identity)))

  return matchingRule?.label ?? 'Unknown'
}

export function modelVariantLabel(model: ModelSummary): string {
  const parts =
    model.fullId
      ?.split('/')
      .map((part) => part.trim())
      .filter(Boolean) ?? []
  const publisher = parts.length > 1 ? parts[0] : undefined

  return publisher ? titleCaseToken(publisher) : '—'
}

export function modelFilterValue(model: ModelSummary, key: ModelFilterKey): string {
  if (key === 'provider') return modelProviderLabel(model)
  if (key === 'license') return modelLicenseLabel(model)
  if (key === 'variant') return modelVariantLabel(model)
  if (key === 'status') return model.status.charAt(0).toUpperCase() + model.status.slice(1)
  if (key === 'family') return modelFamilyLabel(model)
  if (key === 'architecture') return model.moe ? 'MoE' : 'Dense'
  return modelCapabilityLabels(model)[0] ?? 'Text'
}

export function modelFilterValues(model: ModelSummary, key: ModelFilterKey): string[] {
  if (key === 'capabilities') return modelCapabilityLabels(model)
  return [modelFilterValue(model, key)]
}

export function buildModelFilterOptions(models: ModelSummary[], key: ModelFilterKey): FilterOption[] {
  const counts = new Map<string, number>()

  for (const model of models) {
    for (const value of new Set(modelFilterValues(model, key))) {
      counts.set(value, (counts.get(value) ?? 0) + 1)
    }
  }

  return [...counts.entries()]
    .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
    .map(([value, count]) => ({ value, count }))
}

export function selectedModelFilterSet(
  filters: ModelFilterState,
  key: ModelFilterKey,
  options: FilterOption[]
): Set<string> {
  const optionValues = options.map((option) => option.value)
  const selected = filters[key]

  if (!selected) return new Set(optionValues)

  return new Set(selected.filter((value) => optionValues.includes(value)))
}

export function isModelColumnFiltered(
  filters: ModelFilterState,
  key: ModelFilterKey,
  options: FilterOption[]
): boolean {
  if (options.length === 0) return false
  return selectedModelFilterSet(filters, key, options).size < options.length
}

export function modelFilterOptionLabel(value: string): string {
  return value === '—' ? 'None' : value
}
