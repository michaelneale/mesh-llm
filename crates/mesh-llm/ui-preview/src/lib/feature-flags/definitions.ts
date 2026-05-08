import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'

export type FeatureFlagGroup = {
  [featureName: string]: boolean | FeatureFlagGroup
}

export const DEFAULT_FEATURE_FLAGS = {
  global: {
    newConfigurationPage: false
  },
  configuration: {
    signingAttestation: false,
    integrations: false
  },
  chat: {
    systemPromptButton: false,
    transparencyTab: false
  }
} as const satisfies FeatureFlagGroup

export type FeatureFlagSectionId = 'global' | 'configuration' | 'chat'
export type FeatureFlagPath =
  | 'global/newConfigurationPage'
  | 'configuration/signingAttestation'
  | 'configuration/integrations'
  | 'chat/systemPromptButton'
  | 'chat/transparencyTab'

export type FeatureFlagDefinition = {
  sectionId: FeatureFlagSectionId
  key: string
  path: FeatureFlagPath
  label: string
  description: string
}

export type FeatureFlagSectionDefinition = {
  id: FeatureFlagSectionId
  label: string
  description: string
  flags: readonly FeatureFlagDefinition[]
}

export const FEATURE_FLAG_SECTIONS: readonly FeatureFlagSectionDefinition[] = [
  {
    id: 'global',
    label: 'Global',
    description: 'App-wide rollout switches that affect navigation, routes, and shared surfaces.',
    flags: [
      {
        sectionId: 'global',
        key: 'newConfigurationPage',
        path: 'global/newConfigurationPage',
        label: 'New configuration page',
        description: 'Shows the Configuration app surface while the new configuration flow is being rolled out.'
      }
    ]
  },
  {
    id: 'configuration',
    label: 'Configuration',
    description: 'Rollout switches for configuration sections that are still using temporary surfaces.',
    flags: [
      {
        sectionId: 'configuration',
        key: 'signingAttestation',
        path: 'configuration/signingAttestation',
        label: 'Signing / Attestation',
        description:
          'Shows the temporary signing, key binding, attestation receipt, and dirty-state checks section in Configuration.'
      },
      {
        sectionId: 'configuration',
        key: 'integrations',
        path: 'configuration/integrations',
        label: 'Integrations',
        description:
          'Shows the temporary integrations section in Configuration while integration management is being designed.'
      }
    ]
  },
  {
    id: 'chat',
    label: 'Chat',
    description: 'Rollout switches for chat experiences that are still being validated.',
    flags: [
      {
        sectionId: 'chat',
        key: 'systemPromptButton',
        path: 'chat/systemPromptButton',
        label: 'System prompt button',
        description: 'Shows the chat composer control for saving a mesh-llm system prompt.'
      },
      {
        sectionId: 'chat',
        key: 'transparencyTab',
        path: 'chat/transparencyTab',
        label: 'Transparency tab',
        description: 'Shows the chat sidebar transparency tab and per-message transparency inspection controls.'
      }
    ]
  }
]

export type FeatureFlagOverrides = Partial<Record<FeatureFlagSectionId, Record<string, boolean>>>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isEmptyFeatureFlagOverrides(overrides: FeatureFlagOverrides) {
  return Object.values(overrides).every(
    (sectionOverrides) => !sectionOverrides || Object.keys(sectionOverrides).length === 0
  )
}

export function findFeatureFlagDefinition(path: FeatureFlagPath) {
  for (const section of FEATURE_FLAG_SECTIONS) {
    const definition = section.flags.find((flag) => flag.path === path)
    if (definition) return definition
  }

  return undefined
}

export function getBaseFeatureFlagValue(path: FeatureFlagPath): boolean {
  switch (path) {
    case 'global/newConfigurationPage':
      return DEFAULT_FEATURE_FLAGS.global.newConfigurationPage
    case 'configuration/signingAttestation':
      return DEFAULT_FEATURE_FLAGS.configuration.signingAttestation
    case 'configuration/integrations':
      return DEFAULT_FEATURE_FLAGS.configuration.integrations
    case 'chat/systemPromptButton':
      return DEFAULT_FEATURE_FLAGS.chat.systemPromptButton
    case 'chat/transparencyTab':
      return DEFAULT_FEATURE_FLAGS.chat.transparencyTab
  }
}

export function normalizeFeatureFlagOverrides(value: unknown): FeatureFlagOverrides {
  if (!isRecord(value)) return {}

  const normalized: FeatureFlagOverrides = {}

  for (const section of FEATURE_FLAG_SECTIONS) {
    const rawSection = value[section.id]
    if (!isRecord(rawSection)) continue

    for (const flag of section.flags) {
      const rawFlagValue = rawSection[flag.key]
      if (typeof rawFlagValue !== 'boolean') continue
      if (rawFlagValue === getBaseFeatureFlagValue(flag.path)) continue

      normalized[section.id] = {
        ...(normalized[section.id] ?? {}),
        [flag.key]: rawFlagValue
      }
    }
  }

  return normalized
}

export function readStoredFeatureFlagOverrides(
  storageKey = APP_STORAGE_KEYS.featureFlagOverrides
): FeatureFlagOverrides {
  if (typeof window === 'undefined') return {}

  try {
    const storedValue = window.localStorage.getItem(storageKey)
    return storedValue ? normalizeFeatureFlagOverrides(JSON.parse(storedValue)) : {}
  } catch {
    return {}
  }
}

export function writeStoredFeatureFlagOverrides(overrides: FeatureFlagOverrides, storageKey: string): void {
  if (typeof window === 'undefined') return

  try {
    if (isEmptyFeatureFlagOverrides(overrides)) {
      window.localStorage.removeItem(storageKey)
      return
    }

    window.localStorage.setItem(storageKey, JSON.stringify(overrides))
  } catch {
    return
  }
}

export function resolveFeatureFlags(overrides: FeatureFlagOverrides): FeatureFlagGroup {
  const flags: FeatureFlagGroup = {}

  for (const section of FEATURE_FLAG_SECTIONS) {
    const sectionFlags: FeatureFlagGroup = {}

    for (const flag of section.flags) {
      sectionFlags[flag.key] = overrides[section.id]?.[flag.key] ?? getBaseFeatureFlagValue(flag.path)
    }

    flags[section.id] = sectionFlags
  }

  return flags
}

export function removeFeatureFlagOverride(
  overrides: FeatureFlagOverrides,
  sectionId: FeatureFlagSectionId,
  key: string
): FeatureFlagOverrides {
  const sectionOverrides = { ...(overrides[sectionId] ?? {}) }
  delete sectionOverrides[key]

  const nextOverrides = { ...overrides }
  if (Object.keys(sectionOverrides).length === 0) delete nextOverrides[sectionId]
  else nextOverrides[sectionId] = sectionOverrides

  return nextOverrides
}
