import { CFG_CATALOG } from '@/features/app-tabs/data'
import { isUnifiedMemoryNode } from '@/features/configuration/lib/config-math'
import type {
  ConfigAssign,
  ConfigModel,
  ConfigNode,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationTomlSectionId
} from '@/features/app-tabs/types'

type BuildTomlOptions = {
  defaults?: ConfigurationDefaultsHarnessData
  defaultsValues?: ConfigurationDefaultsValues
}

const draftModelModeSettingId = 'speculation-mode'
const draftModelModeValue = 'draft'
const draftModelOnlySettingIds = new Set([
  'draft-selection-policy',
  'draft-max-tokens',
  'incompatible-pairing-behavior'
])
const defaultSectionOrder: readonly ConfigurationTomlSectionId[] = [
  'defaults.model_fit',
  'defaults.hardware',
  'defaults.throughput',
  'defaults.skippy',
  'defaults.speculative',
  'defaults.request_defaults',
  'defaults.multimodal',
  'defaults.advanced.server'
]

function tomlString(value: string): string {
  return JSON.stringify(value)
}

function tomlScalar(value: string): string {
  return /^\d+(\.\d+)?$/.test(value) ? value : tomlString(value)
}

function isBooleanToggleChoice(setting: ConfigurationDefaultsSetting): boolean {
  return (
    setting.control.kind === 'choice' &&
    setting.control.presentation === 'toggle' &&
    setting.control.options.length === 2 &&
    setting.control.options.every((option) => option.value === 'on' || option.value === 'off')
  )
}

function defaultTomlScalar(setting: ConfigurationDefaultsSetting, value: string): string {
  if (isBooleanToggleChoice(setting)) return value === 'on' ? 'true' : 'false'
  if (setting.id === 'memory-margin') return Number(value).toFixed(1)
  if (setting.id === 'temperature' || setting.id === 'top-p' || setting.id === 'repeat-penalty')
    return Number(value).toFixed(2)
  return tomlScalar(value)
}

function tomlKey(value: string): string {
  return value.replaceAll('-', '_')
}

function defaultValue(setting: ConfigurationDefaultsSetting, values: ConfigurationDefaultsValues | undefined): string {
  return values?.[setting.id] ?? setting.control.value
}

function shouldEmitDefaultSetting(
  setting: ConfigurationDefaultsSetting,
  settings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues | undefined
): boolean {
  const speculationModeSetting = settings.find((item) => item.id === draftModelModeSettingId)
  if (!draftModelOnlySettingIds.has(setting.id)) return true
  return speculationModeSetting ? defaultValue(speculationModeSetting, values) === draftModelModeValue : false
}

export function buildTOML(
  nodes: ConfigNode[],
  assigns: ConfigAssign[],
  models: ConfigModel[] = CFG_CATALOG,
  options: BuildTomlOptions = {}
): string {
  const localNode = nodes[0]
  const lines: string[] = [
    '# Mesh LLM generated config preview',
    '# Remote nodes are read-only context and are not serialized from this page.',
    'version = 1',
    ''
  ]

  if (options.defaults) {
    const sectionSettings = new Map<ConfigurationTomlSectionId, ConfigurationDefaultsSetting[]>()

    for (const setting of options.defaults.settings) {
      if (!shouldEmitDefaultSetting(setting, options.defaults.settings, options.defaultsValues)) continue
      sectionSettings.set(setting.tomlSection, [...(sectionSettings.get(setting.tomlSection) ?? []), setting])
    }

    let emittedDefaultsSectionCount = 0
    for (const sectionName of defaultSectionOrder) {
      const settings = sectionSettings.get(sectionName)
      if (!settings) continue

      if (emittedDefaultsSectionCount > 0) lines.push('')
      lines.push(`[${sectionName}]`)
      for (const setting of settings) {
        if (!shouldEmitDefaultSetting(setting, options.defaults.settings, options.defaultsValues)) continue

        const value = defaultValue(setting, options.defaultsValues)
        if (setting.control.kind === 'text' && value.trim().length === 0) continue

        const key = setting.tomlKey
        lines.push(`${tomlKey(key)} = ${defaultTomlScalar(setting, value)}`)
      }
      emittedDefaultsSectionCount += 1
    }
    lines.push('')
  }

  if (!localNode) return lines.join('\n').trimEnd()

  const emitsPerGpuDevice = localNode.placement === 'separate' && !isUnifiedMemoryNode(localNode)
  for (const assign of assigns.filter((item) => item.nodeId === localNode.id)) {
    const model = models.find((catalogModel) => catalogModel.id === assign.modelId)
    const sectionLines = new Map<string, string[]>()

    sectionLines.set('models.model_fit', [`ctx_size = ${assign.ctx}`])
    if (emitsPerGpuDevice) {
      sectionLines.set('models.hardware', [`device = ${tomlString(`cuda:${assign.containerIdx}`)}`, 'gpu_layers = -1'])
    }

    if (lines[lines.length - 1] !== '') lines.push('')
    lines.push('[[models]]', `model = ${tomlString(model?.name ?? assign.modelId)}`)

    for (const sectionName of ['models.model_fit', 'models.hardware']) {
      const values = sectionLines.get(sectionName)
      if (!values?.length) continue
      lines.push('', `[${sectionName}]`, ...values)
    }
  }

  return lines.join('\n').trimEnd()
}
