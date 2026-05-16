import { CFG_CATALOG } from '@/features/app-tabs/data'
import { isUnifiedMemoryNode } from '@/features/configuration/lib/config-math'
import {
  getSettingBaselineValue,
  getSettingValue,
  isSettingDisabled
} from '@/features/configuration/lib/settings-utils'
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

const legacySectionPaths: Partial<Record<ConfigurationDefaultsSetting['categoryId'], string>> = {
  advanced: 'defaults.runtime',
  'speculative-decoding': 'defaults.speculative'
}

function defaultSectionPath(
  setting: ConfigurationDefaultsSetting,
  categoryById: ReadonlyMap<string, ConfigurationDefaultsHarnessData['categories'][number]>
) {
  return (
    setting.tomlSection ??
    categoryById.get(setting.categoryId)?.tomlSection ??
    legacySectionPaths[setting.categoryId] ??
    null
  )
}

function tomlKey(value: string): string {
  return value.replaceAll('-', '_')
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
    const sectionSettings = new Map<string, ConfigurationDefaultsSetting[]>()
    const defaultsValues = options.defaultsValues ?? {}
    const categoryById = new Map(options.defaults.categories.map((category) => [category.id, category] as const))

    for (const setting of options.defaults.settings) {
      if (isSettingDisabled(setting, options.defaults.settings, defaultsValues)) continue

      const value = getSettingValue(setting, defaultsValues)
      if (value === getSettingBaselineValue(setting)) continue
      if (setting.control.kind === 'text' && value.trim().length === 0) continue

      const sectionPath = defaultSectionPath(setting, categoryById)
      if (sectionPath) {
        sectionSettings.set(sectionPath, [...(sectionSettings.get(sectionPath) ?? []), setting])
        continue
      }

      const key = setting.control.kind === 'metric' ? setting.id : setting.control.name
      lines.push(`${tomlKey(key)} = ${defaultTomlScalar(setting, value)}`)
    }

    const orderedSectionPaths = [
      ...defaultSectionOrder.filter((sectionPath) => sectionSettings.has(sectionPath)),
      ...Array.from(sectionSettings.keys()).filter(
        (sectionPath) => !defaultSectionOrder.includes(sectionPath as ConfigurationTomlSectionId)
      )
    ]

    let emittedDefaultsSectionCount = 0
    for (const sectionPath of orderedSectionPaths) {
      const settings = sectionSettings.get(sectionPath)
      if (!settings) continue

      if (lines[lines.length - 1] !== '') lines.push('')
      if (emittedDefaultsSectionCount > 0) lines.push('')
      lines.push(`[${sectionPath}]`)
      for (const setting of settings) {
        const value = getSettingValue(setting, defaultsValues)
        const key = setting.tomlKey ?? (setting.control.kind === 'metric' ? setting.id : setting.control.name)
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
