import { CFG_CATALOG } from '@/features/app-tabs/data'
import { isUnifiedMemoryNode } from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode, ConfigurationDefaultsHarnessData, ConfigurationDefaultsSetting, ConfigurationDefaultsValues } from '@/features/app-tabs/types'

type BuildTomlOptions = {
  defaults?: ConfigurationDefaultsHarnessData
  defaultsValues?: ConfigurationDefaultsValues
}

const draftModelModeSettingId = 'speculation-mode'
const draftModelModeValue = 'draft_model'
const incompatiblePairingBehaviorSettingId = 'incompatible-pairing-behavior'

function tomlString(value: string): string {
  return JSON.stringify(value)
}

function tomlScalar(value: string): string {
  return /^\d+(\.\d+)?$/.test(value) ? value : tomlString(value)
}

function isBooleanToggleChoice(setting: ConfigurationDefaultsSetting): boolean {
  return setting.control.kind === 'choice'
    && setting.control.presentation === 'toggle'
    && setting.control.options.length === 2
    && setting.control.options.every((option) => option.value === 'on' || option.value === 'off')
}

function defaultTomlScalar(setting: ConfigurationDefaultsSetting, value: string): string {
  if (isBooleanToggleChoice(setting)) return value === 'on' ? 'true' : 'false'
  if (setting.id === 'memory-margin') return Number(value).toFixed(1)
  if (setting.id === 'draft-acceptance-threshold') return Number(value).toFixed(2)
  if (setting.id === 'repeat-penalty') return Number(value).toFixed(2)
  return tomlScalar(value)
}

function defaultSectionName(setting: ConfigurationDefaultsSetting): string | null {
  if (setting.categoryId === 'advanced') return 'runtime'
  if (setting.categoryId === 'speculative-decoding') return 'speculative_decoding'
  return null
}

function tomlKey(value: string): string {
  return value.replaceAll('-', '_')
}

function defaultValue(setting: ConfigurationDefaultsSetting, values: ConfigurationDefaultsValues | undefined): string {
  return values?.[setting.id] ?? setting.control.value
}

function shouldEmitDefaultSetting(setting: ConfigurationDefaultsSetting, settings: readonly ConfigurationDefaultsSetting[], values: ConfigurationDefaultsValues | undefined): boolean {
  if (setting.id !== incompatiblePairingBehaviorSettingId) return true

  const speculationModeSetting = settings.find((item) => item.id === draftModelModeSettingId)
  return speculationModeSetting ? defaultValue(speculationModeSetting, values) === draftModelModeValue : false
}

export function buildTOML(nodes: ConfigNode[], assigns: ConfigAssign[], models: ConfigModel[] = CFG_CATALOG, options: BuildTomlOptions = {}): string {
  const localNode = nodes[0]
  const lines: string[] = ['# MeshLLM generated local node config', '# Remote nodes are read-only context and are not written from this page.', '']

  if (options.defaults) {
    lines.push('[defaults]')
    const sectionSettings = new Map<string, ConfigurationDefaultsSetting[]>()

    for (const setting of options.defaults.settings) {
      if (!shouldEmitDefaultSetting(setting, options.defaults.settings, options.defaultsValues)) continue

      const sectionName = defaultSectionName(setting)
      if (sectionName) {
        sectionSettings.set(sectionName, [...(sectionSettings.get(sectionName) ?? []), setting])
        continue
      }

      const value = defaultValue(setting, options.defaultsValues)
      if (setting.control.kind === 'text' && value.trim().length === 0) continue
      const key = setting.control.kind === 'metric' ? setting.id : setting.control.name
      lines.push(`${tomlKey(key)} = ${defaultTomlScalar(setting, value)}`)
    }

    for (const [sectionName, settings] of sectionSettings) {
      lines.push('', `[defaults.${sectionName}]`)
      for (const setting of settings) {
        if (!shouldEmitDefaultSetting(setting, options.defaults.settings, options.defaultsValues)) continue

        const value = defaultValue(setting, options.defaultsValues)
        const key = setting.control.kind === 'metric' ? setting.id : setting.control.name
        lines.push(`${tomlKey(key)} = ${defaultTomlScalar(setting, value)}`)
      }
    }
    lines.push('')
  }

  if (!localNode) return lines.join('\n').trimEnd()

  lines.push('[node]', `id = ${tomlString(localNode.id)}`, `hostname = ${tomlString(localNode.hostname)}`, `region = ${tomlString(localNode.region)}`, `placement = ${tomlString(localNode.placement)}`)
  const emitsGpuIndex = localNode.placement === 'separate' && !isUnifiedMemoryNode(localNode)
  for (const assign of assigns.filter((item) => item.nodeId === localNode.id)) {
    const model = models.find((catalogModel) => catalogModel.id === assign.modelId)
    lines.push('', '[[models]]', `  id = ${tomlString(assign.id)}`, `  model = ${tomlString(model?.name ?? assign.modelId)}`)
    if (emitsGpuIndex) lines.push(`  gpu_index = ${assign.containerIdx}`)
    lines.push(`  ctx = ${assign.ctx}`)
  }

  return lines.join('\n').trimEnd()
}
