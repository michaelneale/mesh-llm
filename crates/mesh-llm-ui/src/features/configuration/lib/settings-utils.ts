import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import type { ConfigurationDefaultsSetting, ConfigurationDefaultsValues } from '@/features/app-tabs/types'

const canonicalDefaultsById = new Map<string, ConfigurationDefaultsSetting>(
  CONFIGURATION_DEFAULTS.settings.map((setting) => [setting.id, setting] as const)
)

export type SettingDependencyStatus = {
  disabled: boolean
  reason?: string
}

export function getSettingValue(setting: ConfigurationDefaultsSetting, values: ConfigurationDefaultsValues) {
  return values[setting.id] ?? setting.control.value
}

export function getSettingBaselineValue(setting: ConfigurationDefaultsSetting) {
  return canonicalDefaultsById.get(setting.id)?.control.value ?? setting.control.value
}

function dependencyRequirementValue(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  const dependency = setting.dependsOn
  if (!dependency) return undefined

  const parentSetting = allSettings.find((item) => item.id === dependency.settingId)
  if (!parentSetting) return undefined

  const currentValue = getSettingValue(parentSetting, values)
  if (dependency.condition(currentValue)) return currentValue

  if (parentSetting.control.kind === 'choice') {
    const matchingOptions = parentSetting.control.options.filter((option) => dependency.condition(option.value))
    if (matchingOptions.length > 0) return matchingOptions.map((option) => option.value)
  }

  return undefined
}

function formatRequirementValue(value: string | readonly string[]) {
  return Array.isArray(value) ? value.join(' or ') : value
}

export function getSettingDependencyStatus(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
): SettingDependencyStatus {
  if (!setting.dependsOn) return { disabled: false }

  const parentSetting = allSettings.find((item) => item.id === setting.dependsOn?.settingId)
  if (!parentSetting) {
    return { disabled: true, reason: `Requires ${setting.dependsOn.settingId}` }
  }

  const parentValue = getSettingValue(parentSetting, values)
  if (setting.dependsOn.condition(parentValue)) return { disabled: false }

  const requiredValue = dependencyRequirementValue(setting, allSettings, values)
  if (requiredValue !== undefined) {
    return { disabled: true, reason: `Requires ${parentSetting.id} = ${formatRequirementValue(requiredValue)}` }
  }

  return { disabled: true, reason: `Requires ${parentSetting.id} to satisfy its dependency` }
}

export function isSettingDisabled(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  return getSettingDependencyStatus(setting, allSettings, values).disabled
}

export function getSettingDisabledReason(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  return getSettingDependencyStatus(setting, allSettings, values).reason
}
