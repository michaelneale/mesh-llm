export type ConfigurationTabId = 'defaults' | 'local-deployment' | 'signing' | 'integrations' | 'toml-review'

export const CONFIGURATION_TAB_IDS = ['defaults', 'local-deployment', 'signing', 'integrations', 'toml-review'] as const satisfies readonly ConfigurationTabId[]

export function isConfigurationTabId(value: string | undefined): value is ConfigurationTabId {
  return value !== undefined && CONFIGURATION_TAB_IDS.some((tabId) => tabId === value)
}
