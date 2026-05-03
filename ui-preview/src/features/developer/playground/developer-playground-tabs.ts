export const DEFAULT_DEVELOPER_PLAYGROUND_TAB = 'shell-controls'

export const DEVELOPER_PLAYGROUND_TAB_IDS = [
  DEFAULT_DEVELOPER_PLAYGROUND_TAB,
  'data-display',
  'chat-components',
  'configuration-controls',
  'tokens-foundations',
  'feature-flags',
  'meshviz-perf',
] as const

export type DeveloperPlaygroundTabId = typeof DEVELOPER_PLAYGROUND_TAB_IDS[number]

export type DeveloperPlaygroundSearch = {
  tab: DeveloperPlaygroundTabId
}

export function isDeveloperPlaygroundTabId(value: unknown): value is DeveloperPlaygroundTabId {
  return typeof value === 'string' && DEVELOPER_PLAYGROUND_TAB_IDS.some((tabId) => tabId === value)
}

export function parseDeveloperPlaygroundSearch(search: Record<string, unknown>): DeveloperPlaygroundSearch {
  return {
    tab: isDeveloperPlaygroundTabId(search.tab) ? search.tab : DEFAULT_DEVELOPER_PLAYGROUND_TAB,
  }
}
