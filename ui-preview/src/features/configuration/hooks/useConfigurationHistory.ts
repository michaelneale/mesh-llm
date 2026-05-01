import { useCallback, useMemo, useState, type SetStateAction } from 'react'
import { containerUsedGB, contextGB, findModel, nodeTotalGB } from '@/features/configuration/lib/config-math'
import { CFG_NODES, INITIAL_ASSIGNS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigModel, ConfigNode, ConfigurationDefaultsValues } from '@/features/app-tabs/types'

export type SeparatePlacementSnapshot = Record<string, number>
export type SeparatePlacementSnapshots = Record<string, SeparatePlacementSnapshot>
export type ConfigurationState = {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  defaultsValues: ConfigurationDefaultsValues
  separatePlacementSnapshots: SeparatePlacementSnapshots
}
type ConfigurationHistoryState = { entries: ConfigurationState[]; index: number }

export function cloneConfigurationState(configuration: ConfigurationState): ConfigurationState {
  return {
    nodes: configuration.nodes.map((node) => ({ ...node, gpus: node.gpus.map((gpu) => ({ ...gpu })) })),
    assigns: configuration.assigns.map((assign) => ({ ...assign })),
    defaultsValues: { ...configuration.defaultsValues },
    separatePlacementSnapshots: Object.fromEntries(Object.entries(configuration.separatePlacementSnapshots).map(([nodeId, snapshot]) => [nodeId, { ...snapshot }])),
  }
}

export function createInitialConfigurationState(nodes: ConfigNode[] = CFG_NODES, assigns: ConfigAssign[] = INITIAL_ASSIGNS, defaultsValues: ConfigurationDefaultsValues = {}): ConfigurationState {
  return cloneConfigurationState({ nodes, assigns, defaultsValues, separatePlacementSnapshots: {} })
}

export function createConfigurationHistory(configuration: ConfigurationState): ConfigurationHistoryState {
  return { entries: [cloneConfigurationState(configuration)], index: 0 }
}

function createInitialConfigurationHistory(configuration = createInitialConfigurationState()): ConfigurationHistoryState {
  return createConfigurationHistory(configuration)
}

export function getPreferredConfigurationSelection(configuration: ConfigurationState, preferredId = 'a2'): { assignId: string | null; nodeId: string | null } {
  const assign = configuration.assigns.find((item) => item.id === preferredId) ?? configuration.assigns[0] ?? null
  return { assignId: assign?.id ?? null, nodeId: assign?.nodeId ?? configuration.nodes[0]?.id ?? null }
}

export function createConfigurationSnapshot(nodes: ConfigNode[], assigns: ConfigAssign[], defaultsValues: ConfigurationDefaultsValues = {}) {
  return JSON.stringify({ nodes, assigns, defaultsValues })
}

export function hasInvalidAllocation(nodes: ConfigNode[], assigns: ConfigAssign[], models?: ConfigModel[]): boolean {
  return assigns.some((assign) => {
    const node = nodes.find((item) => item.id === assign.nodeId)
    const model = findModel(assign.modelId, models)
    if (!node || !model) return false

    const containerIdx = node.placement === 'pooled' ? 0 : assign.containerIdx
    const totalGB = node.placement === 'pooled' ? nodeTotalGB(node) : (node.gpus.find((gpu) => gpu.idx === containerIdx)?.totalGB ?? 0)
    const reservedGB = node.placement === 'pooled'
      ? node.gpus.reduce((sum, gpu) => sum + (gpu.reservedGB ?? 0), 0)
      : (node.gpus.find((gpu) => gpu.idx === containerIdx)?.reservedGB ?? 0)
    const usedByOtherAssignmentsGB = containerUsedGB(assigns.filter((item) => item.id !== assign.id), node.id, containerIdx, models)
    const freeForContextGB = totalGB - reservedGB - usedByOtherAssignmentsGB - model.sizeGB

    return freeForContextGB < 0 || contextGB(model, assign.ctx) > Math.max(0, freeForContextGB)
  })
}

export function useConfigurationHistory(initialConfiguration = createInitialConfigurationState()) {
  const [configurationHistory, setConfigurationHistory] = useState<ConfigurationHistoryState>(() => createInitialConfigurationHistory(initialConfiguration))
  const history = configurationHistory.entries
  const index = configurationHistory.index
  const configuration = useMemo(() => history[index] ?? history.at(-1) ?? initialConfiguration, [history, index, initialConfiguration])

  const updateConfiguration = useCallback((updater: (current: ConfigurationState) => ConfigurationState) => {
    setConfigurationHistory((state) => {
      const current = state.entries[state.index] ?? state.entries.at(-1) ?? initialConfiguration
      const next = updater(current)
      if (createConfigurationSnapshot(next.nodes, next.assigns, next.defaultsValues) === createConfigurationSnapshot(current.nodes, current.assigns, current.defaultsValues)) return state

      const entries = [...state.entries.slice(0, state.index + 1), next]
      return { entries, index: entries.length - 1 }
    })
  }, [initialConfiguration])

  const setAssigns = useCallback((updater: SetStateAction<ConfigAssign[]>) => {
    updateConfiguration((current) => {
      const nextAssigns = typeof updater === 'function' ? updater(current.assigns) : updater
      return { ...current, assigns: nextAssigns }
    })
  }, [updateConfiguration])

  const resetConfiguration = useCallback((configuration: ConfigurationState) => {
    setConfigurationHistory(createConfigurationHistory(configuration))
  }, [])

  const undoConfigurationChange = useCallback(() => {
    setConfigurationHistory((state) => ({ ...state, index: Math.max(0, state.index - 1) }))
  }, [])

  const redoConfigurationChange = useCallback(() => {
    setConfigurationHistory((state) => ({ ...state, index: Math.min(state.entries.length - 1, state.index + 1) }))
  }, [])

  return {
    configuration,
    history,
    index,
    canUndo: index > 0,
    canRedo: index < history.length - 1,
    updateConfiguration,
    setAssigns,
    resetConfiguration,
    undoConfigurationChange,
    redoConfigurationChange,
  }
}
