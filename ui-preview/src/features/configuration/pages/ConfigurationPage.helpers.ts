import {
  findModel,
  findPreferredModelFitContainerIdx,
  containerAssigns
} from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'
import type { SeparatePlacementSnapshot } from '@/features/configuration/hooks/useConfigurationHistory'

export type CatalogTarget = { nodeId: string; preferredContainerIdx: number | null }
export type SelectedContainerTarget = { nodeId: string; containerIdx: number }

export function isTextEditingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true

  return (
    target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement
  )
}

export function getNodeTargetContainerIdx(node: ConfigNode, containerIdx: number): number {
  return node.placement === 'pooled' ? 0 : containerIdx
}

export function nodeAssignmentIds(node: ConfigNode, assigns: ConfigAssign[]): string[] {
  if (node.placement === 'pooled') return containerAssigns(assigns, node.id, 0).map((assign) => assign.id)

  return node.gpus.flatMap((gpu) => containerAssigns(assigns, node.id, gpu.idx).map((assign) => assign.id))
}

export function createSeparatePlacementSnapshot(assigns: ConfigAssign[], nodeId: string): SeparatePlacementSnapshot {
  return assigns.reduce<SeparatePlacementSnapshot>((snapshot, assign) => {
    if (assign.nodeId !== nodeId) return snapshot
    return { ...snapshot, [assign.id]: assign.containerIdx }
  }, {})
}

export function restoreSeparatePlacement(
  assigns: ConfigAssign[],
  node: ConfigNode,
  snapshot: SeparatePlacementSnapshot,
  models?: ConfigModel[]
): ConfigAssign[] {
  const separateNode: ConfigNode = { ...node, placement: 'separate' }
  const otherAssigns = assigns.filter((assign) => assign.nodeId !== node.id)
  const nodeAssigns = assigns.filter((assign) => assign.nodeId === node.id)
  const snapshottedAssigns = nodeAssigns.filter((assign) => typeof snapshot[assign.id] === 'number')
  const pooledAssigns = nodeAssigns.filter((assign) => typeof snapshot[assign.id] !== 'number')
  const placementByAssignId = new Map<string, number>()
  const restoredForFit = [...otherAssigns]

  const restoreAssign = (assign: ConfigAssign, preferredContainerIdx: number | null) => {
    const model = findModel(assign.modelId, models)
    const containerIdx = model
      ? (findPreferredModelFitContainerIdx(
          model,
          separateNode,
          restoredForFit,
          preferredContainerIdx,
          assign.ctx,
          models
        ) ??
        preferredContainerIdx ??
        assign.containerIdx)
      : (preferredContainerIdx ?? assign.containerIdx)

    placementByAssignId.set(assign.id, containerIdx)
    restoredForFit.push({ ...assign, containerIdx })
  }

  snapshottedAssigns.forEach((assign) => {
    restoreAssign(assign, snapshot[assign.id])
  })
  pooledAssigns.forEach((assign) => {
    restoreAssign(assign, null)
  })

  return assigns.map((assign) => {
    const containerIdx = placementByAssignId.get(assign.id)
    return typeof containerIdx === 'number' ? { ...assign, containerIdx } : assign
  })
}
